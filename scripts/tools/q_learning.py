import numpy as np
from scipy.optimize import minimize
from typing import Tuple
from tqdm import tqdm
import sys
from .simulation_tools import *


def Y_non_differentiable(predictions: np.ndarray, ll: float = 0.10, ul: float = 0.30) -> np.ndarray:
    """
    Function to measure how long subject stays in range.
    Returns booleans if observation is in range.
    """
    predictions_below_upper = predictions < ul
    predictions_above_lower = ll < predictions
    reward = (predictions_below_upper) & (predictions_above_lower)

    return reward


def Y(predictions: np.ndarray, ll: float = 0.10, ul: float = 0.30, beta: int = 5) -> np.ndarray:
    """
    Differentiable version of our loss.  Want concentrations to be between ll and ul for as long as possible.  This function gives approximately 0 loss
    for observations between ll and ul, and loss = 1 else.

    Inputs:
    predictions - Samples of predictions from the Bayesian PK model
    ll - Lower limit of desired threshold
    ul - Upper limit of desired threshold
    beta - Controls how close the approximation is to the loss we want.  Larger beta means better approximation

    Outputs:
    reward - How long, as a proportion, of the time we spend within range
    """
    if beta < 1:
        raise ValueError("beta must be a positive integer.")

    mid = (ll + ul) / 2
    radius = (ll - ul) / 2

    arg = 1.0 / radius * (predictions - mid)
    kernel = -1 * np.power(arg, 2 * beta)

    reward = np.exp(kernel)
    return reward


def Q_2(A_2: Tuple, S_2: Tuple) -> float:
    """
    Q-Function for second stage of experiment.

    Inputs:
    A_2 - Proposed dose, in mg.
    S_2 - Tuple containing estimate of concentration at the time the next dose is taken, and the dynamics under a dose of 1 mg with initial condition.

    Outputs:
    E_Y - Value of the action A_2 (proportion of time spent within range).
    """

    # S_2 is the state, which fully determines the prediction function, so just pass the prediction function
    # This cuts down on refitting time.
    initial_condition, dynamics = S_2
    proposed_dose = A_2

    # Negative because we are going to minimize
    # This should compute E(Y|A_2, S_2)
    E_Y = -1 * Y(initial_condition + proposed_dose * dynamics).mean()

    return E_Y


def get_next_dose_time(tobs, dose_times):

    # We observe at t=tobs.  When is the next time a dose is taken?  It would be the first positive element of dose_times-tobs.
    # Negative values in the past, positive values in the future.
    next_dose_time_after_tobs_ix = np.argwhere((dose_times - tobs) > 0).min()
    next_dose_time = dose_times[next_dose_time_after_tobs_ix]  # There is some problem here.  Json doesn't like numpy dtypes, so turn to float
    next_dose_time = float(next_dose_time)

    return next_dose_time


def stage_2_optimization(S_2):

    tobs, yobs, theta, dose_times, dose_size = S_2
    # Assuming we sampled the subject at tobs and got yobs, what would our posterior look like?
    # Fit the model to the observation (tobs, yobs) with dose schedule instantiated from Q_1
    predict = fit(t=tobs, y=yobs, theta=theta, dose_times=dose_times, dose_size=dose_size)

    # We will be splitting the time into two stages.  E.g 4 days on inital dose, 4 days after, etc.  Same number of days in two stages.
    # Likely be sampling near the end of the last day in stage 2.  This means I would need to predict over the same length of time.
    next_dose_time = get_next_dose_time(tobs, dose_times)

    decision_point = int(len(dose_times) / 2)
    tfin = dose_times[decision_point] + 0.5
    tpred = np.arange(0.5, tfin, 0.5)
    # Make predictions of the dynamics.  Estimate the inital_condition (latent concentration a next_dose_time_after_tobs) as well as the dynamics under a unit dose
    # That way, the total dynamics under a new dose us initial_condition + new_dose_size*dynamics.
    # This logic comes from solving the PK ode for an arbitrary inital condition.
    dose_of_1s = np.ones_like(dose_times)

    initial_condition, dynamics = predict(tpred=tpred, new_dose_times=dose_times, new_dose_size=dose_of_1s, c0_time=next_dose_time_after_tobs)
    # Can't give someone negative mg.  Bound the dose.
    dose_bnds = [(0, None)]
    # Pick a dose size to start with.  Remember to multiplt the initial condition by this dose size so that we are estimating the  concentration at tobs.
    D_old = np.unique(dose_size)
    optim = minimize(Q_2, x0=D_old, args=([D_old * initial_condition, dynamics]), bounds=dose_bnds, method="L-BFGS-B")
    # Policty is the argmax
    π_2 = optim.x[0]
    # Undo the negative we did in the objective.
    # Value is the max
    expected_V_2 = -1 * optim.fun

    return π_2, expected_V_2


def Q_1(A_1, S_1):

    # What time are we going to observe the subject?  What are their covariates, and what times are they taking their doses?
    tobs, theta, dose_times = S_1
    # Dose we are thinking about giving them
    proposed_dose = A_1
    # Create a dosing regiment.  Dose of size proposed_dose at every time in dose_times.
    dose_size = np.tile(proposed_dose, len(dose_times))

    # Given only what we know about the subject to date (i.e. the dose we want to give them, the times they are going to take that dose, and their covars)
    # What is the distribution of possible observed concentrations?
    # Sample from the prior predictive distribution in order to answer this.
    possible_futures = prior_predict(tobs, theta, dose_times, dose_size, with_noise=True)

    # To speed up the optimization, we bin the possible observed concentrations at tobs.
    # The number of samples in each bin will be used as weights
    counts, edges = np.histogram(possible_futures)
    centers = 0.5 * (edges[1:] + edges[:-1])
    # Laplace smoothing, just in case.  This should be unneccesary however.
    probabilities = (counts + 1) / sum(counts + 1)

    # Initialize the value for A_1
    expected_v_2 = 0
    for yobs, p in zip(centers, probabilities):

        S_2 = tobs, yobs, theta, dose_times, dose_size
        π_2, V_2 = stage_2_optimization(S_2)
        # Compute a weighted sum of V_2 weighted by the probability of falling in the bucket.
        # In essence integrating over a discretized space.
        expected_v_2 += V_2 * p

    # Now reward under this dose for stage 1.
    # Subject will only take the first dose until the halfway point.
    decision_point = int(len(dose_times) / 2)
    # Predict from t=0.5 to the decision point.
    step_size = 0.5
    tpred = np.arange(0.5, dose_times[decision_point] + step_size, step_size)
    # Now predict over this period and compute the reward.
    prior_predictions = prior_predict(tpred, theta, dose_times, dose_size)
    expected_v_1 = Y(prior_predictions).mean()

    return expected_v_1 + expected_v_2


def stage_1_optimization(S_1: Tuple, dose_max=2, dose_min=20, step=1):

    tobs, theta, dose_times = S_1
    # Need to think of a better way to do this.  Plug and play later
    # Approximate the max of Q_1 using a coarse grid
    coarse_dose_grid = np.arange(dose_max, dose_min, step)
    q_values = []
    for A_1 in tqdm(coarse_dose_grid, desc="Looping Over Inital Doses", file=sys.stdout):
        ev = Q_1(A_1, S_1)
        q_values.append(ev)

    best_dose = coarse_dose_grid[np.argmax(q_values)]

    return best_dose


def perform_q_learning(pk_params, num_days=10, doses_per_day=2, hours_per_dose=12, dose_step=1.5):

    # The last time the subejct could take a dose.  End simulation just before this time.
    tmax = hours_per_dose * doses_per_day * num_days
    # Spread doses out over time
    dose_times = np.arange(0, tmax, hours_per_dose)
    # We get to make a decision about the size of the dose at the half way mark
    decision_point = int(len(dose_times) / 2)

    # Set up times we would observe subejcts
    tobs_min = dose_times[decision_point - 1]
    tobs_max = dose_times[decision_point]

    best_starting_doses = []
    tobs_subjects = []

    for theta in tqdm(pk_params, desc="Looping Over Subjects", file=sys.stdout):

        tobs = np.random.uniform(tobs_min, tobs_max, size=(1,))

        S_1 = (tobs, theta, dose_times)
        best_dose = stage_1_optimization(S_1, step=dose_step)

        best_starting_doses.append(best_dose)
        tobs_subjects.append(tobs[0])

    return (tobs_subjects, best_starting_doses)
