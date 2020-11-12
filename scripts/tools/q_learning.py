import numpy as np
from scipy.optimize import minimize
from typing import Tuple

from .simulation_tools import *

def Y_non_differentiable(predictions: np.ndarray, ll: float=0.10, ul: float=.30) -> np.ndarray:
    '''
    Function to measure how long subject stays in range.
    Returns booleans if observation is in range.
    '''
    predictions_below_upper = predictions<ul
    predictions_above_lower = ll<predictions
    reward = (predictions_below_upper) & (predictions_above_lower)
    
    return reward

def Y(predictions: np.ndarray, ll: float =0.10, ul: float =.30, beta: int = 5) -> np.ndarray:
    '''
    Differentiable version of our loss.  Want concentrations to be between ll and ul for as long as possible.  This function gives approximately 0 loss
    for observations between ll and ul, and loss = 1 else.

    Inputs:
    predictions - Samples of predictions from the Bayesian PK model
    ll - Lower limit of desired threshold
    ul - Upper limit of desired threshold
    beta - Controls how close the approximation is to the loss we want.  Larger beta means better approximation

    Outputs:
    reward - How long, as a proportion, of the time we spend within range
    '''
    if beta<1:
        raise ValueError('beta must be a positive integer.')

    mid = (ll + ul)/2
    radius = (ll - ul)/2

    arg = 1.0/radius * (predictions-mid)
    kernel = -1 * np.power(arg, 2*beta)

    reward = np.exp(kernel)
    return reward


def Q_2(A_2: Tuple, S_2: Tuple) -> float:
    '''
    # TODO: Document
    '''
    
    # S_2 is the state, which fully determines the prediction function, so just pass the prediction function
    # This cuts down on refitting time.
    initial_condition, dynamics = S_2
    proposed_dose = A_2
    
    # Negative because we are going to minimize
    # This should compute E(Y|A_2, S_2)
    E_Y =  -1*Y(initial_condition + proposed_dose*dynamics).mean()

    return E_Y


def stage_2_optimization(S_2):
    
    # Ok, here is where we get V_2 and PI_2
    # PI is the policy; the dose; argmax_a Q_2(S_2, a)
    # V2 is the value; how long we spend in range, max_a Q2(S_2, a)
    
    tobs, yobs, theta, dose_times, dose_size = S_2


    
    predict = fit(t=tobs, y=yobs, theta=theta, dose_times=dose_times, dose_size=dose_size)
    


    # We will be splitting the time into two stages.  E.g 4 days on inital dose, 4 days after, etc.  Same number of days in two stages.
    # Likely be sampling near the end of the last day in stage 2.  This means I would need to predict over the same length of time.
    # We observe at t=tobs.  When is the next time a dose is taken?  It would be the first positive element of dose_times-tobs.
    # Negative values in the past, positive values in the future.
    next_dose_time_ix = np.argwhere((dose_times - tobs)>0).min()
    next_dose_time = dose_times[next_dose_time_ix]

    decision_point = int(len(dose_times)/2)
    tpred = np.arange(0.5, dose_times[decision_point]+0.5, 0.5)
    initial_condition, dynamics = predict(tpred, dose_times, np.ones_like(dose_times), c0_time=next_dose_time)


    # Can't give someone negative mg.  Bound the dose.
    dose_bnds = [(0, None)]
    D_old = np.unique(dose_size)
    optim = minimize(Q_2, x0=5.0, args=([D_old*initial_condition, dynamics]), bounds=dose_bnds, method = 'L-BFGS-B')
    
    π_2 = optim.x[0]
    
    # Undo the negative we did in the objective.
    expected_V_2 = -1*optim.fun 
    
    return π_2, expected_V_2
    

def Q_1(A_1, S_1):
    
    tobs, theta, dose_times = S_1
    proposed_dose = A_1
    dose_size = np.tile(proposed_dose, len(dose_times))
    
    possible_futures = prior_predict(tobs, theta, dose_times, dose_size, with_noise = True)
    
    counts, edges = np.histogram(possible_futures)
    centers = 0.5*(edges[1:] + edges[:-1])
    probabilities = (counts+1)/sum(counts+1)
    
    expected_v_2 = 0
    for yobs, p in zip(centers, probabilities):
        
        S_2 = tobs, yobs, theta, dose_times, dose_size
        π_2, V_2 = stage_2_optimization(S_2)
        expected_v_2+= V_2 * p
        
    
    # Now reward under this dose for stage 1
    decision_point = int(len(dose_times)/2)
    tpred = np.arange(0.5, dose_times[decision_point]+0.5, 0.5)
    prior_predictions = prior_predict(tpred, theta, dose_times, dose_size)
      
    return Y(prior_predictions).mean() + expected_v_2
        
        