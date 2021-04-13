import cmdstanpy
import pickle
import pandas as pd
import numpy as np
from tools.simulation_tools import *
from tools.scoring_and_opt_tools import *
from tools.q_learning import *
from tqdm import tqdm
from itertools import product

tpred, dose_times, dose_sizes, decision_point = setup_experiment(1, num_days=10, doses_per_day=2, hours_per_dose=12)

# Load the parameters I computed from step 01.  These are going to be used in the prior predictive model and the
# model which gives me PK parameters.
with open("data/generated_data/param_summary.pkl", "rb") as file:
    params = pickle.load(file)

# Now that we have drawn covariates via LHS, we can simulate some outcomes from subjects with those covariates.
# I will generate some more PK params, simulate possible outcomes, and compute rewards under the sampled doses.
domain_df = pd.read_csv('data/generated_data/hypercube_sampled_covariates.csv')
sampled_covars_dict = domain_df.to_dict(orient="list")

# Combine the LH sampled covars and the model parameters into a single dictionary.
# Set n so Stan can make all the covars in one go.
params = {**sampled_covars_dict, **params}
params["n_subjects"] = domain_df.shape[0]


# This model is capable of generating PK parameters for subjects given the posterior from the
# original model in step 01.
generative_model = cmdstanpy.CmdStanModel(exe_file="experiment_models/draw_pk_parameters")

# Now sample to get the PK params
fit = generative_model.sample(params, fixed_param=True, iter_sampling=1, seed=19920908)

# Append the pk params.  These are all I need to generate observations and pk curves
domain_df["cl"] = fit.stan_variable("cl").squeeze()
domain_df["ke"] = fit.stan_variable("ke").squeeze()
domain_df["ka"] = fit.stan_variable("ka").squeeze()
domain_df["alpha"] = fit.stan_variable("alpha").squeeze()


# Here is where we sample possible outcomes for each simulated subject.
# Because I know their PK parameters, I can ask myself "what could I resonably expect to see under my model at t=tobs"
# knowing only their covariates?
# Bin the observations because else it would be far too intensive.
possible_outcomes = []
for theta in tqdm(domain_df.to_dict(orient="records")):

    tobs = [theta["tpred"]]
    D = theta["D"]
    yobs = prior_predict(tobs, theta, dose_times, np.ones_like(dose_times) * D)
    counts, bin_edges = np.histogram(yobs)
    centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    possible_outcomes.append(centers)

possible_outcomes_df = pd.DataFrame(np.array(possible_outcomes), columns=[f"y_{j}" for j in range(10)])
training_df = pd.concat((domain_df, possible_outcomes_df), axis=1).assign(ID=np.arange(possible_outcomes_df.shape[0]))
final_training = pd.melt(training_df, id_vars=[j for j in training_df.columns if "y_" not in j], value_name="yobs").drop("variable", axis=1)

# Finally, compute the reward.  At this point, I know what I might expect to see (concentration wise) as well as covariates.
# I can then compute the reward from the stage_2_optimization and then learn a model from that data.
outcomes = []
for theta in tqdm(final_training.to_dict(orient="records")):

    tpred, dose_times, dose_size, decision_point = setup_experiment(1, num_days=10, doses_per_day=2, hours_per_dose=12)

    dose_size = np.ones_like(dose_times) * theta["D"]
    tobs = [theta["tpred"]]
    yobs = [theta["yobs"]]

    S_2 = tobs, yobs, theta, dose_times, dose_size

    π_2, V_2 = stage_2_optimization(S_2)

    outcomes.append(V_2)

final_training["outcomes"] = outcomes
final_training.to_csv("data/generated_data/stage_2_optimization_training.csv", index=False)


