import cmdstanpy
import pickle

import pandas as pd
import numpy as np

# This model is capable of generating PK parameters for subjects given the psoterior from the 
# original model in step 01.  
generative_model = cmdstanpy.CmdStanModel(stan_file='experiment_models/draw_pk_parameters.stan')

# Load the parameters I computed from step 01.
with open('data/param_summary.pkl', 'rb') as file:
    params = pickle.load(file)

# First, resample the covars from the data in step 01.
# Set random state for reproducibility.
# Can convert into a dictionary and merge with the model parameters
sampled_covars = (
    pd.read_csv('data/experiment.csv').
    drop_duplicates(['subjectids']).
    loc[:,['age','sex','weight','creatinine']].
    iloc[[0],].
    sample(100, replace = True, random_state = 19920908)
)

# Convert resampled covars to dict
sampled_covars_dict = sampled_covars.to_dict(orient = 'list')

params = {**sampled_covars.to_dict(orient = 'list'), **params}
params['n_subjects'] = sampled_covars.shape[0]

# Now sample
fit = generative_model.sample(params, fixed_param=True, iter_sampling=1, seed = 19920908)

# Append the pk params.  These are all I need to generate observations and pk curves
sampled_covars['cl'] = fit.stan_variable('cl').squeeze()
sampled_covars['ke'] = fit.stan_variable('ke').squeeze()
sampled_covars['ka'] = fit.stan_variable('ka').squeeze()
sampled_covars['alpha'] = fit.stan_variable('alpha').squeeze()

# Save for later
sampled_covars.to_csv('data/sampled_covars_and_pk.csv', index = False)