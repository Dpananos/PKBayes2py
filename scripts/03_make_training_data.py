import cmdstanpy
import pickle
import pandas as pd
import numpy as np
from scripts.tools.simulation_tools import *
from scripts.tools.scoring_and_opt_tools import *
from tqdm.notebook import tqdm
from smt.sampling_methods import LHS
from itertools import product



N_samples = 1000

# This model is capable of generating PK parameters for subjects given the psoterior from the 
# original model in step 01.  
generative_model = cmdstanpy.CmdStanModel(exe_file='experiment_models/draw_pk_parameters')

# Load the parameters I computed from step 01.
with open('data/param_summary.pkl', 'rb') as file:
    params = pickle.load(file)

# Set up some limits based on our data.
# Use Latin Hypercube Sampling to sample the domain to build a model
# That replaces the optimization of Q_2
tpred, dose_times, dose_sizes, decision_point = setup_experiment(1,num_days=10, doses_per_day=2, hours_per_dose=12)

age_lims = [26.0, 70.0]
weight_lims = [54.7, 136.6]
creatinine_lims = [50, 95]
sex_lims = [0, 1]
tpred_lims = [dose_times[decision_point-1], dose_times[decision_point]]
dose_lims = [1, 20.0]
lims = np.array([age_lims, weight_lims, creatinine_lims, sex_lims, tpred_lims, dose_lims])

sampling = LHS(xlimits = lims)
domain = sampling(1000)
colnames = ['age','weight','creatinine','sex', 'tpred', 'D']
domain_df = pd.DataFrame(domain, columns=colnames).assign(sex = lambda x: x.sex.round())


# Convert resampled covars to dict
sampled_covars_dict = domain_df.to_dict(orient = 'list')

params = {**domain_df.to_dict(orient = 'list'), **params}
params['n_subjects'] = domain_df.shape[0]

# Now sample
fit = generative_model.sample(params, fixed_param=True, iter_sampling=1, seed = 19920908)

# Append the pk params.  These are all I need to generate observations and pk curves
domain_df['cl'] = fit.stan_variable('cl').squeeze()
domain_df['ke'] = fit.stan_variable('ke').squeeze()
domain_df['ka'] = fit.stan_variable('ka').squeeze()
domain_df['alpha'] = fit.stan_variable('alpha').squeeze()


possible_outcomes = []

for theta in domain_df.to_dict(orient = 'records'):
    
    tobs = [theta['tpred']]
    D = theta['D']
    
    yobs = prior_predict(tobs, theta, dose_times, np.ones_like(dose_times)*D)
    
    counts, bin_edges = np.histogram(yobs)
    
    centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
    
    possible_outcomes.append(centers)
    
    
possible_outcomes_df = pd.DataFrame(np.array(possible_outcomes), columns = [f'y_{j}' for j in range(10)])

training_df = pd.concat((domain_df, possible_outcomes_df), axis = 1).assign(ID = np.arange(Nsamples))

final_training = pd.melt(training_df, id_vars = [j for j in training_df.columns if 'y_' not in j ], value_name = 'yobs').drop('variable', axis = 1)

outcomes = []
for theta in tqdm(final_training.to_dict(orient = 'records')):
    
    tpred, dose_times, dose_size, decision_point = setup_experiment(1,num_days=10, doses_per_day=2, hours_per_dose=12)
    
    dose_size = np.ones_like(dose_times)*theta['D']
    tobs = [theta['tpred']]
    yobs = [theta['yobs']]
    
    S_2 = tobs, yobs, theta, dose_times, dose_size
    
    π_2, V_2 = stage_2_optimization(S_2)
    
    outcomes.append(V_2)

final_training['outcomes'] = outcomes
final_training.to_csv('data/stage_2_optimization_training.csv', index = False)