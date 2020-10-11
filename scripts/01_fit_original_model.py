import cmdstanpy
import pickle

import numpy as np
import pandas as pd

from tools.summarize_posterior import *

concentration_data = pd.read_csv("data/experiment.csv")

subject_data = concentration_data.drop_duplicates(['subjectids'])

model_data = dict(
    sex = subject_data.sex.tolist(),
    weight = subject_data.weight.tolist(),
    age = subject_data.age.tolist(),
    creatinine = subject_data.creatinine.tolist(),
    n_subjectids = subject_data.shape[0],
    D = subject_data.D.tolist(),

    subjectids = concentration_data.subjectids.tolist(),
    time = concentration_data.time.tolist(),
    yobs = concentration_data.yobs.tolist(),
    n = concentration_data.shape[0]
)

model = cmdstanpy.CmdStanModel(stan_file = 'experiment_models/original_model.stan')

fit = model.sample(model_data, chains = 12, parallel_chains = 4, seed = 19920908, show_progress=True)

df = fit.draws_as_dataframe()

means = ['mu_cl',
         'mu_tmax',
         'mu_alpha']

coefs = ['beta_cl.1', 
         'beta_cl.2', 
         'beta_cl.3', 
         'beta_cl.4',
         'beta_t.1',
         'beta_t.2',
         'beta_t.3', 
         'beta_t.4',
         'beta_a.1',
         'beta_a.2',
         'beta_a.3',
         'beta_a.4']

sds = ['s_cl','s_t','s_alpha','sigma']

param_dict_normals = fit_norm(df, means + coefs)
param_dict_gammas = fit_gamma(df, sds)
param_dict = {**param_dict_normals, **param_dict_gammas}


# Need to standrdize in the model
weight_mean, weight_sd = np.mean(model_data['weight']), np.std(model_data['weight'], ddof = 1)
age_mean, age_sd = np.mean(model_data['age']), np.std(model_data['age'], ddof = 1)
creatinine_mean, creatinine_sd = np.mean(model_data['creatinine']), np.std(model_data['creatinine'], ddof = 1)

param_dict['weight_mean'] = weight_mean
param_dict['age_mean'] = age_mean
param_dict['creatinine_mean'] = creatinine_mean

param_dict['weight_sd'] = weight_sd
param_dict['age_sd'] = age_sd
param_dict['creatinine_sd'] = creatinine_sd


with open('data/param_summary.pkl', 'wb') as file:
    pickle.dump(param_dict, file)