import cmdstanpy
import pickle

import numpy as np
import pandas as pd

from tools.summarize_posterior import fit_gamma, fit_norm
from tools.fit_original_model import fit_original_model

fit, model_data = fit_original_model()

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