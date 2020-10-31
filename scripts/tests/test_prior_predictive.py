import cmdstanpy
import numpy as np 
import pandas as pd
import pytest
import pickle
from ..tools.fit_original_model import fit_original_model


def test_prior_predictive():

    fit, model_data = fit_original_model()
    t_unique = np.unique(model_data['time'])

    c_check = fit.stan_variable('c_check')
    posterior_from_original = np.quantile(c_check, [0.025,  0.5, 0.975], axis = 0)

    # Fit the prior predictive
    with open('data/param_summary.pkl','rb') as file:
        params = pickle.load(file)

    #The params should be 1 on the z-standardized scale.
    params['sex'] = 1
    params['weight'] = params['weight_mean'] + params['weight_sd']
    params['creatinine'] = params['creatinine_mean'] + params['weight_sd']
    params['age'] = params['age_mean'] + params['age_sd']

    params['nt'] = 8
    params['prediction_times'] = t_unique.tolist()
    params['n_doses'] = 1
    params['dose_times'] = [0]
    params['doses'] = [1.0]

    prior_model = cmdstanpy.CmdStanModel(stan_file = 'experiment_models/prior_predictive.stan')
    prior_predictive = prior_model.sample(params, fixed_param=True, chains=1, iter_sampling = 2000, seed = 19920908) 
    c_check_2 = prior_predictive.stan_variable('C')
    posterior_from_summary= np.quantile(c_check_2, [0.025, 0.5, 0.975], axis = 0)

    np.testing.assert_allclose(posterior_from_original, posterior_from_summary, atol = 0.01)