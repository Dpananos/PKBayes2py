import pandas as pd
import numpy as np
from scipy.optimize import minimize
from tools.q_learning import *
from tools.simulation_tools import *
from tqdm import tqdm

import logging
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


df = pd.read_csv('data/q_learn_results.csv')
pk_params = df.to_dict(orient='records')


num_days=10
doses_per_day=2
hours_per_dose=12
tmax = num_days * doses_per_day * hours_per_dose
step_size = 0.5

dose_times = np.arange(0, tmax, hours_per_dose)
t_pred = np.arange(0.5, tmax + step_size , step_size)

best_dose = []
expected_reward = []
observed_reward = []

reward_func = lambda dose, predictions: -1*Y(dose*predictions).mean()

for theta in tqdm(pk_params, desc = 'Looping over subjects'):
    prior_predictions = prior_predict(t_pred, theta, dose_times, np.ones_like(dose_times))

    dose_bnds = [(0, None)]
    res = minimize(lambda x: reward_func(x, prior_predictions), x0 = 5, method = 'powell', bounds = dose_bnds)
    
    optimal_dose = res.x[0]
    value = -1*res.fun

    best_dose.append(optimal_dose)
    expected_reward.append(value)

    # Now observe under this dose
    yobs, ytrue = observe(t_pred, theta, dose_times, np.tile(optimal_dose, dose_times.size), return_truth = True)

    actual_reward = Y(ytrue).mean()
    observed_reward.append(actual_reward)



df['covar_only_best_dose'] = best_dose
df['covar_only_expected_reward'] = expected_reward
df['covar_only_observed_reward'] = observed_reward


df.to_csv('data/covar_only_results.csv', index = False)