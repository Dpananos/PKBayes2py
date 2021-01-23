import cmdstanpy
import pickle
import pandas as pd
import numpy as np
from scripts.tools.simulation_tools import *
from scripts.tools.scoring_and_opt_tools import *
from tqdm.notebook import tqdm

from itertools import product

'''
Eventually, I will be doing Q learning.  The Q_1 function (the function which returns
the value of the policty for the first stage decision) calls an optimization prodecure
10 times.  Each run of the optimization prodecure takes 2 seconds, so a single call of Q_1
can take 30 seconds.  Not good, especally since i need to call Q_1 many times for 100 subjects.

We can speed it up my learning  model from the results of the optimization.  I need data
to interpolate, so this script makes that data.
'''

# This model is capable of generating PK parameters for subjects given the psoterior from the 
# original model in step 01.  
generative_model = cmdstanpy.CmdStanModel(exe_file='experiment_models/draw_pk_parameters')

# Load the parameters I computed from step 01.
with open('data/param_summary.pkl', 'rb') as file:
    params = pickle.load(file)

# First, resample the covars from the data in step 01.
# Set random state for reproducibility.
# Can convert into a dictionary and merge with the model parameters
# Assign a column called "key " so I can do an outer join later.
covars = (
    pd.read_csv('data/experiment.csv').
    loc[:,['age','weight','creatinine','sex']].
    drop_duplicates().
    assign(key=1)
)
 

# Now, I need to generate a grid of observation times and possible doses.
possible_observation_times = np.arange(dose_times[decision_point-1], dose_times[decision_point])
possible_doses = np.arange(1, 20)

dose_obs_time_combo = pd.DataFrame([combo for combo in product(possible_doses, possible_observation_times)], columns=['dose','obs_time'])
dose_obs_time_combo['key'] = 1

covar_time_dose = pd.merge(covars, dose_obs_time_combo, how = 'outer', on = 'key')

dist_of_possible_obs = []

for theta in tqdm(covar_time_dose.to_dict(orient = 'records')):
    
    tpred, dose_times, dose_size, decision_point = setup_experiment(theta['dose'],num_days=10, doses_per_day=2, hours_per_dose=12)
    possible_futures = prior_predict([theta['obs_time']], theta, dose_times, dose_size, with_noise=True)
    
    counts, edges = np.histogram(possible_futures)
    centers = 0.5 * (edges[1:] + edges[:-1])
    # Laplace smoothing, just in case.  This should be unneccesary however.
    probabilities = (counts + 1) / sum(counts + 1)

    dist_of_possible_obs.append(centers)
    