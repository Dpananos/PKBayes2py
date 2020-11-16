from tools.simulation_tools import *
from tools.q_learning import *
from tqdm import tqdm

import numpy as np
import pandas as pd

import logging
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)



df = pd.read_csv('data/covar_only_results.csv')
pk_params =  df.to_dict(orient='records')

df['q_learning_obsereved_reward'] = [score(theta) for theta in tqdm(pk_params, desc = 'Looping over subjects')]

df.to_csv('data/final_results.csv', index = False)