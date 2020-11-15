import pandas as pd
import numpy as np
import sys
from tools.q_learning import *


#cmdstan shhhhh
import logging
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

df = pd.read_csv("data/sampled_covars_and_pk.csv")
pk_params = df.to_dict(orient='records')

tobs_subjects, best_starting_doses = perform_q_learning(pk_params, num_days = 10)

df['tobs'] = tobs_subjects
df['q_learn_best_starting_dose'] = best_starting_doses
df.to_csv('data/q_learn_results.csv', index=False)