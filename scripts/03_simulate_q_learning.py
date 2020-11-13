import pandas as pd
import numpy as np
from tools.simulation_tools import *
from tools.q_learning import *
from tqdm import tqdm

#cmdstan shhhhh
import logging
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

np.random.seed(19920908)

df = pd.read_csv("data/sampled_covars_and_pk.csv")
pk_params = df.to_dict(orient='records')

def perform_q_learning(pk_params, num_days=10, doses_per_day=2, hours_per_dose = 12):

    # The last time the subejct could take a dose.  End simulation just before this time.
    tmax = hours_per_dose * doses_per_day * num_days
    # Spread doses out over time
    dose_times = np.arange(0, tmax, hours_per_dose)
    # We get to make a decision about the size of the dose at the half way mark
    decision_point = int(len(dose_times)/2)
    
    # Set up times we would observe subejcts
    tobs_min = dose_times[decision_point-1]
    tobs_max = dose_times[decision_point]

    
    best_starting_doses = []
    tobs_subjects =[]
    
    for theta in tqdm(pk_params, desc = 'Looping Over Subjects'):

        tobs = np.random.uniform(tobs_min, tobs_max, size = (1,))

        S_1 = (tobs, theta, dose_times)
        best_dose = stage_1_optimization(S_1, step = 1.5)
        
        best_starting_doses.append(best_dose)
        tobs_subjects.append(tobs)
    
    
    
    return (tobs_subject, best_starting_doses)


tobs_subject, best_starting_doses = perform_q_learning(pk_params, num_days = 10)
df['tobs'] = tobs_subject
df['best_starting_dose'] = best_starting_doses
df.to_csv('data/q_learn_results.csv', index=False)