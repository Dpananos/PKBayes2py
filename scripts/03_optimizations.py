import numpy as np
import pandas as pd
from tools.scoring_and_opt_tools import *
from tools.q_learning import Y
from tqdm import tqdm
import gc

df = pd.read_csv('data/sampled_covars_and_pk.csv')
loss_func = Y

for i,theta in tqdm(enumerate(df.to_dict(orient='records'))):
    results = {}
    # # Select inital dose using covar
    # init_dose, value = covariate_dose_selection(theta, loss_func)
    # results['covar_only_dose'] = init_dose
 

    # # Now do myopic dose selection
    # opt_dose, value = myopic_dose_selection(theta, loss_func, tobs = None)
    # results['myopic_dose'] = opt_dose


    # # Now do time opt dose
    # This processes starts with covar dose.
    # opt_time = optimize_over_time(theta, init_dose)
    # results['opt_time'] = opt_time

    #Now do q learning
    opt_dose, opt_time = perform_q_learning(theta)
    results['q_learn_opt_time'] = opt_time
    results['q_learn_opt_dose'] = opt_dose


    gc.collect()

    pd.DataFrame(results, index = [i]).to_csv(f'data/opts/subject_q_lrn_{i}.csv')

