import numpy as np
import pandas as pd
import pickle
from tools.scoring_and_opt_tools import *
from tools.simulation_tools import *
from tools.q_learning import Y
from tqdm import tqdm
import gc

df = pd.read_csv('data/generated_data/sampled_covars_and_pk.csv')
loss_func = Y

def pop_best_dose_score(theta):

    #Everyone gets the same dose
    pop_best_dose = 8.566196538079296
    return score_dose(theta, Y, pop_best_dose)

def covar_only_dose_score(theta):

    covar_only_dose, value = covariate_dose_selection(theta, loss_func)
    return score_dose(theta=theta, loss_func=Y, dose_1=covar_only_dose)

def tdm_only_score(theta):

    tdm_only_dose, value = tdm_prior_dose_selection(theta, loss_func)
    return score_dose(theta, Y, dose_1 = tdm_only_dose)

def opt_time_score(theta):
    covar_only_dose, value = covariate_dose_selection(theta, loss_func)
    opt_time = optimize_over_time(theta, covar_only_dose)
    optimal_dose, value = myopic_dose_selection(theta, yobs=None, loss_func=loss_func, tobs = [opt_time], start_dose = covar_only_dose)
    return score_dose(theta, Y, dose_1 = covar_only_dose, dose_2 = optimal_dose)

def q_learning_score(theta):
    opt_start_dose, opt_time = perform_q_learning(theta)
    optimal_dose, value = myopic_dose_selection(theta, yobs=None, loss_func=loss_func, tobs = [opt_time], start_dose = opt_start_dose)
    return score_dose(theta, Y, dose_1 = opt_start_dose, dose_2 = optimal_dose)

def theoretically_best_score(theta):
    tpred, dose_times, dose_size, decision_point = setup_experiment(1)
    predict = lambda D, theta: -1*Y(observe(tpred, theta, dose_times, D*dose_size, return_truth = True)[1]).mean()
    dose_bnds = [(1, None)]
    optim = minimize(predict, x0=2.0, args=(theta), bounds=dose_bnds, method="powell", tol = 1e-8, options = dict(maxiter=10000)  )
    return score_dose(theta, Y, optim.x)


def one_sample_score(theta, tobs=None, yobs=None):
    covar_only_dose, value = covariate_dose_selection(theta, loss_func)
    myopic_dose, value = myopic_dose_selection(theta=theta, yobs = yobs, tobs = tobs, loss_func=loss_func, start_dose=covar_only_dose)
    return score_dose(theta, Y, dose_1 = covar_only_dose, dose_2 = myopic_dose)

def one_sample_tdm_score(theta, tobs=None, yobs=None):
    tdm_only_dose, value = tdm_prior_dose_selection(theta, loss_func)
    tdm_dose, value = tdm_dose_selection(theta=theta, loss_func=loss_func, start_dose=tdm_only_dose, yobs = yobs, tobs = tobs)
    return score_dose(theta, loss_func, dose_1 = tdm_only_dose, dose_2 = tdm_dose)



pop_best_scores = []
covar_only_scores = []
tdm_only_scores = []
theoretically_best_scores = []
one_sample_scores = []
one_sample_tdm_scores = []
q_learning_scores = []
opt_time_scores = []

for i,theta in tqdm(enumerate(df.to_dict(orient='records'))):
    # d = covariate_dose_selection(theta, Y)
    # tpred, dose_times, dose_size, decision_point = setup_experiment(d)
    # tobs = [np.random.uniform(low=108, high = 120)]
    # yobs = observe(tobs, theta, dose_times, dose_size,return_truth=False, random_state=i)


    # pop_best_scores.append(pop_best_dose_score(theta))
    # covar_only_scores.append(covar_only_dose_score(theta))
    # tdm_only_scores.append(tdm_only_score(theta))
    # opt_time_scores.append(opt_time_score(theta))
    # q_learning_scores.append(q_learning_score(theta))
    # theoretically_best_scores.append(theoretically_best_score(theta))
    one_sample_scores.append(one_sample_score(theta))
    one_sample_tdm_scores.append(one_sample_tdm_score(theta))


# with open('data/opts/pop_best_scores.txt', 'wb') as file:
#     pickle.dump(pop_best_scores, file)

# with open('data/opts/covar_only_scores.txt', 'wb') as file:
#     pickle.dump(covar_only_scores, file)

# with open('data/opts/tdm_only_scores.txt', 'wb') as file:
#     pickle.dump(tdm_only_scores, file)

# with open('data/opts/theoretically_best_scores.txt', 'wb') as file:
    # pickle.dump(theoretically_best_scores, file)

# with open('data/opts/q_learning_scores.txt', 'wb') as file:
#     pickle.dump(q_learning_scores, file)

# with open('data/opts/opt_time_scores.txt', 'wb') as file:
#     pickle.dump(opt_time_scores, file)

with open('data/opts/one_sample_scores.txt', 'wb') as file:
    pickle.dump(one_sample_scores, file)

with open('data/opts/one_sample_tdm_scores.txt', 'wb') as file:
    pickle.dump(one_sample_tdm_scores, file)

