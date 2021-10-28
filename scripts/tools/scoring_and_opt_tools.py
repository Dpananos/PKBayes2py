import numpy as np
import pandas as pd

from scipy.optimize import minimize

from .simulation_tools import *
from .q_learning import *
from ax.service.managed_loop import optimize


def score_dose(theta, loss_func, dose_1, dose_2=None):
    
    tpred, dose_times, dose_sizes, decision_point = setup_experiment(dose_1)
    
    if dose_2 is not None:
        dose_sizes[decision_point:] = dose_2
    
    y_obs, y_true = observe(tpred, theta, dose_times, dose_sizes)
    
    return loss_func(y_true).mean()
    


def tdm_prior_dose_selection(theta, loss_func):
    
    # Do dose=1.0 since we are going to optimize over dose and the concentration function
    # is proportional to dose
    tpred, dose_times, dose_sizes, decision_point = setup_experiment(dose=1.0)
    prior_predictions = prior_predict_tdm(tpred, theta, dose_times, dose_sizes)
    
    
    reward_func = lambda dose: -loss_func(dose*prior_predictions).mean()
    dose_bnds = [(0, None)]
    res = minimize( reward_func, x0 = 1.0, bounds = dose_bnds, method = 'powell')
    
    optimal_dose = res.x[0]
    value = -res.fun
    
    return optimal_dose, value


def covariate_dose_selection(theta, loss_func):
    
    # Do dose=1.0 since we are going to optimize over dose and the concentration function
    # is proportional to dose
    tpred, dose_times, dose_sizes, decision_point = setup_experiment(dose=1.0)
    prior_predictions = prior_predict(tpred, theta, dose_times, dose_sizes)
    
    
    reward_func = lambda dose: -loss_func(dose*prior_predictions).mean()
    dose_bnds = [(0, None)]
    res = minimize( reward_func, x0 = 1.0, bounds = dose_bnds, method = 'powell')
    
    optimal_dose = res.x[0]
    value = -res.fun
    
    return optimal_dose, value


def myopic_dose_selection(theta, yobs, loss_func, tobs = None, start_dose = None):
    

    if start_dose is not None:
        optimal_dose = start_dose
    else:
        optimal_dose, value = covariate_dose_selection(theta, loss_func)

    tpred, dose_times, dose_sizes, decision_point = setup_experiment(optimal_dose)

    if not tobs:
        # These are the times the doses are taken right before the halfway point
        ts, tf = dose_times[(decision_point-1):(decision_point+1)]
        tobs = np.random.uniform(ts, tf, size = 1)
        
    if yobs is None:
        yobs = observe(tobs, theta, dose_times, dose_sizes, return_truth = False)
    
    predict = fit(tobs, yobs, theta, dose_times, dose_sizes)

    # Make times for future prediction.  Keep in mind we only need half the time
    # since t=0 corresponds to half way point
    future_tpred = np.arange(0.5, dose_times[decision_point], 0.5)
    ones_dose_size = np.ones_like(dose_times)
    
    ic, dyn = predict(future_tpred, dose_times, ones_dose_size, c0_time = tobs[0] )
    
    reward_func = lambda dose: -loss_func(dose*dyn + ic).mean()
    dose_bnds = [(0, None)]
    res = minimize(reward_func, x0 = 1.0, bounds = dose_bnds, method = 'powell')
    
    optimal_dose = res.x[0]
    value = -res.fun
    
    return optimal_dose, value


def tdm_dose_selection(theta, yobs, loss_func, tobs = None, start_dose = None):
    
    if start_dose is not None:
        optimal_dose = start_dose
    else:
        optimal_dose, value = covariate_dose_selection(theta, loss_func)

    
    tpred, dose_times, dose_sizes, decision_point = setup_experiment(dose=optimal_dose)

    if not tobs:
        # These are the times the doses are taken right before the halfway point
        ts, tf = dose_times[(decision_point-1):(decision_point+1)]
        tobs = np.random.uniform(ts, tf, size = 1)

    if yobs is None: 
        yobs = observe(tobs, theta, dose_times, dose_sizes, return_truth = False)
    
    predict = fit_tdm(tobs, yobs, theta, dose_times, dose_sizes)

    # Make times for future prediction.  Keep in mind we only need half the time
    # since t=0 corresponds to half way point
    future_tpred = np.arange(0.5, dose_times[decision_point], 0.5)
    ones_dose_size = np.ones_like(dose_times)
    
    ic, dyn = predict(future_tpred, dose_times, ones_dose_size, c0_time = tobs[0] )
    
    reward_func = lambda dose: -loss_func(dose*dyn + ic).mean()
    dose_bnds = [(0, None)]
    res = minimize(reward_func, x0 = 1.0, bounds = dose_bnds, method = 'powell')
    
    optimal_dose = res.x[0]
    value = -res.fun
    
    return optimal_dose, value

def optimize_over_time(theta, dose):

    tpred, dose_times, dose_sizes, decision_point = setup_experiment(dose)

    # Set up times we would observe subejcts
    tobs_min = dose_times[decision_point - 1]
    tobs_max = dose_times[decision_point]


    tobs_parameter = {
                "name":'tobs',
                'type':'range',
                'bounds':[108.0, 120.0], #Hard code because ax is picky about this
                "value_type": "float"}



    best_parameters, values, experiment, model = optimize(
        parameters=[tobs_parameter],
        experiment_name = 'test',
        objective_name = 'q1',
        minimize = False,
        evaluation_function = lambda x: Q_1(dose,([x['tobs']], theta, dose_times)))

    tobs_subject = best_parameters['tobs']

    return tobs_subject

def perform_q_learning(theta):

    tpred, dose_times, dose_sizes, decision_point = setup_experiment(1.0)


    dose_parameter = {
            "name":'D',
            'type':'range',
            'bounds':[1.0, 20.0],
             "value_type": "float"}

    tobs_parameter = {
                "name":'tobs',
                'type':'range',
                'bounds':[108.0, 120.0], #Hard code because ax is picky about this
                "value_type": "float"}


    best_parameters, values, experiment, model = optimize(
        parameters=[dose_parameter, tobs_parameter],
        experiment_name = 'test',
        objective_name = 'q1',
        minimize = False,
        evaluation_function = lambda x: Q_1(x['D'],([x['tobs']], theta, dose_times)))

    opt_dose = best_parameters['D']
    opt_time = best_parameters['tobs']

    return opt_dose, opt_time
