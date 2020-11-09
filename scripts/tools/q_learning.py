import numpy as np
from scipy.optimize import minimize
from .simulation_tools import *

def Y(predictions, ll=0.10, ul=.30):
    '''
    Function to measure how long subject stays in range.
    Returns booleans if observation is in range.
    '''
    predictions_below_upper = predictions<ul
    predictions_above_lower = ll<predictions
    reward = (predictions_below_upper) & (predictions_above_lower)
    
    return reward

)

def Q_2(A_2, S_2):
    '''
    Q function for second stage.  A_2 first because this is what we optimize over.
    '''
    
    # S_2 is the state, which fully determines the prediction function, so just pass the prediction function
    # This cuts down on refitting time.
    tpred, pred_func, new_dose_times, new_dose_size = S_2
    proposed_dose_mg = A_2
    
    
    proposed_dose_times = new_dose_times.copy()
    proposed_dose_size = new_dose_size.copy()
    proposed_dose_size[np.argwhere(proposed_dose_times>tpred[0])] = proposed_dose_mg
    
    # This is a slow down.  I think I can optimize this by leveraging the fact that the
    # Concentration function is sums of concentration functions and that each time point has some factor of D[i] in it
    # Will optimize later.
    predictions = pred_func(tpred, proposed_dose_times, proposed_dose_size)
    
    
    # Negative because we are going to minimize
    # This should compute E(Y|A_2, S_2)
    return -1*Y(predictions).mean()


def stage_2_optimization(S_2):
    
    # Ok, here is where we get V_2 and PI_2
    # PI is the policy; the dose; argmax_a Q_2(S_2, a)
    # V2 is the value; how long we spend in range, max_a Q2(S_2, a)
    
    tobs, yobs, theta, dose_times, dose_size = S_2
    pred_func = fit(t=tobs, y=yobs, theta=theta, dose_times=dose_times, dose_size=dose_size)
    
    
    #Hard code for now
    # TODO: make specifying dose schedule more flexible
    new_dose_times = dose_times.copy()
    new_dose_size = dose_size.copy()
    tpred = np.arange(tobs[-1], 49)

    
    dose_bnds = [(0, None)]
    # Use powell method since the objective function is not differentiable
    optim = minimize(Q_2, x0=5.0, args=([tpred, pred_func, new_dose_times, new_dose_size]), bounds=dose_bnds, method='powell')
    
    π_2 = optim.x[0]
    
    # Undo the negative we did in the objective.
    expected_V_2 = -1*optim.fun 
    
    return π_2, expected_V_2
    

def Q_1(A_1, S_1):
    
    tobs, theta, dose_times = S_1
    proposed_dose = A_1
    dose_size = np.tile(proposed_dose, len(dose_times))
    
    possible_futures = prior_predict(tobs, theta, dose_times, dose_size, with_noise = True)
    
    counts, edges = np.histogram(possible_futures)
    centers = 0.5*(edges[1:] + edges[:-1])
    probabilities = (counts+1)/sum(counts+1)
    
    expected_v_2 = 0
    for yobs, p in zip(centers, probabilities):
        
        S_2 = tobs, yobs, theta, dose_times, dose_size
        π_2, V_2 = stage_2_optimization(S_2)
        expected_v_2+= V_2 * p
        
    
    #Now reward under this dose
    tpred = np.arange(0.5, tobs[-1])
    prior_predictions = prior_predict(tpred, theta, dose_times, dose_size)
      
    return Y(prior_predictions).mean() + expected_v_2
        
        