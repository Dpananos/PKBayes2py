import cmdstanpy
import numpy as np
import pickle
import warnings

from scipy.stats import gamma, norm
from typing import List, Dict


# Dictionary of parameters to be used in models.  Prevents me from having to 
# Manually enter prior params. See step 01
with open('data/generated_data/param_summary.pkl','rb') as file:
    _params = pickle.load(file)

# # This model is the prior predictive check.  Give it covariates and it will give you predictions
_prior_model = cmdstanpy.CmdStanModel(exe_file = 'experiment_models/prior_predictive')
_prior_tdm_model = cmdstanpy.CmdStanModel(exe_file = 'experiment_models/tdm_prior_predictive')

# # This model fits to simulated data.  Give it covariates and observed data and it will give you predictions.
_conditioning_model = cmdstanpy.CmdStanModel(exe_file = 'experiment_models/condition_on_patients')
_tdm_model = cmdstanpy.CmdStanModel(exe_file = 'experiment_models/bayesian_tdm_single_subject')

def validate_input(x)->np.ndarray:
    '''
    Helper function to turn lists into arrays when I pass arguments to functions.
    Just cuts down on me writing np.array.
    '''
    xc = x.copy()
    if not isinstance(xc, np.ndarray):
            xc = np.array([xc]).ravel()
    return xc


def concentration_function(D: float , t: np.array, cl: float, ke: float, ka: float)->np.array:

    '''
    Returns solution to the differential equation
        y' = D * ke * ka * exp(-ka * t) / cl - ke*y(t)

    Inuts:
    D: float - Dose size in mg
    t: np.array - Times at which to evaluate the solution
    cl: float - Clearance rate in units L/hour
    ke: float - Elimination rate in units 1/hour
    ka: float - Absorption rate in units 1/hour
    '''

    y = (0.5*D * ke * ka)/(cl*(ke - ka)) * (np.exp(-ka*t) - np.exp(-ke*t))

    return np.heaviside(t, 0)*y


def repeated_dose_concentration(cl: float, ke: float, ka:float) -> callable:

    #TODO Comment this function and maybe write unit tests.

    def func(t: np.array, dose_times: np.array, doses: np.array) -> np.array:
        y = 0
        for d, tau in zip(doses, dose_times):
            y += concentration_function(d, t-tau, cl, ke, ka)

        return y
    return func

def prior_predict_tdm(t:List, theta: Dict, dose_times:List, dose_size:List, with_noise = False)->np.ndarray:

    '''
    Generate retrodictions given times, covariates, and a dosing schedule.
    '''
    
    # validate the inputs
    times = validate_input(t)
    dose_timings = validate_input(dose_times)
    doses = validate_input(dose_size)

    if np.any(times<=0):
        raise ValueError('All times for prior predictions must be greater than 0.  Check input t to veryify this requirement.')

    # Create state
    model_data = _params.copy()
    model_data['sex'] = theta['sex']
    model_data['age'] = theta['age']
    model_data['weight'] = theta['weight']
    model_data['creatinine'] = theta['creatinine']
    
    model_data['n_doses'] = doses.size
    model_data['dose_times'] = dose_timings.tolist()
    model_data['doses'] = doses.tolist()
    
    model_data['nt'] = times.size
    model_data['prediction_times'] = times.tolist()

    
    if with_noise:
        return _prior_tdm_model.sample(model_data, fixed_param = True, iter_sampling=2000, seed = 19920908).stan_variable("C_noise")
    else:
        return _prior_tdm_model.sample(model_data, fixed_param = True, iter_sampling=2000, seed = 19920908).stan_variable("C")

def prior_predict(t:List, theta: Dict, dose_times:List, dose_size:List, with_noise = False)->np.ndarray:

    '''
    Generate retrodictions given times, covariates, and a dosing schedule.
    '''
    
    # validate the inputs
    times = validate_input(t)
    dose_timings = validate_input(dose_times)
    doses = validate_input(dose_size)

    if np.any(times<=0):
        raise ValueError('All times for prior predictions must be greater than 0.  Check input t to veryify this requirement.')

    # Create state
    model_data = _params.copy()
    model_data['sex'] = theta['sex']
    model_data['age'] = theta['age']
    model_data['weight'] = theta['weight']
    model_data['creatinine'] = theta['creatinine']
    
    model_data['n_doses'] = doses.size
    model_data['dose_times'] = dose_timings.tolist()
    model_data['doses'] = doses.tolist()
    
    model_data['nt'] = times.size
    model_data['prediction_times'] = times.tolist()

    
    if with_noise:
        return _prior_model.sample(model_data, fixed_param = True, iter_sampling=2000, seed = 19920908).stan_variable("C_noise")
    else:
        return _prior_model.sample(model_data, fixed_param = True, iter_sampling=2000, seed = 19920908).stan_variable("C")
    
    
def observe(t: List, theta: Dict, dose_times:list , dose_size:list, return_truth = True, random_state=None)->tuple:

    '''
    Draw an observation from a simulated patient given a time, covariates, and a dosing schedule.
    '''
    
    times = validate_input(t)
    dose_timings = validate_input(dose_times)
    doses = validate_input(dose_times)

    # Instantiate the repeated dosing function for this patient.  The function depends on their pk params, which are given.
    # We know the pk params becuase we simulated them
    observation_func = repeated_dose_concentration(theta['cl'], theta['ke'], theta['ka'])
    
    # Use the obserbation function to draw true concentration values
    true_concentrations = observation_func(times, dose_times, dose_size)
    
    # Draw a sigma from the prior distribution (fixed with random state) and then corrupt the observation with lognormal noise.
    sigma = gamma(a=_params['shape_sigma'], scale=1.0/_params['rate_sigma']).rvs(1, random_state=0)

    observed_concentrations = true_concentrations*np.exp(norm(loc=0, scale=sigma).rvs(size = times.size, random_state = random_state))
    
    if return_truth:
        return observed_concentrations, true_concentrations
    else:
        return observed_concentrations
        
def fit(t: List, y: List, theta: Dict, dose_times: List, dose_size: List, return_model = False)->callable:
    
    times = validate_input(t)
    yobs = validate_input(y)
    dose_timings = validate_input(dose_times)
    doses = validate_input(dose_size)
    
    model_data = _params.copy()
    model_data['sex'] = theta['sex']
    model_data['age'] = theta['age']
    model_data['weight'] = theta['weight']
    model_data['creatinine'] = theta['creatinine']
    
    model_data['n_doses'] = doses.size
    model_data['dose_times'] = dose_timings.tolist()
    model_data['doses'] = doses.tolist()
    
    model_data['n'] = times.size
    model_data['observed_times'] = times.tolist()
    model_data['observed_concentrations'] = yobs.tolist()
    
    model_data['nt'] = times.size
    model_data['prediction_times'] = times.tolist()
    model_data['c0_time'] = [0]
    
    conditioned_model = _conditioning_model.sample(model_data, adapt_delta=0.99, seed = 19920908)
    
    def predict(tpred: List, new_dose_times: List, new_dose_size:List, c0_time: float = 0,  with_noise = False)->np.ndarray:
        
        time_pred = validate_input(tpred)
        new_doses = validate_input(new_dose_size)
        new_dose_timings = validate_input(new_dose_times)

        model_data['nt'] = time_pred.size
        model_data['prediction_times'] = time_pred.tolist()
        model_data['n_doses'] = new_doses.size
        model_data['dose_times'] = new_dose_timings.tolist()
        model_data['doses'] = new_doses.tolist()
        model_data['c0_time'] = [c0_time]
            

        gqs = _conditioning_model.generate_quantities(model_data, conditioned_model, seed = 19920908)
        ypred_col_ix = ['ypred' in j for j in gqs.column_names]
        initial_conc_col_ix = ['initial_concentration' in j for j in gqs.column_names]
        
        dynamics = gqs.generated_quantities[:, ypred_col_ix]
        initial_condition = gqs.generated_quantities[:, initial_conc_col_ix]

        # There is an enormous bug somewhere.  
        # On some occassions, initial_condition or dynamics will have Nans. 
        # I have no explanation for this at the moment.  It appears even when the model fits well 
        # and diagnostics show no pathological behaviour.
        # I suspect it might be a problem with Stan, but that is unlikely.
        # For now, I'm performing a work around by just removing rows from both dynamics and initial_condition
        # which have nans
        # if np.isnan(initial_condition).any() or np.isnan(dynamics).any():
            # warnings.warn('Caution, one of dynamics or initial_condition has missing values')

        rows_no_nan = (~np.isnan(initial_condition).any(axis=1)) & (~np.isnan(dynamics).any(axis=1))
        
        filtered_initial_condition = initial_condition[rows_no_nan]
        filtered_dynamics = dynamics[rows_no_nan]

        return (filtered_initial_condition, filtered_dynamics)

    if return_model:
        return predict, conditioned_model
    else:
        return predict

def fit_tdm(t: List, y: List, theta: Dict, dose_times: List, dose_size: List, return_model = False)->callable:
    
    times = validate_input(t)
    yobs = validate_input(y)
    dose_timings = validate_input(dose_times)
    doses = validate_input(dose_size)
    
    model_data = _params.copy()
    model_data['sex'] = theta['sex']
    model_data['age'] = theta['age']
    model_data['weight'] = theta['weight']
    model_data['creatinine'] = theta['creatinine']
    
    model_data['n_doses'] = doses.size
    model_data['dose_times'] = dose_timings.tolist()
    model_data['doses'] = doses.tolist()
    
    model_data['n'] = times.size
    model_data['observed_times'] = times.tolist()
    model_data['observed_concentrations'] = yobs.tolist()
    
    model_data['nt'] = times.size
    model_data['prediction_times'] = times.tolist()
    model_data['c0_time'] = [0]
    
    conditioned_model = _tdm_model.sample(model_data, adapt_delta=0.99, seed = 19920908)
    
    def predict_tdm(tpred: List, new_dose_times: List, new_dose_size:List, c0_time: float = 0,  with_noise = False)->np.ndarray:
        
        time_pred = validate_input(tpred)
        new_doses = validate_input(new_dose_size)
        new_dose_timings = validate_input(new_dose_times)

        model_data['nt'] = time_pred.size
        model_data['prediction_times'] = time_pred.tolist()
        model_data['n_doses'] = new_doses.size
        model_data['dose_times'] = new_dose_timings.tolist()
        model_data['doses'] = new_doses.tolist()
        model_data['c0_time'] = [c0_time]
            

        gqs = _tdm_model.generate_quantities(model_data, conditioned_model, seed = 19920908)
        ypred_col_ix = ['ypred' in j for j in gqs.column_names]
        initial_conc_col_ix = ['initial_concentration' in j for j in gqs.column_names]
        
        dynamics = gqs.generated_quantities[:, ypred_col_ix]
        initial_condition = gqs.generated_quantities[:, initial_conc_col_ix]

        # There is an enormous bug somewhere.  
        # On some occassions, initial_condition or dynamics will have Nans. 
        # I have no explanation for this at the moment.  It appears even when the model fits well 
        # and diagnostics show no pathological behaviour.
        # I suspect it might be a problem with Stan, but that is unlikely.
        # For now, I'm performing a work around by just removing rows from both dynamics and initial_condition
        # which have nans
        # if np.isnan(initial_condition).any() or np.isnan(dynamics).any():
            # warnings.warn('Caution, one of dynamics or initial_condition has missing values')

        rows_no_nan = (~np.isnan(initial_condition).any(axis=1)) & (~np.isnan(dynamics).any(axis=1))
        
        filtered_initial_condition = initial_condition[rows_no_nan]
        filtered_dynamics = dynamics[rows_no_nan]

        return (filtered_initial_condition, filtered_dynamics)

    if return_model:
        return predict_tdm, conditioned_model
    else:
        return predict_tdm

        
def make_problem(num_days=2, D = 5):

    doses_per_day = 2
    hours_per_dose = 12
    
    theta = dict(age=40, sex=0, weight=87, creatinine=95, cl=2.07, ke=0.13, ka=0.68)

    dose_times = np.arange(0, num_days*doses_per_day*hours_per_dose + 0.1, hours_per_dose)
    dose_size = np.tile(D, dose_times.size)

    decision_point = int(len(dose_times)/2)
    tobs = np.random.uniform(dose_times[decision_point-1], dose_times[decision_point], size = (1,))
    yobs = observe(tobs, theta, dose_times, dose_size, return_truth=False)

    predict = fit(tobs, yobs, theta, dose_times, dose_size)

    return (theta, dose_times, dose_size, tobs, yobs, predict)

def setup_experiment(dose,num_days=10, doses_per_day=2, hours_per_dose=12):

    # The last time the subejct could take a dose.  End simulation just before this time.
    tmax = hours_per_dose * doses_per_day * num_days
    step = 0.5
    tpred = np.arange(step, tmax+step, step)
    # Spread doses out over time
    dose_times = np.arange(0, tmax, hours_per_dose)
    dose_sizes = np.tile(dose, dose_times.size)
    # We get to make a decision about the size of the dose at the half way mark
    decision_point = int(len(dose_times) / 2)

    return tpred, dose_times, dose_sizes, decision_point