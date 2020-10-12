import cmdstanpy
import numpy as np
from scipy.stats import gamma
import pickle
import warnings

# Dictionary of parameters to be used in models.  Prevents me from having to 
# Manually enter prior params. See step 01
with open('data/param_summary.pkl','rb') as file:
    _params = pickle.load(file)

# This model is the prior predictive check.  Give it covariates and it will give you predictions
_prior_model = cmdstanpy.CmdStanModel(exe_file = 'experiment_models/generate_single_patient')

# This model fits to simulated data.  Give it covariates and observed data and it will give you predictions.
_conditioning_model = cmdstanpy.CmdStanModel(exe_file = 'experiment_models/condition_on_patients')

class SimulatedSubject():

    def __init__(self, pk_params):

        '''
        pk_params: dict - Dictionary housing patient covariates in addition to simulated PK parameters (see step 02)

        '''
        self.age = pk_params['age']
        self.sex = pk_params['sex']
        self.weight = pk_params['weight']
        self.creatinine = pk_params['creatinine']
        self.cl = pk_params['cl']
        self.ke = pk_params['ke']
        self.ka = pk_params['ka']

        # likelihood.  Gives concentrations for repeated doses
        self.observe_func = repeated_dose_concentration(self.cl, self.ke, self.ka)

        #Flags

        self._scheduled_flag = False


    def schedule_doses(self, dose_times, doses):

        self.dose_times = validate_input(dose_times)
        self.doses = validate_input(doses)
        self._scheduled_flag = True

    def observe(self, observed_times):

        if not self._scheduled_flag:
            raise ValueError('Doses not yet scheduled')
        
        times = validate_input(observed_times)

        true_concentrations = self.observe_func(times, self.dose_times, self.doses)

        sigma = gamma(a=_params['shape_sigma'], scale=1.0/_params['rate_sigma']).rvs(1, random_state =0)

        observed_concentrations = np.exp(np.log(true_concentrations) + np.random.normal(size = times.size)*sigma)

        return observed_concentrations, true_concentrations

    def make_model_data(self, t, y):

        times = validate_input(t)

        yobs = validate_input(y)

        self.model_data = _params.copy()

        self.model_data['n'] = times.size
        self.model_data['observed_times'] = times.tolist()
        self.model_data['observed_concentrations'] = yobs.tolist()

        self.model_data['sex'] = self.sex
        self.model_data['age'] = self.age
        self.model_data['weight'] = self.weight
        self.model_data['creatinine'] = self.creatinine

        self.model_data['nt'] = times.size
        self.model_data['prediction_times'] = times

        self.model_data['n_doses'] = self.dose_times.size
        self.model_data['dose_times'] = self.dose_times.tolist()
        self.model_data['doses'] = self.doses.tolist()

    def make_prediction_data(self, prediction_times):
     
        times = validate_input(prediction_times)

        self.prediction_data = self.model_data.copy()

        self.prediction_data['nt'] = times.size
        self.prediction_data['prediction_times'] = times.tolist()

    def fit(self, t, y):


        self.make_model_data(t, y)

        self.model_fit = _conditioning_model.sample(self.model_data, adapt_delta=0.99)

    def predict(self, t):

        self.make_prediction_data(t)

        return _conditioning_model.generate_quantities(self.prediction_data, self.model_fit).generated_quantities

    def prior_predict(self, t):

        self.make_prediction_data(t)

        return _prior_model.sample(self.prediction_data, fixed_param = True, iter_sampling = 500).stan_variable("C")


def validate_input(x):
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
    ka: float - Absorptionrate in units 1/hour
    '''

    y = (D * ke * ka)/(cl*(ke - ka)) * (np.exp(-ka*t) - np.exp(-ke*t))

    return np.heaviside(t, 0)*y


def repeated_dose_concentration(cl: float, ke: float, ka:float) -> callable:

    #TODO Comment this function and maybe write unit tests.

    def func(t: np.array, dose_times: np.array, doses: np.array) -> np.array:
        y = 0
        for d, tau in zip(doses, dose_times):
            y += concentration_function(d, t-tau, cl, ke, ka)

        return y
    return func