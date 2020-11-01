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
_prior_model = cmdstanpy.CmdStanModel(exe_file = 'experiment_models/prior_predictive')

# This model fits to simulated data.  Give it covariates and observed data and it will give you predictions.
_conditioning_model = cmdstanpy.CmdStanModel(exe_file = 'experiment_models/condition_on_patients')

class SimulatedSubject(object):

    def __init__(self, pk_params):

        self.age = pk_params['age']
        self.sex = pk_params['sex']
        self.weight = pk_params['weight']
        self.creatinine = pk_params['creatinine']
        self.cl = pk_params['cl']
        self.ke = pk_params['ke']
        self.ka = pk_params['ka']

        # likelihood.  Gives concentrations for repeated doses
        self.observe_func = repeated_dose_concentration(self.cl, self.ke, self.ka)

        # Data used in prior models and conditioning models
        self.model_data = _params.copy()
        self.model_data['sex'] = self.sex
        self.model_data['age'] = self.age
        self.model_data['weight'] = self.weight
        self.model_data['creatinine'] = self.creatinine

        self._scheduled_flag = False


    def schedule_doses(self, dose_times, doses):

        self.dose_times = validate_input(dose_times)
        self.doses = validate_input(doses)
        self._scheduled_flag = True

        self.model_data['n_doses'] = self.dose_times.size
        self.model_data['dose_times'] = self.dose_times.tolist()
        self.model_data['doses'] = self.doses.tolist()

        self.flag = True

        return self

    def observe(self, observed_times, return_true = True):

        if not self._scheduled_flag:
            raise ValueError('Doses not yet scheduled')
        
        times = validate_input(observed_times)

        true_concentrations = self.observe_func(times, self.dose_times, self.doses)

        # The data generating process needs to be fixed.
        # Sample from the prior with a set seed for reproducibility
        # Then generate lognormal random variables (the hard way)
        # Where the true concentrations are the mean on the natural scale (log concentration is mean on log scale)
        sigma = gamma(a=_params['shape_sigma'], scale=1.0/_params['rate_sigma']).rvs(1, random_state=0)

        observed_concentrations = np.exp(np.log(true_concentrations) + np.random.normal(size = times.size)*sigma)

        if return_true:
            return observed_concentrations, true_concentrations
        else:
            return observed_concentrations

    def fit(self, t, y):

        if not self._scheduled_flag:
            raise ValueError('Doses not yet scheduled')
        
        times = validate_input(t)
        yobs = validate_input(y)

        self.fit_data = self.model_data.copy()

        self.fit_data['n'] = times.size
        self.fit_data['observed_times'] = times.tolist()

        self.fit_data['observed_concentrations'] = yobs.tolist()
        self.fit_data['nt'] = times.size
        self.fit_data['prediction_times'] = times


        self.model_fit = _conditioning_model.sample(self.fit_data, show_progress=False)

    def predict(self, t):

        if not self._scheduled_flag:
            raise ValueError('Doses not yet scheduled')
        times = validate_input(t)
        
        # Problem is here.  When I update dose schedule, .predict does not see this.
        self.prediction_data = self.model_data.copy()

        # These are not used by the generated quantities block
        # _condition_model expects then to be passed, but the prediction does not use these
        # Pass something so the model can atleast run.
        self.prediction_data['n'] = times.size
        self.prediction_data['observed_times'] = times.tolist()
        self.prediction_data['observed_concentrations'] = times.tolist()

        # The prediction DOES use this stuff, so it us important these are correct.
        self.prediction_data['nt'] = times.size
        self.prediction_data['prediction_times'] = times.tolist()

        return _conditioning_model.generate_quantities(self.prediction_data, self.model_fit).generated_quantities

    def prior_predict(self, t, with_noise = False):

        if not self._scheduled_flag:
            raise ValueError('Doses not yet scheduled')

        self.prediction_data = self.model_data.copy()
        times = validate_input(t)
        self.prediction_data['nt'] = times.size
        self.prediction_data['prediction_times'] = times.tolist()

        if with_noise:
            return _prior_model.sample(self.prediction_data, fixed_param = True, iter_sampling=2000).stan_variable("C_noise")
        else:
            return _prior_model.sample(self.prediction_data, fixed_param = True, iter_sampling=2000).stan_variable("C")



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