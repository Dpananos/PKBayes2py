import numpy as np
from scipy.stats import gamma

class SimulatedSubject():

    def __init__(self, cl, ke, ka):

        self.cl = cl
        self.ke = ke
        self.ka = ka

        self.observe_func = repeated_dose_concentration(self.cl, self.ke, self.ka)

        self.oberved_times = None
        self.true_concentration = None
        self.observed_concentration = None

    def observe(self, t_obs, dose_times, dose_size):
        
        self.observed_times = t_obs
        self.true_concentration = self.observe_func(t_obs, dose_times, dose_size)

        # Rediculous, right?  I got these from the estimates I made in step 01.
        sigma = gamma(a=365.0699706545347, scale = 1.0/2153.536256909682).rvs(1, random_state = 19920908)

        # Lognorm is such a pain in the ass and I this might come back to haunt me later because I can't control exactly what is returned
        # TODO Make this better.
        self.observed_concentration = np.exp(np.log(self.true_concentration) + sigma*np.random.normal(size = self.true_concentration.size))

    


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

    def func(t: np.array, dose_times: np.array, dose_size: np.array) -> np.array:
        y = 0
        for d, tau in zip(dose_size, dose_times):
            y += concentration_function(d, t-tau, cl, ke, ka)

        return y
    return func