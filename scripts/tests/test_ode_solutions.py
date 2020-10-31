import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from ..tools.subject import repeated_dose_concentration

def test_analytical_repeated_dose_function():
    
    # Test if the functions I have wrote to compute the concentrations in python
    # are sufficiently close to the numerical solutions of the ODEs.
    # If this test fails, the simulations are affected since the observed concentrations
    # rely  tools.subjcts.repear_dose_concentration
    
    t = np.arange(0, 49, 0.25)
    cl = 3.0
    ke = 0.4
    ka = 1.0
    dose_times = np.arange(0, 48+1, 12)
    doses = 2.5*np.ones_like(dose_times)
    repeat_dose_conc = repeated_dose_concentration(cl, ke, ka)

    asol = repeat_dose_conc(t, dose_times, doses)
    
    def f(y,t):
        cl = 3.0
        ke = 0.4
        ka = 1.
        dose_times = np.arange(0, 48+1, 12)
        doses = 2.5*np.ones_like(dose_times)
        dy = -ke*y
        for tau,D in zip(dose_times, doses):
            dy += np.heaviside(t - tau, 0)*(D/cl)*ke*ka*np.exp(-ka*(t-tau))

        return dy

    sol = odeint(f, y0 = [0], t = t).ravel()
    
    np.testing.assert_allclose(asol, sol, rtol = 1e-4)