import cmdstanpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from ..tools.simulation_tools import repeated_dose_concentration

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
            dy += np.heaviside(t - tau, 0)*(0.5*D/cl)*ke*ka*np.exp(-ka*(t-tau))

        return dy

    sol = odeint(f, y0 = [0], t = t).ravel()
    
    np.testing.assert_allclose(asol, sol, rtol = 1e-4)


def test_stan_ode_solution():

    # Test if my stan model and my analytical calculation fo the concentration function match up
    model_code='''
    functions{
    vector heaviside(vector t){
        
        vector[size(t)] y;
        for(i in 1:size(t)){
        y[i] = t[i]<=0 ? 0 : 1;
        }
        return y;
    }
    
    
    vector conc(real D, vector t, real Cl, real ka, real ke){
        return heaviside(t) .* (exp(-ka*t) - exp(-ke*t)) * (0.5 * D * ke * ka ) / (Cl *(ke - ka));
    }
    
    }
    data{
        int nt;
        vector[nt] t;
        real cl;
        real ka;
        real ke;
        
        int n_doses;
        vector[n_doses] dose_times;
        vector[n_doses] doses;
    }
    model{}
    generated quantities{
        vector[nt] C = rep_vector(0.0, nt);
        for(i in 1:n_doses){
        C += conc(doses[i], t - dose_times[i], cl, ka, ke);
    }
    }
    '''


    model_dir = '/tmp/test_model.stan'
    with open(model_dir, 'w') as f:
        f.write(model_code)
        

    t = np.arange(0, 48.05, 0.05).tolist()
    nt = len(t)
    cl = 3.3
    ka = 1.0
    ke = 0.2
    dose_times = np.arange(0, 48+12, 12).tolist()
    doses = np.tile(10, len(dose_times)).tolist()
    n_doses = len(dose_times)

    model = cmdstanpy.CmdStanModel(stan_file = model_dir)
    model_data = dict(t=t, cl=cl, ka=ka, ke=ke, dose_times=dose_times, doses=doses, nt=nt, n_doses=n_doses)

    conc_from_stan = model.sample(model_data, fixed_param=True, iter_sampling=1).stan_variable("C").ravel()

    repeat_dose_conc = repeated_dose_concentration(cl, ke, ka)
    conc_from_analytical = repeat_dose_conc(np.array(t), np.array(dose_times), np.array(doses)) 

    np.testing.assert_allclose(conc_from_analytical, conc_from_stan, rtol = 1e-4, atol = 1e-4)
