import cmdstanpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scripts.tools.simulation_tools import repeated_dose_concentration, observe, fit, prior_predict
from tqdm import tqdm

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


def test_stan_concentration_function():

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

def test_prediction_function():

    # Doses twice per day over two days
    num_days = 2
    doses_per_day = 2
    hours_per_dose = 12

    # Covars for subject.
    theta = dict(age=40, sex=0, weight=87, creatinine=95, cl=2.07, ke=0.13, ka=0.68)

    # Set up dose times
    dose_times = np.arange(0, num_days*doses_per_day*hours_per_dose, hours_per_dose)
    # Dose to take.
    D = 5
    dose_size = np.tile(D, dose_times.size)

    
    tobs = [16]
    yobs = observe(tobs, theta, dose_times, dose_size, return_truth = False)

    predict = fit(tobs, yobs, theta, dose_times, dose_size)

    # Ok, here is what we are testing.  I can predict one of two ways:
    # The first way would be to predict to a time t from t=0.  
    # Here we assume C0 = 0 i.e. no drug in the blood.
    tt = np.arange(12, 36, 0.25)
    y_pred_1 = predict(tt, dose_times, dose_size)
    # In order to facilitate the optimization, predict returns a tuple for concentration-time profiles for the inital condition (c0*exp(-ke*t))
    # and another profile for  the concentration function D*F*ke*ka/ (Cl*(ke - ka))(exp(-ka*t) - exp(-ke*t)).
    # Note that the sum of these two profiles constitutes a solution to the initial value problem y' = f(t,y) y(0) = c_0
    # This is important because I can epress different doses with different initial conditions, we we do next.
    y_pred_1 = (y_pred_1[0] + y_pred_1[1]).mean(0)

    #Note here that the dose schedule is 1 mg, but since I have decomposed the solution into two parts, the latter of which is linear in D
    # then I can simply multiply both parts by the appropriate dose size and recover the same solution.
    dose_size = np.ones_like(dose_times)
    y_pred_2 = predict(tt-12, dose_times, dose_size, 12) # tt-12 to start at 0, when we give a dose
    y_pred_2 = (5*y_pred_2[0] + 5*y_pred_2[1]).mean(0)

    np.testing.assert_allclose(y_pred_1, y_pred_2, rtol=1e-3)

