from scipy.stats import norm, gamma
import pandas as pd


def fit_norm(draws : pd.DataFrame, params : dict) -> dict:
    '''
    Take a list of parameters from a cmdstanfit and summarize then using a normal distribution.
    Save each summary parameter to a dictionary for later use.

    Inputs:
        draws : pd.DataFrame - output from model.draws_as_dataframe().  Contains draws for each parmeter

    Outputs:
        param_dict : dict - dictionary containing summary parameters.  Naming scheme is (summary parameter)_(model parameter)
                            (e.g mean_mu_cl, sd_mu_cl, mean_mu_tmax, etc.)
    '''
    param_dict = {}
    for theta in params:
        mu, sd = norm.fit(draws[theta])
        dict_theta = theta.replace(".", '_')
        param_dict[f'mean_{dict_theta}'] = mu
        param_dict[f'sd_{dict_theta}'] = sd
        
    return param_dict



def fit_gamma(draws : pd.DataFrame, params : dict) -> dict:
    '''
    Take a list of parameters from a cmdstanfit and summarize then using a gamma distribution.
    Save each summary parameter to a dictionary for later use.

    Inputs:
        draws : pd.DataFrame - output from model.draws_as_dataframe().  Contains draws for each parmeter

    Outputs:
        param_dict : dict - dictionary containing summary parameters.  Naming scheme is (summary parameter)_(model parameter)
                            (e.g rate_s_cl, shape_s_cl.)
    '''
    param_dict = {}

    #Only gotcha here is that scipy uses a weird parameterization.  Stan needs rate which is 1/scale according to wikipedia.
    for theta in params:
        a, _, scale = gamma.fit(draws[theta], floc = 0)
        dict_theta = theta.replace(".", '_')
        param_dict[f'shape_{dict_theta}'] = a
        param_dict[f'rate_{dict_theta}'] = 1/scale
        
    return param_dict