import cmdstanpy
import numpy as np
import pandas as pd

def fit_original_model():
    
    # There are a few places where I fit this model.  
    # Easiest if I just make it a function.  Less copy pasta
    
    concentration_data = pd.read_csv("data/generated_data/experiment.csv")

    subject_data = concentration_data.drop_duplicates(['subjectids'])

    model_data = dict(
        sex = subject_data.sex.tolist(),
        weight = subject_data.weight.tolist(),
        age = subject_data.age.tolist(),
        creatinine = subject_data.creatinine.tolist(),
        n_subjectids = subject_data.shape[0],
        D = subject_data.D.tolist(),

        subjectids = concentration_data.subjectids.tolist(),
        time = concentration_data.time.tolist(),
        yobs = concentration_data.yobs.tolist(),
        n = concentration_data.shape[0]
    )

    model = cmdstanpy.CmdStanModel(stan_file = 'experiment_models/original_model.stan')

    fit = model.sample(model_data, chains = 12, parallel_chains = 4, seed = 19920908, show_progress=True)
    
    return fit, model_data