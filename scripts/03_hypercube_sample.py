import cmdstanpy
import pickle
import pandas as pd
import numpy as np
from smt.sampling_methods import LHS
from itertools import product


# The point of this script is to generate some training data to estimate the reward in the Q learning step
# The reward is expensive to compute, so instead we will simulate some fake data and then train a ML model.
# I'm going to sample N_sample covariates using Latin Hypercube Sampling (LHS)

N_samples = 1000

# LHS requires some limits in order to sample from the covariate space.
# Set up the dose times and frequency.  Will need these to pass to LHS
tpred, dose_times, dose_sizes, decision_point = setup_experiment(1, num_days=10, doses_per_day=2, hours_per_dose=12)

# These limits are the mins/max of the actual data.
age_lims = [26.0, 70.0]
weight_lims = [54.7, 136.6]
creatinine_lims = [50, 95]
sex_lims = [0, 1]
tpred_lims = [dose_times[decision_point - 1], dose_times[decision_point]]
dose_lims = [1, 20.0]
lims = np.array([age_lims, weight_lims, creatinine_lims, sex_lims, tpred_lims, dose_lims])

# Do latin hypercube sampling
sampling = LHS(xlimits=lims)
domain = sampling(N_samples)

# Put the LH samples into a dataframe.  Sex is a continuous variable from the sampling, so we have to round to be binary
# (remember, sex is an indicator 1 -- male)
colnames = ["age", "weight", "creatinine", "sex", "tpred", "D"]
domain_df = pd.DataFrame(domain, columns=colnames).assign(sex=lambda x: x.sex.round())

# Now write to csv for later use
domain_df.to_csv('data/generated_data/hypercube_sampled_covariates.csv', index=False)