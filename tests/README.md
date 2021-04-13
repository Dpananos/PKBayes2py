# Unit Tests

This directory contains unit tests for various functions used within the study.  These tests are used not only to ensure implementations are correct, but also that any changes made are not breaking changes.

# `test_ode_solutions.py`

* `test_analytical_repeated_dose_function` compares my written helper functions `scripts.tools.simulation_tools.repeated_dose_concentration` to the numerical solution to the PK ODE.  Passing this test means my analytical helper function agrees with the numerical solution to within an acceptable relative error.

* `test_stan_concentration_function` compares the analytical repeated dosing function written in Stan to the helper function `scripts.tools.simulation_tools.repeated_dose_concentration`.  Passing this test means these two functions agree to within acceptable relative error.

# `test_prior_predictive.py`

* `test_prior_predictive` compares the posterior predictive distribution obtained after fitting the original model with my method for summarizing posteriors and passing them as priors.  Passing this test means that my priors for models like `experiment_models/prior_predictive.stan` and `experiment_models/condition_on_patients.stan` have the correct priors being passed to them, and consequently generate realistic prior predictions.