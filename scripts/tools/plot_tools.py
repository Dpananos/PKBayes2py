import numpy as np 
import matplotlib.pyplot as plt
from .simulation_tools import prior_predict, observe, fit

def plot_course(tobs, theta, dose_times, dose_size):

    fig, ax = plt.subplots(dpi = 120, figsize = (20, 5))
    ax.set_xlabel('Hours Post Initial Dose')
    ax.set_ylabel('Concentration (mg/L)')
    ax.grid(True, zorder = 0)

    # Times over which to plot the predictions.
    t_predictions = np.arange(0.05, max(dose_times), 0.05)

    # What do we expect before observing?
    prior_predictions = prior_predict(t_predictions, theta, dose_times, dose_size)
    prior_E_y = prior_predictions.mean(0)
    prior_Q_5, prior_Q_95 = np.quantile(prior_predictions, q=[0.05, 0.95], axis = 0)

    # Now. observe the subject at tobs
    yobs, *_ = observe(tobs, theta, dose_times, dose_size)

    # Fit the model to the subject

    posterior_predict = fit(tobs, yobs, theta, dose_times, dose_size)
    posterior_predictions = posterior_predict(t_predictions, dose_times, dose_size)

    posterior_E_y = posterior_predictions.mean(0)
    posterior_Q_5, posterior_Q_95 = np.quantile(posterior_predictions, q=[0.05, 0.95], axis = 0)

    # Now, make some plots
    ax.plot(t_predictions, prior_E_y, color = 'C0', label = 'Prior')
    ax.fill_between(t_predictions, prior_Q_5, prior_Q_95, color = 'C0', alpha = 0.25)

    ax.plot(t_predictions, posterior_E_y, color = 'red', label = 'Posterior')
    ax.fill_between(t_predictions, posterior_Q_5, posterior_Q_95, color = 'red', alpha = 0.25)

    ax.scatter(tobs, yobs, c='black', zorder = 10)

    # Plot the truth
    *_, y_true = observe(t_predictions, theta, dose_times, dose_size)

    ax.plot(t_predictions, y_true, 'k--', label = 'Latent Concentration')

    ax.legend(loc = 'best')


    return ax