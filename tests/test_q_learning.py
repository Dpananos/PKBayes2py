import numpy as np
from scripts.tools.q_learning import *


def test_get_next_dose_time():

    dose_times = np.array([0, 12, 24, 36, 48])
    tobs = 16.4

    expected_next_dose_time = 24
    next_dose_time = get_next_dose_time(tobs, dose_times)

    assert next_dose_time==expected_next_dose_time
    assert type(next_dose_time)==float