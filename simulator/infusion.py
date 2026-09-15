"""
Layer 1: leakage/infusion volume accumulation.

    dVL/dt = QL(t) - QD(t)

QD (drainage/redistribution) is set to 0 for V1 short-duration trials, per the
project spec (Section 2.5). Revisit this simplification if you extend trial
duration beyond ~30-60 minutes.
"""
import numpy as np


def simulate_leaked_volume(ql_array_mlph, duration_s, dt):
    """
    Parameters
    ----------
    ql_array_mlph : np.ndarray
        Leak rate in mL/h at each timestep. Allows variable onset and dynamic rates.
    duration_s : float
    dt : float

    Returns
    -------
    t  : np.ndarray, time in seconds
    VL : np.ndarray, cumulative leaked volume in mL
    """
    n_steps = int(duration_s / dt)
    t = np.arange(n_steps) * dt
    ql_mlps = ql_array_mlph / 3600.0       # mL/h -> mL/s
    VL = np.cumsum(ql_mlps * dt)           # cumulative volume, QD = 0 assumption
    return t, VL
