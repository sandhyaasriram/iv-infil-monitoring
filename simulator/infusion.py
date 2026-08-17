"""
Layer 1: leakage/infusion volume accumulation.

    dVL/dt = QL(t) - QD(t)

QD (drainage/redistribution) is set to 0 for V1 short-duration trials, per the
project spec (Section 2.5). Revisit this simplification if you extend trial
duration beyond ~30-60 minutes.
"""
import numpy as np


def simulate_leaked_volume(ql_rate_mlph, duration_s, dt):
    """
    Parameters
    ----------
    ql_rate_mlph : float
        Constant leak rate in mL/h for this trial. V1 uses a constant rate per
        trial; a natural V2 extension is to let this ramp or vary over time.
    duration_s : float
    dt : float

    Returns
    -------
    t  : np.ndarray, time in seconds
    VL : np.ndarray, cumulative leaked volume in mL
    """
    n_steps = int(duration_s / dt)
    t = np.arange(n_steps) * dt
    ql_mlps = ql_rate_mlph / 3600.0       # mL/h -> mL/s
    VL = ql_mlps * t                       # cumulative volume, QD = 0 assumption
    return t, VL
