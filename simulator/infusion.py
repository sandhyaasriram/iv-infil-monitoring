"""
Layer 1: Interstitial fluid accumulation with physiological mass balance.

    dVL/dt = QL(t) - QD(t)

Where QD(t) represents physiological lymphatic drainage and capillary redistribution:
    QD(t) = k_drain * VL(t)
"""
import numpy as np
from simulator.parameters import K_DRAIN_PER_S, DT


def simulate_leaked_volume(ql_array_mlph, duration_s, dt=DT, drainage=True):
    """
    Parameters
    ----------
    ql_array_mlph : np.ndarray
        Leak rate in mL/h at each timestep.
    duration_s : float
    dt : float
    drainage : bool, default True
        Whether to model physiological lymphatic clearance (Starling forces).

    Returns
    -------
    t  : np.ndarray, time in seconds
    VL : np.ndarray, cumulative leaked volume in mL
    """
    n_steps = int(duration_s / dt)
    t = np.arange(n_steps) * dt
    ql_mlps = ql_array_mlph / 3600.0       # mL/h -> mL/s

    if not drainage:
        VL = np.cumsum(ql_mlps * dt)
        return t, VL

    VL = np.zeros(n_steps)
    k_drain = K_DRAIN_PER_S
    for i in range(1, n_steps):
        # Forward Euler integration with positive non-negativity constraint
        dVL = (ql_mlps[i - 1] - k_drain * VL[i - 1]) * dt
        VL[i] = max(0.0, VL[i - 1] + dVL)

    return t, VL

