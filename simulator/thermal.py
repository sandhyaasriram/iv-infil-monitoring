"""
Layer 3: 0D LUMPED thermal model (V1, per spec).

    dT/dt = (T0 - T)/tau + KL * QL(t) * (TI - T)

NOTE on a bug caught during sanity-checking: the recovery term pulls tissue
temperature back toward its own baseline T0 (perfusion-maintained skin temp),
NOT toward ambient TA. Skin is thermoregulated by blood perfusion, so with no
leak it should hold near T0 indefinitely, not drift toward room temperature
over 30 minutes. Using TA as the recovery target made every severity class
converge to the same value regardless of leak rate -- caught by plotting
temperature-vs-time across severities before trusting the model (do this
check yourself, on any parameter changes you make).

This is a single lumped-parameter ODE (no spatial resolution), consistent with
V1's feature-engineering-based ML approach, which doesn't need spatial detail.
A full 2D Pennes bioheat model is a valid V2 refinement if you later need
sensor-placement-sensitivity results, but don't build both in parallel now.

Explicit Euler integration, dt should match DS18B20 practical sampling interval.
"""
import numpy as np
from simulator.parameters import T0, TI, TAU, KL, DT


def simulate_temperature(ql_rate_mlph, duration_s, dt=DT):
    """
    Parameters
    ----------
    ql_rate_mlph : float, constant leak rate for this trial (mL/h)
    duration_s   : float
    dt           : float, Euler integration timestep (s)

    Returns
    -------
    T : np.ndarray, simulated skin-surface temperature (deg C) over time
    """
    n_steps = int(duration_s / dt)
    T = np.zeros(n_steps)
    T[0] = T0
    ql_mlps = ql_rate_mlph / 3600.0

    for i in range(1, n_steps):
        dT = (T0 - T[i - 1]) / TAU + KL * ql_mlps * (TI - T[i - 1])
        T[i] = T[i - 1] + dT * dt

    return T
