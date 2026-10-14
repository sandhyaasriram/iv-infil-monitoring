"""
Layer 3: Pennes Bioheat Convective Cooling Model with Systemic Vasomotor Coupling.

Governing dynamic heat equation:
    dT/dt = (T_target(t) - T)/tau_perf(t) + K_conv * QL(t) * (TI - T)

Where:
- T_target(t) represents the local perfusion equilibrium temperature. Under resting
  conditions, this equals T0 (skin baseline ~33 C).
- Under sympathetic nervous activation (stress, pain, hypothermia, vasoconstriction),
  peripheral blood flow is reduced, dropping T_target by up to DELTA_T_VASOCONSTRICTION_MAX.
- K_conv is grounded in convective heat transport: rho_f * c_f / (rho_t * c_t * V_eff).
"""
import numpy as np
from simulator.parameters import (
    T0, TI, TAU, KL, DT, DELTA_T_VASOCONSTRICTION_MAX,
)


def simulate_temperature(ql_array_mlph, duration_s, dt=DT, vasoconstriction_profile=None):
    """
    Parameters
    ----------
    ql_array_mlph : np.ndarray
        Leak rate at each timestep (mL/h)
    duration_s : float
    dt : float, Euler integration timestep (s)
    vasoconstriction_profile : np.ndarray or None
        Normalized autonomic vasoconstriction intensity [0.0, 1.0] over time.
        0.0 = normothermic resting baseline
        1.0 = peak sympathetic vasoconstriction (cold/stress induced)

    Returns
    -------
    T : np.ndarray, simulated skin-surface temperature (deg C) over time
    """
    n_steps = int(duration_s / dt)
    T = np.zeros(n_steps)
    T[0] = T0
    ql_mlps = ql_array_mlph / 3600.0

    if vasoconstriction_profile is None:
        vaso = np.zeros(n_steps)
    else:
        vaso = np.asarray(vasoconstriction_profile)
        if len(vaso) < n_steps:
            vaso = np.pad(vaso, (0, n_steps - len(vaso)), mode='edge')
        elif len(vaso) > n_steps:
            vaso = vaso[:n_steps]

    for i in range(1, n_steps):
        # Autonomic baseline modulation
        t_base_i = T0 - DELTA_T_VASOCONSTRICTION_MAX * vaso[i - 1]
        tau_eff = TAU * (1.0 + 0.5 * vaso[i - 1]) # reduced perfusion slows recovery

        # Convective heat sink + perfusion recovery
        dT_perf = (t_base_i - T[i - 1]) / tau_eff
        dT_leak = KL * ql_mlps[i - 1] * (TI - T[i - 1])
        dT = dT_perf + dT_leak
        T[i] = T[i - 1] + dT * dt

    return T

