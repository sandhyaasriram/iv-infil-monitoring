"""
Layer 2: Biomechanical tissue swelling and ideal FSR pressure response.

Supports:
1. Nonlinear compliance (V2 default, grounded in soft-tissue biomechanics & Bicen et al. 2018):
   - J-shaped skin stretch curve with toe-region compliance at small volumes (<0.5 mL)
   - Rapid radial surface deflection from 1 to 5 mL
   - Sublinear saturation at >5 mL due to collagen limit and fascial dissection
2. Linear model (V1 legacy mode for backward compatibility and model ablation)
"""
import numpy as np
from simulator.parameters import (
    KS, KP, P0, S_MAX, V_HALF, HILL_EXPONENT, ETA_VISCOUS, DT,
)


def simulate_swelling(VL, model="nonlinear"):
    """
    Simulates skin surface displacement / strain from accumulated extravasated volume.

    Parameters
    ----------
    VL : np.ndarray
        Accumulated leaked volume (mL) over time.
    model : str, {"nonlinear", "linear"}
        "nonlinear": Hill-formulated soft-tissue hyperelastic compliance curve.
        "linear": Legacy V1 proportional model.

    Returns
    -------
    S : np.ndarray, surface deformation (mm)
    """
    if model == "linear":
        return KS * VL

    # Nonlinear Hill compliance curve
    vl_safe = np.maximum(VL, 0.0)
    vl_pow = np.power(vl_safe, HILL_EXPONENT)
    vhalf_pow = np.power(V_HALF, HILL_EXPONENT)
    s_nonlinear = S_MAX * (vl_pow / (vhalf_pow + vl_pow + 1e-12))
    return s_nonlinear


def simulate_pressure_ideal(VL, model="nonlinear", dt=DT):
    """
    Simulates ideal (noise-free) FSR force sensor response (ADC-equivalent counts).
    Includes elastic displacement plus viscoelastic strain-rate resistance:
        P(t) = P0 + Kp * S(t) + eta * dS/dt
    """
    S = simulate_swelling(VL, model=model)
    if model == "linear":
        return P0 + KP * S

    # Rate of swelling (viscoelastic tissue resistance)
    dS_dt = np.gradient(S, dt)
    visco_term = ETA_VISCOUS * dS_dt
    P = P0 + KP * S + visco_term
    return np.maximum(P, 0.0)

