"""
Layer 2: tissue swelling (V1 LINEAR model, per spec) and ideal FSR pressure response.

    S(t) = ks * VL(t)
    P(t) = P0 + kp * S(t)          (sensor noise added separately in sensors/noise.py)

Known V1 simplification: real tissue compliance is nonlinear (resists more as it
stretches). Document this explicitly as a stated limitation; plan to replace with
a nonlinear compliance term (e.g. saturating function of VL) in V2 once phantom
data shows whether linearity is actually a bad assumption.
"""
from simulator.parameters import KS, KP, P0


def simulate_swelling(VL):
    """VL: np.ndarray of leaked volume (mL) -> swelling S (mm, arbitrary unit)."""
    return KS * VL


def simulate_pressure_ideal(VL):
    """Ideal (noise-free) FSR reading, ADC-equivalent counts."""
    S = simulate_swelling(VL)
    return P0 + KP * S
