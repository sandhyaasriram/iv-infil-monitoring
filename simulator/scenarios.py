"""
Scenario generator: produces labeled ideal (noise-free) trials across the four
severity classes defined in the project spec (Section 2.9): none / early /
moderate / severe. Sensor noise/drift/missing-data is applied downstream in
sensors/noise.py, kept separate so the ideal physics and the sensor emulation
can be validated independently.
"""
import numpy as np
from simulator.parameters import QL_RANGE, SEVERITY_LABELS, DURATION_S, DT
from simulator.infusion import simulate_leaked_volume
from simulator.tissue import simulate_pressure_ideal
from simulator.thermal import simulate_temperature


def generate_trial(severity, duration_s=DURATION_S, dt=DT, rng=None):
    """Generate one ideal trial for a given severity class."""
    rng = rng or np.random.default_rng()
    lo, hi = QL_RANGE[severity]
    ql_rate = 0.0 if hi == 0 else rng.uniform(lo, hi)

    t, VL = simulate_leaked_volume(ql_rate, duration_s, dt)
    P_ideal = simulate_pressure_ideal(VL)
    T_ideal = simulate_temperature(ql_rate, duration_s, dt)

    return {
        "time": t,
        "temperature_ideal": T_ideal,
        "pressure_ideal": P_ideal,
        # Hidden ground-truth variables -- keep for analysis/plots ONLY.
        # NEVER feed these to the ML model (project Section 2.16).
        "leak_rate_mlph": ql_rate,
        "leaked_volume_ml": VL,
        "severity": severity,
        "label": SEVERITY_LABELS[severity],
    }


def generate_dataset(n_per_class=50, duration_s=DURATION_S, dt=DT, seed=42):
    """Generate n_per_class ideal trials for every severity class."""
    rng = np.random.default_rng(seed)
    trials = []
    for severity in QL_RANGE.keys():
        for _ in range(n_per_class):
            trials.append(generate_trial(severity, duration_s, dt, rng))
    return trials
