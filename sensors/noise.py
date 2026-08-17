"""
Sensor-emulation layer (project Section 2.8): injects realistic noise, drift,
quantization, and transient artifacts onto the ideal simulated physics signals.
Also supports the "missing sensor" robustness scenarios from Section 2.10.

    y(t) = x(t) + n(t) + d(t) + a(t)

This layer is what makes the synthetic dataset a genuine IoT robustness problem
rather than a toy clean-signal classification task.
"""
import numpy as np
from simulator.parameters import (
    DS18B20_NOISE_STD, DS18B20_RESOLUTION, DS18B20_DRIFT_RATE,
    FSR_NOISE_STD, FSR_ADC_MAX, FSR_MOTION_ARTIFACT_PROB,
)


def emulate_ds18b20(temp_ideal, dt, noise_level="normal", rng=None):
    """
    noise_level: "normal" or "high" (for the noisy_temp robustness scenario)
    """
    rng = rng or np.random.default_rng()
    n = len(temp_ideal)
    noise_std = DS18B20_NOISE_STD * (3.0 if noise_level == "high" else 1.0)

    noise = rng.normal(0, noise_std, n)
    drift = np.cumsum(rng.normal(0, DS18B20_DRIFT_RATE, n)) * dt
    raw = temp_ideal + noise + drift

    quantized = np.round(raw / DS18B20_RESOLUTION) * DS18B20_RESOLUTION
    return quantized


def emulate_fsr(pressure_ideal, noise_level="normal", rng=None):
    """
    noise_level: "normal" or "high" (for the noisy_pressure robustness scenario)
    """
    rng = rng or np.random.default_rng()
    n = len(pressure_ideal)
    noise_std = FSR_NOISE_STD * (3.0 if noise_level == "high" else 1.0)

    noise = rng.normal(0, noise_std, n)
    raw = pressure_ideal + noise

    # occasional motion-artifact spikes (patient arm movement)
    spikes = rng.random(n) < FSR_MOTION_ARTIFACT_PROB
    raw[spikes] += rng.normal(0, 40, spikes.sum())

    return np.clip(np.round(raw), 0, FSR_ADC_MAX)


def apply_missing_sensor(signal, mode="none"):
    """
    mode: "none" (pass through unchanged) or "missing" (returns NaN array,
    same length -- for the missing-sensor robustness scenario, Section 2.10).
    """
    if mode == "missing":
        return np.full_like(signal, np.nan, dtype=float)
    return signal
