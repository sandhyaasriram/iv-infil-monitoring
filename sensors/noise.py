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

    # 1. First-order thermal lag (low-pass filter)
    tau = 5.0 # thermal time constant for sensor casing
    alpha = dt / (tau + dt)
    temp_lag = np.zeros_like(temp_ideal)
    temp_lag[0] = temp_ideal[0]
    for i in range(1, n):
        temp_lag[i] = alpha * temp_ideal[i] + (1 - alpha) * temp_lag[i-1]

    # 2. Add noise and drift
    noise = rng.normal(0, noise_std, n)
    drift = np.cumsum(rng.normal(0, DS18B20_DRIFT_RATE, n)) * dt
    
    # 3. Accuracy offset clamp (+/- 0.5 C)
    accuracy_offset = rng.uniform(-0.5, 0.5) 
    
    raw = temp_lag + noise + drift + accuracy_offset

    # 4. Quantize to 0.0625 C steps
    quantized = np.round(raw / DS18B20_RESOLUTION) * DS18B20_RESOLUTION
    
    # 5. Occasional +/- 1 count jitter
    jitter_prob = 0.1
    jitter = rng.choice([-1, 0, 1], size=n, p=[jitter_prob/2, 1-jitter_prob, jitter_prob/2])
    quantized += jitter * DS18B20_RESOLUTION
    
    return quantized


def emulate_fsr(pressure_ideal, noise_level="normal", rng=None):
    """
    noise_level: "normal" or "high" (for the noisy_pressure robustness scenario)
    """
    rng = rng or np.random.default_rng()
    n = len(pressure_ideal)
    noise_std = FSR_NOISE_STD * (3.0 if noise_level == "high" else 1.0)

    # Power-law / log-log inspired response: R ~ a * F^(-b)
    # We map pressure_ideal (F) to an ADC value (which depends on R).
    # Since V_out = V_cc * R_pullup / (R_pullup + R_fsr), it's highly non-linear.
    # Let's approximate the final ADC response with a non-linear transform.
    # We'll treat pressure_ideal as a baseline linear component, and apply a power law.
    # To prevent log(0) issues, add a small offset.
    # Let's say adc_ideal = c1 * (pressure_ideal + 1)^c2
    c1 = 50.0 # base scale
    c2 = 0.8  # power law exponent
    adc_ideal = c1 * np.power(pressure_ideal + 1.0, c2)
    
    # Hysteresis (~10% loading/unloading)
    adc_hyst = np.zeros_like(adc_ideal)
    adc_hyst[0] = adc_ideal[0]
    for i in range(1, n):
        diff = adc_ideal[i] - adc_hyst[i-1]
        # if loading, follow closely. if unloading, lag behind
        if diff > 0:
            adc_hyst[i] = adc_hyst[i-1] + diff * 0.95
        else:
            adc_hyst[i] = adc_hyst[i-1] + diff * 0.85

    noise = rng.normal(0, noise_std, n)
    raw = adc_hyst + noise

    # occasional motion-artifact spikes (patient arm movement)
    spikes = rng.random(n) < FSR_MOTION_ARTIFACT_PROB
    raw[spikes] += rng.normal(0, 40, spikes.sum())

    return np.clip(np.round(raw), 0, FSR_ADC_MAX)


def apply_missing_sensor(signal, mode="none", rng=None):
    """
    mode: "none" (pass through unchanged) or "missing" (applies Markov burst dropout).
    """
    if mode == "missing":
        rng = rng or np.random.default_rng()
        n = len(signal)
        # 2-state Markov process for burst dropouts
        # state 0: OK, state 1: MISSING
        p_01 = 0.05  # Prob of transitioning from OK to MISSING
        p_10 = 0.20  # Prob of transitioning from MISSING to OK (keeps bursts short-ish)
        
        state = 0
        result = np.copy(signal)
        for i in range(n):
            if state == 0:
                if rng.random() < p_01:
                    state = 1
            else:
                if rng.random() < p_10:
                    state = 0
                    
            if state == 1:
                result[i] = np.nan
        return result
        
    return signal
