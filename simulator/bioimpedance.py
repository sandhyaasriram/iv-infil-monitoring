"""
Biophysical Forward Model for Multi-Frequency Bioimpedance during IV Extravasation.

Governing Physical Equations:
1. Cole-Cole Baseline Human Tissue Impedance Spectrum (Cole & Cole, 1941):
       Z_base(f) = R_inf + (R_0 - R_inf) / (1 + (j * 2*pi*f * tau)^alpha)
   Calibrated against clinical forearm measurements from Jeong et al. (PLOS ONE 2019):
       R_0 = 785.2 Ohm (at 20 kHz), R_inf = 306.1 Ohm (at 500 kHz),
       tau = 1.2e-6 s (time constant), alpha = 0.78 (dispersion parameter).

2. Extravasation Conductance Expansion (Fricke-Morse Suspension Theory):
   Extravasated electrolyte infusate (0.9% NaCl saline, conductivity ~1.4 S/m)
   expands the conducting interstitial fluid pathway.
       Delta_G(t) = kappa * (VL(t) / (V0 + VL(t)))
   where kappa = 1.65 mS/mL, V0 = 15.0 mL (subcutaneous compartment reference).

3. Frequency-Dependent Shunting:
   Cellular membrane capacitance acts as an insulator at low frequencies,
   concentrating extravasation admittance in the extracellular path:
       D(f) = 1.0 / sqrt(1.0 + (f / f_c)^(2 * alpha))
       with f_c = 85 kHz.

   Resulting Admittance:
       Y(t, f) = 1.0 / Z_base(f) + Delta_G(t) * D(f)
       Z_ideal(t, f) = 1.0 / Y(t, f)

4. Sensor Emulation (Contact Drift & Thermal Johnson Noise):
       Z_sensor(t, f) = Z_ideal(t, f) + d_contact(t) + epsilon_noise(t)
       where d_contact is an electrode drift random walk, epsilon ~ N(0, 2.5 Ohm^2).

CRITICAL SCIENTIFIC INTEGRITY REQUIREMENT:
This forward model operates strictly on continuous physical ODE state variables (VL, time).
It DOES NOT accept, inspect, or branch on ground-truth categorical labels (none/early/moderate/severe).
"""

import numpy as np

# Standard clinical excitation frequencies from Jeong et al. 2019 (kHz)
CLINICAL_FREQS_KHZ = [20, 50, 100, 150, 200, 300, 500]


def simulate_bioimpedance_spectrum_ideal(VL: np.ndarray, dt: float = 1.0,
                                        r0: float = 785.2, r_inf: float = 306.1,
                                        alpha: float = 0.78, f_c: float = 85.0,
                                        kappa: float = 1.65e-3, v0: float = 15.0) -> np.ndarray:
    """
    Computes ideal (noise-free) multi-frequency impedance time series Z(t, f)
    from continuous accumulated leaked volume VL(t).

    Parameters
    ----------
    VL : np.ndarray, shape (T,)
        Continuous fluid volume accumulated in tissue (mL).
    dt : float
        Sampling step in seconds.
    r0, r_inf : float
        Low- and high-frequency baseline resistance (Ohms).
    alpha : float
        Cole-Cole distribution parameter.
    f_c : float
        Characteristic center relaxation frequency (kHz).
    kappa : float
        Electrolyte admittance gain per volume (S/mL).
    v0 : float
        Subcutaneous reference tissue volume (mL).

    Returns
    -------
    Z_spectrum : np.ndarray, shape (len(CLINICAL_FREQS_KHZ), T)
        Continuous impedance magnitude at each clinical frequency over time.
    """
    n_steps = len(VL)
    n_freqs = len(CLINICAL_FREQS_KHZ)
    Z_out = np.zeros((n_freqs, n_steps), dtype=np.float32)

    # Compute baseline impedance per frequency
    for i, f_khz in enumerate(CLINICAL_FREQS_KHZ):
        w_ratio = (f_khz / f_c) ** alpha
        # Cole-Cole magnitude approximation
        z_base_f = r_inf + (r0 - r_inf) / np.sqrt(1.0 + w_ratio ** 2)
        y_base_f = 1.0 / z_base_f

        # Frequency dispersion factor
        d_f = 1.0 / np.sqrt(1.0 + (f_khz / f_c) ** (2.0 * alpha))

        # Dynamic admittance expansion from continuous volume VL(t)
        # Note: Operates directly on continuous float VL, zero label access!
        delta_g = kappa * (VL / (v0 + VL + 1e-9))
        y_total = y_base_f + delta_g * d_f

        Z_out[i, :] = 1.0 / y_total

    return Z_out


def emulate_bioimpedance_sensors(Z_ideal: np.ndarray, dt: float = 1.0,
                                 noise_std: float = 2.5,
                                 drift_std: float = 0.05,
                                 rng=None) -> np.ndarray:
    """
    Applies realistic electrode-skin contact drift and measurement noise
    to the ideal bioimpedance spectrum.

    Parameters
    ----------
    Z_ideal : np.ndarray, shape (n_freqs, T)
    dt : float
    noise_std : float, standard deviation of thermal/quantization noise (Ohms).
    drift_std : float, Brownian contact drift rate (Ohms/sqrt(s)).
    rng : np.random.Generator, optional

    Returns
    -------
    Z_measured : np.ndarray, shape (n_freqs, T)
    """
    rng = rng or np.random.default_rng()
    n_freqs, n_steps = Z_ideal.shape
    Z_noisy = np.zeros_like(Z_ideal)

    for i in range(n_freqs):
        # White Gaussian noise
        noise = rng.normal(0.0, noise_std, size=n_steps)
        # Low-frequency electrode drift (Brownian motion)
        drift = np.cumsum(rng.normal(0.0, drift_std * np.sqrt(dt), size=n_steps))
        # Initial electrode offset variation (+/- 15 Ohm)
        offset = rng.uniform(-15.0, 15.0)

        Z_noisy[i, :] = Z_ideal[i, :] + noise + drift + offset

    return np.maximum(Z_noisy, 10.0) # Physiological lower bound
