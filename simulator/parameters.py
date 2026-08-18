"""
Global physical and sensor parameters for the IV infiltration simulator (V1).

IMPORTANT: these are documented starting assumptions, NOT calibrated physiological
constants. Every value marked (TUNE) should be recalibrated once you have real
phantom-arm data (Step 8-10 of the project plan).
"""

# --- Time settings ---
DT = 1.0            # simulation timestep, seconds (matches ~1 Hz practical DS18B20 sampling)
DURATION_S = 1800   # 30 minutes per simulated trial

# --- Infusion / leakage rate ranges (mL/h) ---
# Reference ranges loosely based on the 2021 wearable-system study (normal ~10-60 mL/h,
# extravasation ~1-15 mL/h). Treat as experimental reference ranges, not clinical thresholds.
QL_RANGE = {
    "none":     (0, 0),
    "early":    (1, 3),
    "moderate": (3, 8),
    "severe":   (8, 15),
}

# --- Tissue swelling model (V1: LINEAR, per spec) ---
#Trial 0 
# KS = 0.02     # swelling sensitivity: mm deformation per mL leaked                    (TUNE)
# KP = 5.0      # pressure sensor gain: ADC-equivalent counts per mm swelling            (TUNE)
# P0 = 50.0     # baseline FSR reading at rest, ADC-equivalent counts                    (TUNE)

#Trial 1
KS = 0.08     # increased swelling sensitivity 
KP = 15.0     # increased pressure sensor gain
P0 = 50.0     # baseline - retained 

# --- Thermal model (V1: 0D LUMPED, per spec) ---
TA = 25.0     # ambient temperature, deg C
T0 = 33.0     # initial SKIN-SURFACE tissue temperature, deg C (NOT core temp 37C -- common mistake)
TI = 22.0     # infusion fluid temperature, deg C (room-temp saline)
TAU = 120.0   # thermal recovery time constant, seconds                               (TUNE)
# KL = 0.15   # leakage-to-temperature coupling coefficient                           (TUNE)
KL = 0.6      # increased for more pronounced thermal response

# --- Sensor noise / drift characteristics (DS18B20 + FSR realistic specs) ---
DS18B20_NOISE_STD = 0.15        # deg C, random gaussian noise
DS18B20_RESOLUTION = 0.0625     # deg C, quantization step (12-bit DS18B20 mode)
DS18B20_DRIFT_RATE = 0.0005     # deg C per second, slow random-walk baseline drift

FSR_NOISE_STD = 3.0             # ADC-equivalent counts, random gaussian noise
FSR_ADC_MAX = 4095              # 12-bit ADC ceiling (ESP32 native ADC resolution)
FSR_MOTION_ARTIFACT_PROB = 0.01 # probability per timestep of a motion-induced spike

# --- Severity label mapping (loosely aligned to INS infiltration Grade 0-3 scale for V1) ---
SEVERITY_LABELS = {"none": 0, "early": 1, "moderate": 2, "severe": 3}
