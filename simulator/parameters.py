"""
Global physical and sensor parameters for the IV infiltration simulator (V1).

IMPORTANT: these are documented starting assumptions, NOT calibrated physiological
constants. Every value marked (TUNE) should be recalibrated once you have real
phantom-arm data (Step 8-10 of the project plan). State this explicitly in the paper.
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
# v2 update: KS/KP raised ~4x/3x after ablation showed pressure_only F1 ~0.16
# (near-random) -- ideal signal amplitude was ~0.45 counts vs FSR_NOISE_STD=3.0.
KS = 0.08     # swelling sensitivity: mm deformation per mL leaked                    (TUNE)
KP = 15.0     # pressure sensor gain: ADC-equivalent counts per mm swelling            (TUNE)
P0 = 50.0     # baseline FSR reading at rest, ADC-equivalent counts                    (TUNE)

# --- Thermal model (V1: 0D LUMPED, per spec) ---
TA = 25.0     # ambient temperature, deg C
T0 = 33.0     # initial SKIN-SURFACE tissue temperature, deg C (NOT core temp 37C -- common mistake)
TI = 22.0     # infusion fluid temperature, deg C (room-temp saline)
TAU = 120.0   # thermal recovery time constant, seconds                               (TUNE)
# v2 update: KL raised 4x -- moderate/severe temperature curves were separated by
# only ~0.06 C at v1 values, smaller than DS18B20's own 0.0625 C quantization step.
KL = 0.6      # leakage-to-temperature coupling coefficient                           (TUNE)

# --- Sensor noise / drift characteristics (DS18B20 + FSR realistic specs) ---
DS18B20_NOISE_STD = 0.15        # deg C, random gaussian noise
DS18B20_RESOLUTION = 0.0625     # deg C, quantization step (12-bit DS18B20 mode)
DS18B20_DRIFT_RATE = 0.0005     # deg C per second, slow random-walk baseline drift

FSR_NOISE_STD = 3.0             # ADC-equivalent counts, random gaussian noise
FSR_ADC_MAX = 4095              # 12-bit ADC ceiling (ESP32 native ADC resolution)
FSR_MOTION_ARTIFACT_PROB = 0.01 # probability per timestep of a motion-induced spike

# --- Severity label mapping (loosely aligned to INS infiltration Grade 0-3 scale for V1) ---
SEVERITY_LABELS = {"none": 0, "early": 1, "moderate": 2, "severe": 3}

# --- Dynamic (per-window) label thresholds, by accumulated leaked volume (mL) ---
# V1 assigned the trial's overall severity to EVERY window, including windows at
# t=0 before any fluid had leaked -- physically indistinguishable from "none" but
# labeled "early/moderate/severe" anyway. This caused a real, diagnosable bias
# (early misclassified as none more than any other error). Fixed by labeling each
# window from its OWN accumulated volume at that point in time, so a trial's early
# windows correctly read as "none" and label only changes once volume crosses a
# threshold -- matching how infiltration actually progresses over time.
# Thresholds set relative to the QL_RANGE endpoints x DURATION_S (0.5h trial):
#   severe QL up to 15 mL/h -> ~7.5 mL by trial end; moderate up to 8 mL/h -> ~4 mL;
#   early up to 3 mL/h -> ~1.5 mL. Midpoints used as boundaries below.
VOLUME_LABEL_THRESHOLDS = {
    "early": 0.3,      # mL -- below this, treat as indistinguishable from none
    "moderate": 1.5,
    "severe": 4.0,
}
