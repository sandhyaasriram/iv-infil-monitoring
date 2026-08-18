"""
Top-level script: runs the full simulation + sensor-emulation pipeline
(project Section 2.17) and saves a labeled CSV dataset ready for ML training.

Pipeline:
    Physics simulator -> Synthetic sensor signals -> Noise/drift/missing injection
    -> (this script stops here; filtering/normalization/windowing happens in your
        training notebook, kept separate so you can experiment with those steps
        without regenerating the raw dataset each time)

Usage:
    python generate_dataset.py
    (or run the equivalent cell in the Colab notebook -- see README.md)
"""
import numpy as np
import pandas as pd

from simulator.scenarios import generate_dataset
from sensors.noise import emulate_ds18b20, emulate_fsr, apply_missing_sensor


ROBUSTNESS_CONDITIONS = [
    "normal",           # clean signal, both sensors present
    "noisy_temp",       # temperature channel has elevated noise
    "noisy_pressure",   # pressure channel has elevated noise
    "missing_temp",     # temperature channel entirely missing (NaN)
    "missing_pressure", # pressure channel entirely missing (NaN)
]


def build_labeled_dataframe(n_per_class=50, seed=42):
    trials = generate_dataset(n_per_class=n_per_class, seed=seed)
    rng = np.random.default_rng(seed + 1)
    rows = []

    for trial_id, trial in enumerate(trials):
        # cycle through robustness conditions so every severity class gets
        # roughly equal coverage of each condition
        condition = ROBUSTNESS_CONDITIONS[trial_id % len(ROBUSTNESS_CONDITIONS)]

        temp_noise_level = "high" if condition == "noisy_temp" else "normal"
        press_noise_level = "high" if condition == "noisy_pressure" else "normal"

        temp_sensor = emulate_ds18b20(
            trial["temperature_ideal"], dt=1.0,
            noise_level=temp_noise_level, rng=rng,
        )
        press_sensor = emulate_fsr(
            trial["pressure_ideal"],
            noise_level=press_noise_level, rng=rng,
        )

        if condition == "missing_temp":
            temp_sensor = apply_missing_sensor(temp_sensor, "missing")
        if condition == "missing_pressure":
            press_sensor = apply_missing_sensor(press_sensor, "missing")

        for i, t in enumerate(trial["time"]):
            rows.append({
                "trial_id": trial_id,
                "time_s": t,
                "temperature_c": temp_sensor[i],
                "pressure_adc": press_sensor[i],
                "condition": condition,
                "severity": trial["severity"],          # trial's nominal class (static, legacy)
                "label": trial["dynamic_label"][i],       # USE THIS: per-timestep, volume-based label
                "_static_label_GT": trial["label"],       # old trial-level label, kept for comparison only
                # Ground-truth columns kept ONLY for your own analysis/plots.
                # Prefix with underscore as a visual reminder: never select
                # these as ML input features (project Section 2.16).
                "_leak_rate_mlph_GT": trial["leak_rate_mlph"],
                "_leaked_volume_ml_GT": trial["leaked_volume_ml"][i],
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_labeled_dataframe(n_per_class=50)
    df.to_csv("synthetic_iv_infiltration_dataset.csv", index=False)
    print(f"Saved {len(df)} rows across {df['trial_id'].nunique()} trials.")
    print(df.groupby(["severity", "condition"]).size())
