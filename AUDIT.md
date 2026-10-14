# System Audit: Edge-Based IV Infiltration Monitoring

**Audit Date:** October 2026  
**Audited Repository:** iv-infiltration-project (main branch)  
**Primary Target:** Final-year engineering research project on edge-based intravenous (IV) infiltration monitoring.

---

## 1. Executive Summary & Audit Answers

| Audit Dimension | Finding & Status |
| :--- | :--- |
| **Data Nature** | **100% Synthetic & Emulated.** No human clinical or biological bench trials exist in the repository. |
| **Pipeline Reproducibility** | **Reproducible.** generate_dataset.py and 	rain_and_export.py run end-to-end under Python 3.11 with .venv. |
| **Reported Performance** | **Severely optimistic (Train-Set Evaluation).** 	rain_and_export.py reports F1 = 0.98, but it tests directly on the training set. |
| **Data Leakage in Notebook** | iv_infil_v3.ipynb implements a disjoint trial split (	rain_test_split(unique_trials)), avoiding subject-leakage, but 	rain_and_export.py evaluates on train data with zero holdout. |
| **Physical Validity** | Highly simplified lumped 0D ODE thermal model and linear pressure expansion. Parameter coupling values were tuned manually to separate classes. |
| **Target Firmware Integration** | Embedded TinyML pipeline functions via m2cgen exporting C code to irmware/sketch/tinyml_model.h. |

---

## 2. Module-by-Module Code Analysis

### 2.1 generate_dataset.py
* **Function:** Top-level generator orchestrating the physics simulator and sensor noise emulator. Creates synthetic_iv_infiltration_dataset.csv (14,200 timesteps across 200 trials).
* **Execution Flow:**
  1. Calls simulator.scenarios.generate_dataset(n_per_class=50, seed=42).
  2. Rotates trials deterministically across 5 robustness conditions: 
ormal, 
oisy_temp, 
oisy_pressure, missing_temp, missing_pressure.
  3. Emulates DS18B20 digital temperature and FSR resistive force sensors.
  4. Stores per-timestep ground-truth volume and dynamic infiltration labels.

### 2.2 simulator/
* **parameters.py:** Holds simulation constants:
  * Timestep $\Delta t = 1.0\text{ s}$, duration $= 1800\text{ s}$ (30 min).
  * Infusion leak ranges: None ($), Early (\text{--}3\text{ mL/h}$), Moderate (\text{--}8\text{ mL/h}$), Severe (\text{--}15\text{ mL/h}$).
  * Baseline skin temperature  = 33^\circ\text{C}$, infusate  = 22^\circ\text{C}$ (room-temp saline), thermal recovery $\tau = 120\text{ s}$, thermal coupling  = 0.6$.
  * Tissue swelling linear sensitivity  = 0.08\text{ mm/mL}$, FSR gain  = 15.0\text{ ADC/mm}$, baseline ADC  = 50.0$.
* **infusion.py:** Implements cumulative fluid volume (t) = \int Q_L(t) dt$. Assumes physiological drainage/lymphatic redistribution (t) = 0$.
* **	issue.py:** Linear swelling (t) = K_S \cdot V_L(t)$ and linear pressure (t) = P_0 + K_P \cdot S(t)$. Real tissue compliance is known to be non-linear viscoelastic.
* **	hermal.py:** 0D lumped thermal ODE:
  \frac{dT}{dt} = \frac{T_0 - T}{\tau} + K_L \cdot Q_L(t) \cdot (T_i - T)
  Integrated via forward Euler. Correctly pulls recovery toward skin baseline $ rather than ambient $.
* **scenarios.py:** Assigns dynamic volume labels: None ($<0.3\text{ mL}$), Early (.3\text{--}1.5\text{ mL}$), Moderate (.5\text{--}4.0\text{ mL}$), Severe ($>4.0\text{ mL}$). Fixes the previous bias where =0$ in severe trials was labeled severe.

### 2.3 sensors/noise.py
* **emulate_ds18b20:** Models 1st-order casing thermal lag ($\tau=5\text{ s}$), Gaussian white noise ($\sigma=0.15^\circ\text{C}$), random-walk baseline drift (.0005^\circ\text{C/s}$), hardware offset clamp ($\pm 0.5^\circ\text{C}$), 12-bit quantization step (.0625^\circ\text{C}$), and bit jitter.
* **emulate_fsr:** Models power-law transfer curve ($\text{ADC} \propto (P+1)^{0.8}$), asymmetric mechanical hysteresis (.95$ loading vs .85$ unloading), Gaussian noise ($\sigma=3.0\text{ ADC counts}$), and stochastic motion spikes (=0.01$).
* **pply_missing_sensor:** 2-state Markov chain ({01}=0.05, p_{10}=0.20$) injecting burst packet dropouts (NaN).

### 2.4 train_and_export.py
* **Function:** Extracts sliding-window features (25-second non-overlapping windows) across temperature and pressure: mean, std, slope (polyfit), min, max, and missingness indicator.
* **Defect:** Fits a RandomForestClassifier(n_estimators=30, max_depth=5) and evaluates on the training data (clf.fit(X_imputed, y); y_pred = clf.predict(X_imputed)). Outputs a false \%$ macro-F1. Exports C decision tree code via m2cgen into irmware/sketch/tinyml_model.h.

### 2.5 gen_trace.py
* **Function:** Hardcodes a 100-sample mock trajectory of temperature and pressure points across classes and writes it directly into irmware/sketch/sketch.ino as a static array FALLBACK_TRACE for offline microcontroller demonstration without live sensors.

### 2.6 iv_infil_v3.ipynb
* **Function:** Colab experimentation notebook. Demonstrates Logistic Regression (F1 $\approx 0.62$) vs Random Forest (F1 $\approx 0.94$ on 25% held-out trials). Includes sensor ablation (pressure-only vs temp-only) and window sweep (\text{ s}, 25\text{ s}, 50\text{ s}, 100\text{ s}$).

---

## 3. Parameter Provenance & Physical Assumptions

| Parameter | Value | Status / Grounding | Physical Reality / Citation |
| :--- | :--- | :--- | :--- |
| **Leak Rates ($)** | , 1\text{--}3, 3\text{--}8, 8\text{--}15\text{ mL/h}$ | Experimental Assumption | Loosely inspired by pediatric/neonatal IV infusion ranges. Real adult IV infusions run \text{--}250\text{ mL/h}$; accidental extravasation is a fraction of pump rate. |
| **Fluid Temp ($)** | .0^\circ\text{C}$ | Reasonable physical constant | Room-temperature intravenous saline/Ringer's solution. |
| **Skin Temp Baseline ($)** | .0^\circ\text{C}$ | Clinically grounded | Typical peripheral extremity skin temperature (forearm/hand). |
| **Thermal Recovery ($\tau$)** | .0\text{ s}$ | Arbitrary / Estimated | Perfusion heat exchange time constant. Highly subject-dependent (vasoconstriction/vasodilation). |
| **Thermal Coupling ($)** | .6$ | **Arbitrary calibration** | Scaled up by \times$ specifically so temperature curves would separate by more than the DS18B20 quantization step. |
| **Tissue Compliance ($)** | .08\text{ mm/mL}$ | **Arbitrary calibration** | Scaled up \times$ in v2 to force pressure feature separation above FSR noise floor ($\sigma=3$). |
| **Drainage Rate ($)** | .0\text{ mL/s}$ | Unrealistic Assumption | Assumes edema fluid does not redistribute or absorb during a 30-minute window. |
| **Tissue Linearity** | Linear | Simplification | Real tissue swelling exhibits nonlinear, strain-hardening viscoelastic behavior. |

---

## 4. Windowing, Splitting, and Data Leakage Audit

1. **Window Construction:**
   * Uses non-overlapping 25-second windows ( \leftarrow start + 25$).
   * Label is chosen by 
p.bincount(l_win).argmax() (majority vote of dynamic labels).
   * Slopes are calculated with 
p.polyfit(..., nan_to_num(x, nan=np.nanmean(x)), 1).
2. **Train/Test Split:**
   * In 	rain_and_export.py: **CRITICAL DEFECT.** No split is performed. Train set is evaluated as test set.
   * In iv_infil_v3.ipynb: Correctly splits at the **trial level** (	rain_test_split(unique_trials, test_size=0.25)). Because windows from the same trial do not cross into the test set, window-correlation leakage is avoided in the notebook.
3. **Preprocessing Leakage:**
   * In 	rain_and_export.py, imputer is fitted on the whole dataset.
   * In the notebook, Pipeline([('impute', SimpleImputer()), ('scale', StandardScaler()), ('clf', ...)]) is correctly fitted on X_train only.

---

## 5. Output Characterization: Real vs Synthetic vs Mock

* **synthetic_iv_infiltration_dataset.csv:** **100% Synthetic.** Generated by forward ODE simulation with synthetic sensor noise models.
* **generated_trace.txt / FALLBACK_TRACE:** **Mock / Illustrative.** 100 hardcoded synthetic numbers created for ESP32 bench testing without live sensors.
* **irmware/sketch/tinyml_model.h:** **Derived artifact.** C-code decision tree compiled from synthetic training data.
* **Real human/animal measurements present in repository:** **Zero (0).**

---

## 6. Retention & Prioritized Action Plan

### Retain
* The modular sensor noise emulation logic (sensors/noise.py), which realistically reflects physical transducer limitations (DS18B20 quantization, FSR drift/spikes/hysteresis).
* The TinyML export pipeline (m2cgen to C header) for final microcontroller deployment.
* The trial-level splitting logic from iv_infil_v3.ipynb.

### Fix Immediately
1. **Fix 	rain_and_export.py evaluation:** Implement proper trial-level GroupKFold or train/test splitting before calculating metrics.
2. **Eliminate circular tuning:** The synthetic generator parameters (, K_S$) were tuned specifically to ensure classification algorithms succeed. Evaluating an algorithm on data generated by hand-tuned rules to ensure separability is circular.
3. **Ground model in real physiology:** Real skin temperature fluctuations are heavily confounded by systemic autonomic nervous activity (vasoconstriction from stress, cold, emotion), motion artifacts, and core temperature shifts.
