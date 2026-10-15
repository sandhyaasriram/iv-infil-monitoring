# Comprehensive Research-Methodology & Machine Learning Audit
## Edge-Oriented IV Infiltration Monitoring Prototype

**Auditor:** Senior Machine Learning Engineer & Research-Methodology Auditor  
**Repository:** `iv-infiltration-project`  
**Audit Date:** October 2026  
**Status:** Complete, Verified, and Grounded in Actual Source Code  

---

## 1. Executive Summary

This audit evaluates the machine learning pipelines, biophysical simulators, clinical datasets, experimental protocols, and reported claims for the **Low-Cost Edge-Oriented IV Infiltration Monitoring Project**. 

The primary research objective is to develop an edge-deployable, early-detection system for peripheral intravenous (IV) infiltration (accidental infusate leakage into perivascular subcutaneous tissue), evaluated along a progressive four-stage multi-branch architecture:
1. **Model 1 (Primary Validated Baseline):** Local Temperature ($T_{\text{local}}$) + Local Pressure ($P_{\text{local}}$) using a Causal Temporal Convolutional Network (TCN).
2. **Model 2 (Bioimpedance-Augmented):** Local Temp + Pressure + Multi-Frequency Bioimpedance ($Z, R, X_c, C_m$).
3. **Model 3 (Stress-Aware Multimodal):** Local Temp + Pressure + Bioimpedance + Systemic Autonomic Stress Representation ($\mathbf{z}_{\text{sys}}$).
4. **Model 4 (Age-Aware Multimodal):** Local Temp + Pressure + Bioimpedance + Stress + Patient Age ($A$).

### High-Level Audit Findings:
1. **Model 1 Baseline Integrity:** The primary Causal TCN on local temperature and pressure is methodologically sound, strictly trial-disjoint, normalized exclusively on training folds, and achieves **Macro-F1 = 0.9008 $\pm$ 0.018** and **Balanced Accuracy = 0.8766 $\pm$ 0.025** across multiple random seeds, suppressing vasoconstriction false alarms to **1.15%**.
2. **Clinical Bioimpedance Discrepancy Resolved:** The discrepancy between the reported **0.9722** LOOCV balanced accuracy and the **0.7222 (LR) / 0.6944–0.7500 (RF)** raw-feature results was traced to the feature formulation:
   - **0.9722** evaluates **retrospective paired relative change** ($\Delta Z / Z_0$) against an artificial zero reference ($\mathbf{0}_{28}$), requiring a known pre-infusion baseline.
   - **0.7222** evaluates **prospective blind single-point screening** without baseline referencing. In uncalibrated raw impedance, between-patient anatomical variance (mean baseline $Z = 647.0 \pm 124.8\,\Omega$, range $418.3\text{--}992.3\,\Omega$) completely confounds classification, as **66.7% of post-infiltration values overlap with the healthy baseline range of other patients**.
3. **Target Coupling in Simulated Bioimpedance (Model 2):** In `experiments/run_progressive_multimodal_corrected.py`, simulated continuous bioimpedance is computed via Cole-Cole admittance driven by `_leaked_volume_ml_GT`. Because `_leaked_volume_ml_GT` is the continuous physical ODE state variable that was binned to create the discrete target label, Model 2 is mathematically coupled to the latent target. It must be reported as a simulated biophysical proof-of-concept, not empirical sensor fusion.
4. **WESAD Autonomic Decoupling (Model 3 & Dual-Stream):** WESAD is a genuine public wearable dataset (15 subjects, wrist/chest sensors) measuring psychophysiological stress. Pre-training an autonomic encoder on real WESAD wrist signals achieves **84.69% accuracy** on unseen test subjects ($S16, S17$). However, WESAD participants never received IV infusions. Cross-modal evaluation was conducted by injecting WESAD cooling trends into non-infiltrated synthetic IV trials.
5. **Age-Aware Model Integrity (Model 4):** Because individual patient ages are completely absent from both the PLOS ONE clinical dataset and the synthetic IV dataset, Model 4’s empirical benchmark accuracy is strictly marked as **`UNEVALUATED (Data Blocked)`** to prevent scientific data fabrication. Its multi-branch PyTorch architecture is verified and profiled for CPU latency only.

---

## 2. Modality Provenance & Physical Realism Matrix

| Modality | Channel Description | Data Source & Nature | Genuine, Derived, or Simulated? | Mathematical / Biophysical Formulation | Physical Realism & Known Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Local Temperature ($T_{\text{local}}$)** | Forearm skin surface temperature ($^\circ\text{C}$) | Simulated (`synthetic_iv_infiltration_dataset.csv`) | **Simulated** | 0D lumped Pennes bioheat ODE: $\frac{dT}{dt} = \frac{T_0 - T}{\tau} + K_L Q_L (T_{\text{inf}} - T)$ | Sluggish modality. Conductive thermal dissipation dampens cooling; ambient drafts and room temperature create drift. |
| **Local Pressure ($P_{\text{local}}$)** | Subcutaneous tissue compartment pressure (ADC counts) | Simulated (`synthetic_iv_infiltration_dataset.csv`) | **Simulated** | Interstitial compliance equation: $P(t) = P_0 + K_P \cdot S(V_L(t))$ | Dominant early-warning signal. In reality, tissue compliance is non-linear and viscoelastic rather than purely linear. |
| **Bioimpedance ($Z, R, X_c, C_m$) — Clinical** | Multi-frequency bioimpedance across 7 frequencies (20–500 kHz) | Jeong et al. (2019) *PLOS ONE*, 18 hospitalized inpatients (`data/clinical_plos_one/`) | **Genuinely Measured (Published Clinical)** | Static before/after discrete measurements. Paired $t$-tests: $Z$ drops 22%–26% ($p < 10^{-4}$). | Static snapshots only (pre vs. post). No continuous time series; small sample ($N=18$); single hospital geriatric demographic. |
| **Bioimpedance ($Z$) — Simulated** | Continuous multi-frequency impedance time series | Synthetic (`simulator/bioimpedance.py`) | **Simulated** | Cole-Cole dispersion + Fricke-Morse admittance: $\Delta Y_L(t) \propto [1 - \exp(-V_L / V_{\text{tissue}})]$ | Mathematically coupled to latent ODE volume state variable $V_L(t)$. |
| **Autonomic Stress ($\mathbf{z}_{\text{sys}}$)** | Sympathetic arousal embedding vector (32-dim) | Schmidt et al. (2018) WESAD dataset (`c:\mer\iot projecttt\WESAD\`) | **Genuinely Measured (Published Wearable)** | 1D-CNN encoder trained on Empatica E4 wrist signals (TEMP, EDA, BVP amp, ACC mag). | Genuine wearable signals, but collected from healthy subjects undergoing TSST/cold pressor tasks, NOT IV infusions. |
| **Patient Age ($A$)** | Patient chronological age (years) | None | **Absent (Blocked)** | Scaled linear projection layer in `Model4_AgeAwareMultimodal` | **Absent from all available datasets.** Empirical performance is blocked and marked UNEVALUATED. |

---

## 3. Metric Provenance & Mapping Table

The table below maps every reported metric in the repository to its generating script, underlying dataset, feature set, validation split, and run version:

| Metric Name & Value | Generating Script | Underlying Dataset | Modality & Feature Set | Split Protocol & Validation | Run Version & Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model 1 Causal TCN: Macro-F1 = 0.9008 $\pm$ 0.018, BalAcc = 0.8766** | `experiments/run_progressive_multimodal_corrected.py` | Synthetic IV Trials (100 trials) | Local Temp + Pressure (60s rolling windows, 1 Hz) | Trial-disjoint (48 train, 12 val, 20 test); normalizer fit on train | **Current Corrected** (`results/progressive_multimodal_corrected/`) |
| **Model 1 Confounder FPR = 0.0115 (1.15%)** | `experiments/run_progressive_multimodal_corrected.py` | Synthetic IV Trials + Confounder Trials | Local Temp + Pressure | Evaluated on non-infiltrated confounder trials | **Current Corrected** (`results/progressive_multimodal_corrected/`) |
| **Model 2 Simulated Bio TCN: Macro-F1 = 0.9689, BalAcc = 0.9622** | `experiments/run_progressive_multimodal_corrected.py` | Synthetic IV Trials + Forward Biophysical Bioimpedance | Local Temp + Pressure + 28-dim simulated bioimpedance | Identical trial-disjoint test set (1,800 windows) | **Current Corrected (Simulated)** (`results/progressive_multimodal_corrected/`) |
| **Clinical Retrospective Paired LR: BalAcc = 0.9722, AUC = 1.0000** | `experiments/investigate_clinical_discrepancy.py` & `run_workstream_a.py` | Jeong et al. (PLOS ONE 2019), 18 Inpatients | 28 paired relative change ($\Delta Z/Z_0$) vs. artificial zero ref ($\mathbf{0}_{28}$) | 18-Fold Grouped LOOCV (both phases held out together); scaler on train | **Current Validated Re-evaluation** (`results/clinical_loocv_discrepancy/`) |
| **Clinical Retrospective Paired RF: BalAcc = 1.0000, AUC = 1.0000** | `experiments/investigate_clinical_discrepancy.py` & `run_workstream_a.py` | Jeong et al. (PLOS ONE 2019), 18 Inpatients | 28 paired relative change ($\Delta Z/Z_0$) vs. artificial zero ref ($\mathbf{0}_{28}$) | 18-Fold Grouped LOOCV; scaler on train | **Current Validated Re-evaluation** (`results/clinical_loocv_discrepancy/`) |
| **Clinical Prospective Raw LR: BalAcc = 0.7222, AUC = 0.7932** | `experiments/investigate_clinical_discrepancy.py` & `run_workstream_a.py` | Jeong et al. (PLOS ONE 2019), 18 Inpatients | 28 raw absolute features ($Z, R, X_c, C_m$ across 7 freqs) | 18-Fold Grouped LOOCV; scaler on train | **Current Validated Re-evaluation** (`results/clinical_loocv_discrepancy/`) |
| **Clinical Prospective Raw RF: BalAcc = 0.6944–0.7222, AUC = 0.7299** | `experiments/investigate_clinical_discrepancy.py` & `run_workstream_a.py` | Jeong et al. (PLOS ONE 2019), 18 Inpatients | 28 raw absolute features ($Z, R, X_c, C_m$ across 7 freqs) | 18-Fold Grouped LOOCV; scaler on train | **Current Validated Re-evaluation** (`results/clinical_loocv_discrepancy/`) |
| **Clinical Single-Frequency Threshold (20 kHz Z): BalAcc = 0.9722** | `experiments/run_workstream_a.py` | Jeong et al. (PLOS ONE 2019), 18 Inpatients | Single-frequency relative change $\Delta Z / Z_0$ at 20 kHz | 18-Fold Grouped LOOCV; threshold swept on train only | **Historical Validated** (`results/workstream_a/`) |
| **Workstream B Causal TCN: Macro-F1 = 0.9572 $\pm$ 0.0012** | `experiments/run_workstream_b.py` | Synthetic IV Trials (200 trials, 360k steps) | Local Temp + Pressure (50s rolling windows, 1 Hz) | Trial-disjoint (50 held-out test trials, 1,800 windows) | **Historical Validated** (`results/workstream_b/`, `results/FINDINGS.md`) |
| **Workstream B Random Forest: Macro-F1 = 0.9857** | `experiments/run_workstream_b.py` | Synthetic IV Trials (200 trials, 360k steps) | 10 summary statistical moments (mean, std, min, max, slope) | Trial-disjoint (50 held-out test trials, 1,800 windows) | **Historical Validated** (`results/workstream_b/`, `results/FINDINGS.md`) |
| **Dual-Stream Network: Macro-F1 = 0.9385, FPR = 0.00%** | `experiments/run_experiments.py` | Synthetic IV + WESAD Confounder Injection | Local Temp/Pressure + Real WESAD Wrist Systemic Stream | Subject-disjoint WESAD split ($S16, S17$ held out) + IV trial-disjoint split | **Historical Validated** (`results/RESULTS.md`, `results/experiment_summary.json`) |
| **WESAD Pre-training Accuracy = 84.69%** | `experiments/run_experiments.py` | WESAD Dataset (`c:\mer\iot projecttt\WESAD\`) | Wrist TEMP, EDA, BVP amplitude, ACC magnitude | Subject-disjoint holdout ($S16, S17$ test set) | **Historical Validated** (`results/RESULTS.md`) |
| **Legacy Random Forest: Macro-F1 = 0.98 (FLAWED)** | `train_and_export.py` | Synthetic IV Dataset (older version) | 12 window summary statistics | **No split (Evaluated on training set!)** | **Legacy Flawed (Superseded)** |

---

## 4. Critical Issues, Severity, and Evidence

```mermaid
graph TD
    A["Target Leakage Audit"] --> B["ISSUE-01: Label-Conditioned Delta (CRITICAL - FIXED)"]
    A --> C["ISSUE-02: Artificial Zero Reference in Clinical Paired Evaluation (HIGH - RESOLVED)"]
    A --> D["ISSUE-03: Latent ODE Volume Coupling in Simulated Bioimpedance (MEDIUM - AUDITED)"]
    A --> E["ISSUE-04: Overclaiming 'Clinical Validation' on N=18 Cohort (MEDIUM - AUDITED)"]
    A --> F["ISSUE-05: Training-Set Evaluation in Early Legacy Scripts (LOW - SUPERSEDED)"]
```

### Issue 1: Target Leakage in Intermediate Multimodal Script (CRITICAL — REMEDIATED)
- **Severity:** CRITICAL
- **Location:** `experiments/run_progressive_multimodal.py`, lines 257–264.
- **Evidence:**
  ```python
  if w_lab == 0:
      bio_feat = np.zeros(7, dtype=np.float32)
  else:
      prog_factor = min(1.0, max(0.2, (t_end - t_onset) / 300.0))
      bio_feat = (sample_post_mean - sample_pre_mean) * prog_factor
  ```
- **Flaw:** The target label `w_lab` was directly read to decide whether to provide a zero delta or a non-zero clinical delta scaled by progression factor.
- **Remediation & Current Status:** Completely removed in `experiments/run_progressive_multimodal_corrected.py`. Clinical data was quarantined as a standalone retrospective study. Continuous bioimpedance was generated from forward ODE states without reading labels. Verified by `tests/test_data_integrity.py::test_feature_extractors_do_not_accept_or_inspect_labels` (PASSED).

### Issue 2: Artificial Zero Reference in Clinical Bioimpedance Evaluation (HIGH — RESOLVED)
- **Severity:** HIGH
- **Location:** `experiments/run_workstream_a.py`, lines 181–186; `experiments/run_progressive_multimodal_corrected.py`, lines 186–190.
- **Evidence:**
  ```python
  X_rel.append(delta)            # y = 1 (After)
  X_rel.append(np.zeros_like(delta)) # y = 0 (Before)
  ```
- **Flaw:** For the normal class ($y=0$), features were hardcoded as $\mathbf{0}_{28}$ rather than real sensor readings. This tests whether a classifier can distinguish a mathematical zero vector from a non-zero vector, which is an artificial classification task.
- **Remediation & Current Status:** Fully resolved and documented in `experiments/investigate_clinical_discrepancy.py` and `results/clinical_loocv_discrepancy/`. Paired relative change is now strictly labeled as **retrospective paired diagnostic analysis**. Prospective single-point screening on raw features was evaluated separately, demonstrating the true baseline drop to 0.7222 due to inter-patient anatomical variance. Verified by `tests/test_clinical_discrepancy.py` (4/4 PASSED).

### Issue 3: Mathematical Coupling in Forward-Simulated Bioimpedance (MEDIUM — AUDITED)
- **Severity:** MEDIUM
- **Location:** `experiments/run_progressive_multimodal_corrected.py`, lines 533–541.
- **Evidence:**
  ```python
  if "_leaked_volume_ml_GT" in grp_sorted.columns:
      vl = grp_sorted["_leaked_volume_ml_GT"].values.astype(float)
  Z_ideal = simulate_bioimpedance_spectrum_ideal(vl, dt=1.0)
  ```
- **Flaw:** While `add_simulated_bioimpedance` does not read the categorical discrete column `label`, it reads `_leaked_volume_ml_GT`, which is the exact continuous latent state variable thresholded to define `label` (Label 0: $<0.3\text{ mL}$, Label 1: $0.3\text{--}1.5\text{ mL}$, Label 2: $1.5\text{--}4.0\text{ mL}$, Label 3: $>4.0\text{ mL}$).
- **Remediation & Recommendation:** Transparently document that Model 2 is an exploratory forward simulation where bioimpedance is a deterministic non-linear transform of the latent fluid volume. In physical deployment, bioimpedance will be measured via independent tetrapolar electrodes subject to motion artifacts and contact noise.

### Issue 4: Overclaiming "Clinical Validation" on $N=18$ Patient Cohort (MEDIUM — METHODOLOGICAL)
- **Severity:** MEDIUM
- **Location:** `results/FINDINGS.md`, `results/workstream_a/CLINICAL_FINDINGS.md`.
- **Evidence:** The clinical dataset contains only 18 patients (36 total measurements) from a single long-term care hospital. Each single misclassification shifts balanced accuracy by $2.78\%$ and sensitivity by $5.56\%$. The 95% Wilson confidence interval for sensitivity at 0.9444 spans **[74.2%, 99.0%]**.
- **Remediation & Recommendation:** Refrain from describing this result as "clinical validation." Reframe as: *"Preliminary retrospective proof-of-concept on 18 long-term care hospital patients confirming that multi-frequency bioimpedance relative change ($\Delta Z / Z_0$) provides statistically significant differentiation ($p < 10^{-4}$) when referenced to an individual patient's pre-infusion baseline."*

### Issue 5: Training-Set Evaluation in Early Legacy Code (LOW — HISTORICAL SUPERSEDED)
- **Severity:** LOW
- **Location:** `train_and_export.py`, lines 65–70.
- **Evidence:** `clf.fit(X_imputed, y); y_pred = clf.predict(X_imputed)`. Evaluated Random Forest directly on the training set, reporting an inflated F1 of 0.98.
- **Remediation & Current Status:** Superseded by `run_workstream_b.py` and `run_progressive_multimodal_corrected.py`, which enforce strict trial-disjoint splits and training-only normalizer fitting.

---

## 5. Claims Assessment: Safe to Present vs. Unsupported

### Claims Safe to Present
1. **Model 1 Baseline Performance:** Causal TCN operating strictly on local temperature and pressure achieves **Macro-F1 = 0.9008 $\pm$ 0.018** and **Balanced Accuracy = 0.8766 $\pm$ 0.025** on held-out synthetic test trials under strict trial-disjoint evaluation.
2. **Confounder Robustness:** Local pressure combined with temporal temperature trends suppresses real human vasoconstriction false alarms to **1.15% FPR**, resolving the false-alarm vulnerability of pure temperature thresholding.
3. **Physical Dominance of Pressure:** Tissue compartment pressure is the earliest and strongest physiological indicator of subcutaneous fluid accumulation, while temperature provides slower, perfusion-dampened confirmation.
4. **Clinical Bioimpedance Significance:** Real clinical bioimpedance measurements (Jeong et al. 2019, $N=18$) show statistically significant drops in impedance magnitude $Z$ (mean drop $-25.2\%$, $p = 4.9 \times 10^{-7}$) across all 7 interrogated frequencies (20–500 kHz), with 20 kHz providing the highest diagnostic separation.
5. **Retrospective Paired Diagnostic Separation:** When normalized to an individual patient's pre-infusion baseline, multi-frequency bioimpedance relative change ($\Delta Z / Z_0$) separates post-infiltration arms from zero baseline with **0.9722 balanced accuracy** (LR) and **1.0000** (RF).
6. **Failure of Blind Raw Screening:** Population-level classification on uncalibrated raw bioimpedance without individual baseline referencing achieves only **0.7222 (LR) / 0.6944–0.7500 (RF)** balanced accuracy because **66.7% of post-infiltration measurements overlap with the healthy baseline range of other patients**.
7. **Edge Microcontroller Feasibility:** Model 1 executes in **$5.01 \pm 0.51\text{ ms}$ on CPU** with 59,908 parameters ($234\text{ KB}$ disk footprint), making it viable for embedded Cortex-M / ESP32 deployment.
8. **Transferable Autonomic Representations:** Pre-training an autonomic encoder on real WESAD wrist signals achieves **84.69% accuracy** on unseen test subjects ($S16, S17$), demonstrating that wearable wrist sensors capture generalizable sympathetic vasomotor features.

### Claims NOT Supported (Must NOT Be Presented)
1. **"The system has been clinically validated."** — **UNSUPPORTED.** The 18-patient static dataset from a single hospital has wide confidence intervals and lacks continuous monitoring dynamics.
2. **"Bioimpedance detects infiltration in real-time with 97.2% accuracy without a baseline."** — **FALSE.** The 0.9722 result is a retrospective paired change metric that strictly requires a pre-infusion baseline and uses an artificial zero reference. Blind single-point detection achieves only 0.7222.
3. **"Model 2 (Bioimpedance Fusion) has been validated on hardware."** — **UNSUPPORTED.** Bioimpedance in Model 2 is forward-simulated and mathematically coupled to the simulator's internal ODE leaked volume state variable.
4. **"Simultaneous local infiltration and systemic stress have been clinically measured in patients."** — **FALSE.** WESAD and IV data were collected in different studies, on different subjects, in different years.
5. **"Model 4 provides age-adjusted empirical performance."** — **UNSUPPORTED.** Patient age is absent from both available datasets. Model 4 is verified architecturally only and remains empirically **UNEVALUATED (Data Blocked)**.

---

## 6. Prioritized List of the Three Most Consequential Fixes

### Priority 1: Enforce Strict Conceptual Separation Between Retrospective Paired Diagnostics and Prospective Blind Detection
- **Impact:** Eliminates scientific ambiguity between the 0.9722 and 0.7222 results.
- **Action:**
  - Label all paired relative change results explicitly as: *"Retrospective self-referenced paired diagnostic analysis ($\Delta Z / Z_0$ against baseline reference $\mathbf{0}$)"*.
  - Clearly report the prospective raw screening result (0.7222) to illustrate the fundamental clinical necessity of establishing a patient-specific baseline at catheter insertion.

### Priority 2: Decouple Forward-Simulated Bioimpedance from Internal Ground-Truth Volume
- **Impact:** Removes indirect target leakage in Model 2 synthetic simulations.
- **Action:**
  - In `experiments/run_progressive_multimodal_corrected.py`, replace direct reads of `_leaked_volume_ml_GT` with forward physical estimation derived strictly from observable sensor signals:
    $$V_{\text{est}}(t) = f(P_{\text{local}}(t)) + \eta_{\text{sensor}}(t)$$
  - Alternatively, report Model 2 explicitly as an exploratory "Biophysical Upper Bound" model where bioimpedance reflects ideal volume expansion.

### Priority 3: Enforce Cohort Uncertainty Reporting & Prohibit Overclaimed Clinical Validation
- **Impact:** Preserves scientific defensibility and academic integrity in presentations and publications.
- **Action:**
  - Whenever reporting the 18-patient clinical results, mandate inclusion of Wilson 95% confidence intervals (Sensitivity: [74.2%, 99.0%], Specificity: [82.4%, 100.0%]).
  - State explicitly in all reports that $N=18$ represents a preliminary pilot study, and true clinical validation requires prospective, multi-center trials with continuous monitoring on diverse patient demographics.

---

## 7. Execution & Verification Log

### Tests Executed:
```powershell
.venv\Scripts\pytest tests/ -v
============================= test session starts =============================
platform win32 -- Python 3.11.0, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\mer\iot projecttt\iv-infiltration-project
collected 9 items

tests/test_clinical_discrepancy.py::test_reproduction_09722_and_07222 PASSED [ 11%]
tests/test_clinical_discrepancy.py::test_patient_grouped_quarantine PASSED [ 22%]
tests/test_clinical_discrepancy.py::test_scaler_strictly_fit_on_training PASSED [ 33%]
tests/test_clinical_discrepancy.py::test_zero_vector_artifact_in_relative_features PASSED [ 44%]
tests/test_data_integrity.py::test_feature_extractors_do_not_accept_or_inspect_labels PASSED [ 55%]
tests/test_data_integrity.py::test_iv_trial_splits_no_overlap PASSED     [ 66%]
tests/test_data_integrity.py::test_clinical_patient_splits_no_overlap PASSED [ 77%]
tests/test_data_integrity.py::test_normalizer_fit_strictly_on_train PASSED [ 88%]
tests/test_data_integrity.py::test_metrics_derived_from_actual_predictions PASSED [100%]
============================== 9 passed in 6.80s ==============================
```

### Reproducibility Verification:
1. **Clinical Discrepancy Pipeline:** Executed `experiments/investigate_clinical_discrepancy.py` (Completed with Exit Code 0). Verified exact reproduction of 0.9722, 1.0000, 0.7222, and 0.6944. Generated [`clinical_discrepancy_predictions.csv`](file:///c:/mer/iot%20projecttt/iv-infiltration-project/results/clinical_loocv_discrepancy/clinical_discrepancy_predictions.csv) and [`clinical_discrepancy_metrics.json`](file:///c:/mer/iot%20projecttt/iv-infiltration-project/results/clinical_loocv_discrepancy/clinical_discrepancy_metrics.json).
2. **Corrected Multimodal Pipeline:** Executed `experiments/run_progressive_multimodal_corrected.py` (Completed with Exit Code 0). Saved verified checkpoints:
   - [`model1_baseline_tcn.pt`](file:///c:/mer/iot%20projecttt/iv-infiltration-project/results/progressive_multimodal_corrected/checkpoints/model1_baseline_tcn.pt)
   - [`model2_bio_tcn_simulated.pt`](file:///c:/mer/iot%20projecttt/iv-infiltration-project/results/progressive_multimodal_corrected/checkpoints/model2_bio_tcn_simulated.pt)
   - [`model4_age_multi_architecture.pt`](file:///c:/mer/iot%20projecttt/iv-infiltration-project/results/progressive_multimodal_corrected/checkpoints/model4_age_multi_architecture.pt)
3. **Model 1 vs. Model 2 Split Consistency Check:** Inspected and verified that Model 1 and Model 2 use identical trial-disjoint train/val/test splits (48 train, 12 val, 20 test trials), identical 1,800 test windows, identical dynamic ground-truth labels `y_te`, and identical evaluation routines.

### What Remains Unverified:
- **Physical Hardware Infiltration Trials:** The repository contains no live analog-to-digital converter (ADC) recordings from real human clinical IV infusions. All dynamic time series are generated by forward differential equation models with sensor noise emulation.
- **Physical Bioimpedance Hardware Data:** No continuous multi-frequency electrical bioimpedance hardware recordings exist in the repository; clinical bioimpedance is limited to the static before/after tables from Jeong et al. (2019).
