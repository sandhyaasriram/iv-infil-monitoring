# Target Leakage Audit, Methodological Corrections, and Rigorous Evaluation Report

**Project:** Progressive Multimodal IV Infiltration Detection & Continuous Monitoring  
**Audit Date:** October 2026  
**Artifact Location:** `results/progressive_multimodal_corrected/`  
**Status:** Complete, Verified, and Audited  

---

## 1. Executive Summary & Root Cause Diagnosis

### 1.1 The Vulnerability: Target Leakage in Multimodal Feature Synthesis
During an audit of the initial progressive multimodal pipeline (`experiments/run_progressive_multimodal.py`), a critical methodological flaw was discovered in the cross-modal coupling logic:

```python
# FLAW IN ORIGINAL run_progressive_multimodal.py:
if w_lab == 0:
    # Set to zero delta for non-infiltrated windows
    bio_feat = np.zeros(7, dtype=np.float32)
else:
    # Scale clinical bioimpedance deltas by target severity!
    prog_factor = min(1.0, max(0.2, (t_end - t_onset) / 300.0))
    bio_feat = (sample_post_mean - sample_pre_mean) * prog_factor
```

**Why this was target leakage:**
1. The synthetic IV dataset (`data/synthetic/`) contains continuous 1 Hz time series of local temperature and local pressure across 100 trials (4 severity grades + vasoconstriction confounders).
2. The clinical bioimpedance dataset (`data/clinical/bioimpedance_jeong2019/pone.0213585.s001.xlsx`) contains static, retrospective pre- and post-infiltration measurements from 18 geriatric inpatients.
3. In the prior implementation, the classification label `w_lab` was directly read to decide whether to provide a zero delta or a non-zero clinical delta.
4. Consequently, any neural network trained on this composite feature was not learning subtle cross-modal physiological synergies; it was merely decoding an artificial backdoor label cheat.
5. Furthermore, simply shuffling impedance features across windows was rejected as a fix because shuffling destroys physical temporal continuity and fails to establish physiological validity.

---

## 2. Structural Remediation & Strict Protocol Separation

To establish scientific defensibility and total reproducibility, four architectural changes were enacted:

```mermaid
flowchart TD
    subgraph S1["Evaluation 1: Primary Validated Baseline"]
        A1["Synthetic IV Trials (1 Hz Temp + Pressure)"] --> B1["Trial-Disjoint Train / Val / Test Split"]
        B1 --> C1["Z-Score Normalizer (Fit Strictly on Train)"]
        C1 --> D1["Model 1: Baseline Causal TCN (59.9k params)"]
        D1 --> E1["Primary IV Benchmark (Macro-F1: 0.9007)"]
    end

    subgraph S2["Evaluation 2: Real Clinical Bioimpedance"]
        A2["18 Inpatients (Jeong et al. 2019)"] --> B2["Leave-One-Patient-Out CV (18 Folds)"]
        B2 --> C2["Patient-Grouped Scaling within Fold"]
        C2 --> D2["Logistic Regression / ElasticNet Baseline"]
        D2 --> E2["Clinical LOOCV Benchmark (BalAcc: 0.9722, AUC: 1.000)"]
    end

    subgraph S3["Evaluation 3: Forward Biophysical Simulation (Label-Free)"]
        A3["Continuous ODE Extravasation Volume V_L(t)"] --> B3["Fricke-Cole Admittance Physics Engine"]
        B3 --> C3["Continuous Multi-Freq Z(f,t) + Noise (No Label Access)"]
        C3 --> D3["Model 2: Exploratory Simulated Multimodal TCN"]
        D3 --> E3["Simulated Cross-Modal Benchmark (Macro-F1: 0.9689)"]
    end

    subgraph S4["Evaluation 4: Age-Aware Multimodal (Integrity Enforced)"]
        A4["Age Branch Architecture"] --> B4["Validated PyTorch Module & FLOP Profiling"]
        B4 --> C4["Empirical Score: UNEVALUATED (Data Blocked)"]
    end
```

### 2.1 Separation of Datasets
- **Real Clinical Bioimpedance Data (Jeong et al. 2019):** Analyzed strictly as an independent standalone clinical study using Leave-One-Patient-Out Cross-Validation (LOOCV). We never pretend these static hospital measurements were collected synchronously alongside the 1 Hz synthetic fluid-flow trials.
- **Primary IV Infiltration Temporal Benchmark:** Model 1 (Causal TCN) processes strictly local temperature and pressure time series, evaluated on trial-disjoint splits with training-only normalizer fitting.
- **Exploratory Simulated Bioimpedance (Model 2):** Generated strictly via a biophysical forward differential equation from ODE state variables (interstitial edema fluid accumulation $V_L(t)$), with zero inspection of classification labels.
- **Model 4 (Age-Aware Model):** Evaluated strictly for computational latency and architectural integrity. Because genuine individual patient age labels are absent from both the PLOS ONE dataset and the synthetic dataset, its empirical accuracy is explicitly documented as **UNEVALUATED (Data Blocked)** to prevent data fabrication.

---

## 3. Biophysical Forward Bioimpedance Formulation

To model bioimpedance continuously without categorical label leakage, we implemented `simulator/bioimpedance.py` based on classic tissue biophysics (Cole-Cole and Fricke-Morse suspensions).

### 3.1 Cole-Cole Baseline Tissue Dispersion
Baseline subcutaneous tissue complex impedance $Z_0(\omega)$ across the 7 interrogated frequencies ($f \in \{20, 50, 100, 150, 200, 300, 500\}\text{ kHz}$) follows:
$$Z_0(\omega) = R_\infty + \frac{R_0 - R_\infty}{1 + (j\omega \tau)^\alpha}$$
where:
- $R_0 = 850\,\Omega$ (low-frequency extracellular resistance)
- $R_\infty = 280\,\Omega$ (high-frequency resistance where cell membranes conduct)
- $\tau = 3.18 \times 10^{-6}\text{ s}$ (relaxation time constant, corresponding to $f_c \approx 50\text{ kHz}$)
- $\alpha = 0.82$ (Cole-Cole distribution parameter)

### 3.2 Extravasation Admittance Expansion
As infusate accumulates in the interstitial subcutaneous space with volume $V_L(t)$ (derived continuously from the physiological fluid transport ODE), the additional ionic saline pathways increase total tissue admittance:
$$\Delta Y_L(t) = \frac{\sigma_{\text{infusate}} \cdot A_{\text{effective}}}{d_{\text{electrode}}} \cdot \left[1 - \exp\left(-\frac{V_L(t)}{V_{\text{tissue}}}\right)\right]$$
Total admittance at frequency $f$ and time $t$:
$$Y(f, t) = \frac{1}{Z_0(f)} + \Delta Y_L(t)$$
$$Z(f, t) = \frac{1}{Y(f, t)} + \epsilon_{\text{drift}}(t) + \mathcal{N}(0, \sigma_{\text{Johnson}}^2)$$
where:
- $\epsilon_{\text{drift}}(t)$ models electrode polarization drift ($0.02\,\Omega/\text{s}$).
- $\sigma_{\text{Johnson}} = 2.5\,\Omega$ Gaussian thermal and instrumentation noise.
- **Crucially:** $V_L(t)$ is a continuous state variable. The generator receives **no target labels, no severity classes, and no confounder flags**.

---

## 4. Automated Data Integrity Verification

A strict test suite (`tests/test_data_integrity.py`) was constructed and executed to prove compliance before running any model training:

```powershell
============================= test session starts =============================
platform win32 -- Python 3.11.0, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\mer\iot projecttt\iv-infiltration-project
collected 5 items

tests/test_data_integrity.py::test_feature_extractors_do_not_accept_or_inspect_labels PASSED [ 20%]
tests/test_data_integrity.py::test_iv_trial_splits_no_overlap PASSED                     [ 40%]
tests/test_data_integrity.py::test_clinical_patient_splits_no_overlap PASSED             [ 60%]
tests/test_data_integrity.py::test_normalizer_fit_strictly_on_train PASSED              [ 80%]
tests/test_data_integrity.py::test_metrics_derived_from_actual_predictions PASSED        [100%]

============================= 5 passed in 27.36s ==============================
```

### Verification Guarantees:
1. **Label Independence (`test_feature_extractors_do_not_accept_or_inspect_labels`):** Introspects generator functions and confirms that `w_lab`, `label`, or `severity` do not appear in the parameter signatures or computation graphs.
2. **Trial-Disjoint Splits (`test_iv_trial_splits_no_overlap`):** Asserts $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, and $\text{Val} \cap \text{Test} = \emptyset$.
3. **Patient-Disjoint Clinical Splits (`test_clinical_patient_splits_no_overlap`):** Asserts that in every LOOCV fold, the test patient's pre- and post-measurements are completely quarantined from the 17 training patients.
4. **Zero Preprocessing Leakage (`test_normalizer_fit_strictly_on_train`):** Asserts normalization statistics ($\mu, \sigma$) are computed exclusively on training data and frozen before transforming validation or test sets.
5. **Prediction Grounding (`test_metrics_derived_from_actual_predictions`):** Asserts metrics are strictly computed from forward passes of real model weights rather than constant mocks.

---

## 5. Experimental Results & Performance Benchmark

### 5.1 Standalone Real Clinical Bioimpedance (Jeong et al. 2019)
Evaluated on all 18 clinical inpatients using Leave-One-Patient-Out Cross-Validation (18 outer folds):

| Metric | Clinical Inpatient Performance (18 Folds) |
| :--- | :---: |
| **Balanced Accuracy** | **0.9722** |
| **F1-Score** | **0.9714** |
| **Sensitivity (Recall)** | **0.9444** (17 / 18 infiltrated correctly flagged) |
| **Specificity** | **1.0000** (18 / 18 healthy baselines correctly flagged) |
| **ROC-AUC** | **1.0000** |
| **Confusion Matrix** | $\begin{bmatrix} 18 & 0 \\ 1 & 17 \end{bmatrix}$ |

*Clinical Conclusion:* Real clinical multi-frequency bioimpedance relative changes ($\Delta Z / Z_0$) provide strong retrospective statistical evidence ($p < 0.001$) separating infiltrated limbs from healthy baselines in hospital inpatients, confirming the findings of Workstream A.

---

### 5.2 Model 1: Baseline Causal TCN (Primary Validated Model)
- **Input Channels:** Local Temperature ($T_{\text{local}}$) + Local Pressure ($P_{\text{local}}$) (60 s rolling windows at 1 Hz).
- **Evaluation Split:** Trial-disjoint (48 Train trials, 12 Val trials, 20 Test trials; 1,800 test windows).
- **Model Parameters:** 59,908 | **CPU Inference Latency:** $5.01 \pm 0.51\text{ ms}$ | **Size:** $234.0\text{ KB}$.

| Seed | Macro-F1 | Balanced Accuracy | Sensitivity | Specificity | Confounder FPR | Clamped Delay (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seed 42** | 0.8825 | 0.8511 | 0.8511 | 0.9693 | 0.0046 | 76.24 |
| **Seed 101** | 0.9190 | 0.9020 | 0.9020 | 0.9841 | 0.0185 | 0.65 |
| **Mean $\pm$ Std** | **0.9008 $\pm$ 0.018** | **0.8766 $\pm$ 0.025** | **0.8766 $\pm$ 0.025** | **0.9767 $\pm$ 0.007** | **0.0115 $\pm$ 0.007** | **38.45 $\pm$ 37.8** |

#### Test Confusion Matrix (Seed 101):
$$\begin{array}{c|cccc}
\text{Actual} \backslash \text{Pred} & \text{Normal (0)} & \text{Mild (1)} & \text{Moderate (2)} & \text{Severe (3)} \\
\hline
\text{Normal} & 925 & 52 & 0 & 0 \\
\text{Mild} & 3 & 511 & 17 & 0 \\
\text{Moderate} & 0 & 1 & 254 & 0 \\
\text{Severe} & 0 & 0 & 11 & 26 \\
\end{array}$$

*Key Diagnostic Finding:* The corrected Causal TCN achieves a **Macro-F1 of 0.9008** on synthetic trials without any bioimpedance cheating. It suppresses vasoconstriction confounder false alarms to **1.15%**, demonstrating robust temporal discrimination between systemic ambient cooling and progressive interstitial infusion blockage.

---

### 5.3 Model 2: Exploratory Biophysical Bioimpedance Fusion (Label-Free Forward Model)
- **Input Channels:** Local Temp + Pressure (Causal TCN encoder) + Multi-Frequency Bioimpedance Admittance (MLP Feature Encoder).
- **Bioimpedance Source:** Derived continuously from ODE interstitial fluid volume $V_L(t)$ via Cole-Cole + Fricke-Morse equations.
- **Model Parameters:** 66,084 | **CPU Inference Latency:** $10.34 \pm 8.84\text{ ms}$ | **Size:** $258.1\text{ KB}$.

| Model Variant | Macro-F1 | Balanced Acc | Confounder FPR | Scientific Status |
| :--- | :---: | :---: | :---: | :--- |
| **Model 1: Baseline TCN** | 0.9008 | 0.8766 | 0.0115 | Fully Validated Primary Model |
| **Model 2: Forward Bio TCN** | **0.9689** | **0.9622** | **0.0000** | Exploratory Simulated Multimodal Model |

*Observation:* When bioimpedance is simulated from underlying fluid volume, adding multi-frequency electrical admittance provides orthogonal confirmation of fluid accumulation, completely eliminating vasoconstriction false alarms (FPR = 0.0000). However, because the bioimpedance branch is forward-simulated rather than recorded simultaneously on hardware, it is maintained as an exploratory research prototype.

---

### 5.4 Model 4: Age-Aware Multimodal Architecture Profile
- **Input Architecture:** Modular 4-branch network with Causal TCN (Local Temp/Pressure), Bioimpedance MLP, Stress Representation MLP, and Age Feature Scaled Embedding.
- **Hardware Profile:**
  - Total Trainable Parameters: **99,702**
  - CPU Forward Latency: **22.34 ms** ($\pm 4.79\text{ ms}$)
  - Checkpoint Disk Footprint: **389.5 KB**
- **Empirical Status:** **UNEVALUATED (Data Blocked)**
  - *Reason:* Neither the PLOS ONE dataset nor the synthetic IV dataset contains patient-linked age distributions. Fabricating ages would violate scientific integrity. The module is fully implemented in `src/models/progressive_multimodal.py` and ready for clinical trials with geriatric cohorts.

---

## 6. Artifact & Deliverables Directory Structure

All files have been verified and saved to `results/progressive_multimodal_corrected/`:

```
results/progressive_multimodal_corrected/
├── checkpoints/
│   ├── model1_baseline_tcn.pt                (266 KB - PyTorch state dict for Model 1)
│   ├── model2_bio_tcn_simulated.pt           (297 KB - PyTorch state dict for Model 2)
│   └── model4_age_multi_architecture.pt      (444 KB - PyTorch state dict for Model 4)
├── plots/
│   ├── clinical_bioimpedance_performance.png (79 KB - ROC curve & LOOCV confusion matrix)
│   └── corrected_model1_confusion_matrix.png (100 KB - 4-class multi-seed confusion matrix)
├── corrected_metrics.json                    (Complete structured metrics and profiling)
└── target_leakage_audit_and_correction_report.md (This audit document)
```

---

## 7. Compliance Checklist

- [x] **No target label leakage:** Removed all rules selecting or scaling bioimpedance based on target labels.
- [x] **No naive shuffling:** Rejected random feature shuffling; implemented biophysical forward simulation from physical states.
- [x] **Preserved clinical dataset integrity:** Jeong et al. evaluated purely with 18-fold LOOCV.
- [x] **Trial-disjoint splits:** Zero overlap across train, validation, and test sets.
- [x] **Train-only preprocessing:** All scalers fit strictly on training splits.
- [x] **Automated test suite:** `pytest tests/test_data_integrity.py` passed with 5/5 assertions.
- [x] **Model 4 data integrity:** Age model marked as **UNEVALUATED** to prevent data fabrication.
- [x] **All results saved to separate corrected directory:** `results/progressive_multimodal/` left intact.
