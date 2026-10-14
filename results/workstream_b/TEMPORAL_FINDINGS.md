# Temporal Infiltration-State Estimation — Findings (Workstream B)
## IV Infiltration Monitoring — Temporal Model Comparison

> [!NOTE]
> All results are from **synthetic simulator data** (N=200 trials, 1800 timesteps/trial at 1 Hz).
> The simulator generates temperature (°C) and pressure (ADC counts) traces using a physics-grounded
> ODE model (Pennes bioheat with autonomic vasoconstriction coupling + nonlinear Hill tissue compliance
> with lymphatic clearance).
> Parameters are calibrated based on physiological literature but are **not** bench-validated against real IV clinical patient data.
> Results must not be claimed as clinical evidence.

> [!IMPORTANT]
> **Window Label Assignment & Bug Fix:**
> The per-timestep `label` column is used (0=none, 1=early, 2=moderate, 3=severe), with the window label defined as the maximum label within each 50s window.
> This accurately models real-time onset detection. In earlier versions of this project, assigning severity as a static trial-level label caused pre-onset windows in infiltrating trials to be mislabeled as infiltration, resulting in neural model collapse (macro-F1 ~0.103). Resolving this label definition restored high-fidelity classification across all models.

---

## 1. Experimental Setup

| Item | Value |
|------|-------|
| Dataset | `synthetic_iv_infiltration_dataset.csv` — 200 trials (360,000 timesteps) |
| Severities | none (50), early (50), moderate (50), severe (50) |
| Robustness Conditions | normal (40), noisy_temp (40), noisy_pressure (40), missing_temp (40), missing_pressure (40) |
| Trial Split | 120 train / 30 val / 50 test trials (Seed 42, stratified by severity, strictly trial-disjoint) |
| Window Length | 50s (default); ablations at 10s and 25s |
| Stride | 25s train (50% overlap), 50s eval (non-overlapping, zero window leakage) |
| Normalisation | Per-channel z-score normalisation, fit on training windows only, NaN-safe imputation |
| Threshold Tuning | Validation set percentile grid sweep for optimal balanced accuracy |
| Neural Training | AdamW, lr=5e-4, CosineAnnealingLR, class-weighted cross-entropy, patience=15 |
| Multi-Seed Protocol | Evaluated across seeds [42, 101, 202] for neural models; mean ± std reported |

---

## 2. Model Architectures & Parameter Footprint

| Model | Type | Architecture / Complexity | Parameters |
|-------|------|---------------------------|------------|
| **Threshold** | Heuristic Baseline | Dual-threshold on linear temperature and pressure window slopes | 2 parameters |
| **Logistic Regression** | Linear Tabular | Pipeline: 10 summary moments (mean, std, min, max, slope) + StandardScaler + L2 LogReg | 44 weights |
| **Random Forest** | Ensemble Tabular | 100 trees, max_depth=6, class_weight='balanced' on 10 summary moments | ~50 trees |
| **DeepResNet1D** | Deep Residual CNN | 6-layer 1D ResNet: Conv stem + 3 Residual Blocks with dilations [1, 1, 2] + GlobalAvgPool | ~100k weights |
| **TCN1D** | Causal Convolutional | 4 Causal Dilated Residual Blocks (dilations: 1, 2, 4, 8), strictly left-padded (no lookahead) | 59,908 weights |

---

## 3. Main Results — Full Sensor Set (Temp + Pressure, Window=50s)

Evaluated on held-out test trials (50 complete trials = 1,800 non-overlapping windows).

| Model | Macro-F1 | Balanced Acc | Raw Delay (s) | Clamped Delay (s) | Premature Detections | Confounder FPR (95% CI) |
|-------|----------|--------------|---------------|-------------------|----------------------|-------------------------|
| **Threshold Baseline** | 0.325 | 0.364 | −258.03s | 39.38s | 30 / 37 | 0.2361 [0.196, 0.276] (102/432) |
| **Logistic Regression** | 0.9684 | 0.9606 | +35.22s | 51.68s | 19 / 37 | **0.0000 [0.000, 0.000]** (0/432) |
| **Random Forest** | **0.9857** | **0.9901** | −28.30s | 5.00s | 30 / 37 | **0.0000 [0.000, 0.000]** (0/432) |
| **DeepResNet1D** (3 seeds) | 0.9634 ± 0.0078 | 0.9769 | −40.91s ± 1.27s | 1.03s | 33 / 37 | **0.0000 [0.000, 0.000]** (0/432) |
| **TCN1D** (3 seeds) | 0.9572 ± 0.0012 | 0.9579 | −93.16s ± 30.48s | 0.08s | 36 / 37 | 0.0023 [0.000, 0.007] (1/432) |

### Detection Delay & Sustained Alert Analysis
- **Definition:** Sustained detection requires $\ge 2$ consecutive positive window predictions ($\ge$ class 1). Raw delay is calculated as $t_{\text{detection}} - t_{\text{onset}}$ relative to physical pump onset.
- **Negative Delays (Premature Alerts):** All non-linear models exhibit negative mean raw delays (-28s to -93s). Because the sensor signals begin deviating smoothly before the strict mathematical label boundary transitions, sensitive models trigger early alarms.
- **Clamped Delays:** When clamped at $0.0s$ ($t_{\text{detection}} \ge t_{\text{onset}}$), TCN detects infiltration almost immediately upon label onset (0.08s mean clamped delay), DeepResNet at 1.03s, and Random Forest at 5.00s.
- **Missed Infiltrations:** Zero across all models (all 37 infiltration test trials were successfully flagged).

---

## 4. Systematic Ablation Experiments

### A. Sensor Modality Ablation (TCN1D, 3 seeds)
Evaluates the relative importance of local temperature vs. tissue pressure:

| Modality Subset | Channels | Macro-F1 (Mean ± Std) | Key Observation |
|-----------------|----------|-----------------------|-----------------|
| **Temperature Only** | `temperature_c` | 0.5784 ± 0.0122 | Severe degradation: thermal diffusion is sluggish and vulnerable to noise |
| **Pressure Only** | `pressure_adc` | 0.9540 ± 0.0109 | Retains 99.6% of full performance: biomechanical strain is the primary driver |
| **Temp + Pressure** | Both | **0.9572 ± 0.0012** | Dual-modality provides optimal accuracy and minimizes cross-seed variance |

### B. Window Length Sensitivity (TCN1D, Seed 42)
Evaluates temporal receptive field lookback:

| Window Size (seconds) | Macro-F1 | Receptive Field Adequacy |
|-----------------------|----------|--------------------------|
| **10s** | 0.9086 | Suboptimal: high-frequency noise creates transient false triggers |
| **25s** | 0.9195 | Moderate: captures early derivative changes |
| **50s** | **0.9564** | Optimal: sufficiently integrates physiological slope dynamics |

### C. Temporal Order Shuffle Control (Temporal Structure Test)
To verify whether the deep sequence models actually exploit sequential ordering or merely acting on static window distributions, the time axis within each window was randomly permuted during training:

| Model Architecture | Original Macro-F1 | Shuffled Time Macro-F1 | Performance Drop ($\Delta$) |
|--------------------|-------------------|------------------------|------------------------------|
| **TCN1D** | 0.9572 | 0.9551 | −0.0021 (0.2% drop) |
| **DeepResNet1D** | 0.9634 | 0.9680 | +0.0046 (no degradation) |

> [!IMPORTANT]
> **Key Finding on Temporal Modeling:**
> Shuffling the time-axis within windows produces virtually zero performance degradation (<0.003 F1 drop). In physical fluid infiltration, accumulation manifests as sustained level shifts and monotonic trends (increasing pressure, decreasing temperature). Because first- and second-order statistical moments (mean, min, max, linear slope) capture almost all necessary information, complex causal convolutional sequence modeling provides no significant architectural advantage over tabular ensembles operating on window summary features.

### D. Data Scarcity Ablation (20% Training Trials, N=24 trials, 3 seeds)
Evaluates sample efficiency when labeled infiltration data is severely constrained:

| Model Architecture | 20% Data Macro-F1 (Mean ± Std) | Full Data Macro-F1 | Scarcity Drop |
|--------------------|--------------------------------|--------------------|---------------|
| **Random Forest** | **0.9752 ± 0.0067** | 0.9857 | **−0.0105** |
| **TCN1D** | 0.8356 ± 0.0405 | 0.9572 | −0.1216 |

> [!TIP]
> **Sample Efficiency Conclusion:**
> In data-scarce settings (24 training trials), Random Forest retains 98.9% of its full performance (0.975 F1), whereas TCN degrades substantially to 0.836 F1 (a 12.2% drop). For edge deployment where labeled clinical events are scarce, hand-crafted feature pipelines with tree ensembles are far more sample-efficient and robust than deep neural networks.

---

## 5. Robustness & Confounder Analysis

Evaluated on held-out test trials partitioned by physical simulation scenario:

| Scenario / Condition | Test Trials | Windows | TCN Macro-F1 | Balanced Acc | Confounder Rejection |
|----------------------|-------------|---------|--------------|--------------|----------------------|
| **Normal Infusion** | 7 | 252 | 0.8818 | 0.8456 | Baseline clean conditions |
| **Noisy Temperature** | 13 | 468 | 0.9601 | 0.9493 | Robust: relies on pressure |
| **Noisy Pressure** | 13 | 468 | 0.9674 | 0.9720 | Robust: filtering via dilation |
| **Missing Temperature** | 11 | 396 | 0.9759 | 0.9812 | Imputed seamlessly |
| **Missing Pressure** | 6 | 216 | 0.9390 | 0.9504 | Temperature slope acts as backup |

### Confounder Specificity
- Confounder trials consist of non-infiltrating infusions subjected to sensor noise and systemic disturbances.
- While the simple **Threshold Baseline** collapsed under confounders with a **23.61% False Positive Rate** (102 false alarms across 432 windows), **Random Forest**, **Logistic Regression**, and **DeepResNet** achieved **0.00% FPR** (0 false alarms).
- **TCN1D** achieved a negligible **0.23% FPR** (1 false alarm out of 432 windows).

---

## 6. Summary of Architectural Takeaways

1. **Random Forest is the Strongest Baseline:** With 0.9857 Macro-F1, 0.00% Confounder FPR, and 0.9752 F1 under 20% data scarcity, Random Forest on 10 summary moment features outperforms both Deep 1D-ResNet (0.9634) and causal TCN (0.9572).
2. **Pressure Dominates Detection:** Bioimpedance and pressure reflect mass accumulation directly; temperature is a secondary, sluggish modality (0.578 F1 alone).
3. **Temporal Networks Are Not Inherently Superior Here:** Because infiltration is characterized by monotonic drifts rather than complex periodic or semantic temporal grammar, causal temporal convolutions do not outperform moment-based tabular models.
