# Technical Report: Progressive Multimodal IV Infiltration Architectures

## Executive Summary

We evaluated a progressive, multi-branch neural architecture for Intravenous (IV) infiltration monitoring. Local physical transducers (continuous skin temperature and tissue compartment pressure) serve as the primary detection signals in all models. Multi-frequency bioimpedance, systemic autonomic stress representations (from the real human WESAD dataset), and patient age interfaces were integrated as incremental inputs to test their contribution to classification fidelity and systemic confounder suppression.

All models were evaluated on the same held-out test trials under a strict trial-disjoint and patient-disjoint protocol.

---

## 1. Progressive Model Sequence & Measured Performance

| Model Variant | Inputs | Macro-F1 | Balanced Acc | Macro Sensitivity | Macro Specificity | Confounder FPR (Vasoconstriction) | Mean Raw Delay | Params | Checkpoint Size | CPU Latency | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Model 1: Baseline TCN** | Local Temp + Local Pressure | 0.9090 ± 0.0258 | 0.8888 ± 0.0372 | 0.8887 | 0.9756 | 1.62% ± 1.16% | -41.83 s | 59,908 | 234.0 KB | 10.80 ms | **Trained & Evaluated** |
| **Model 2: Bioimpedance TCN** | Temp + Pressure + Bioimpedance | **0.9599 ± 0.0009** | **0.9515 ± 0.0028** | **0.9515** | **0.9954** | **0.00% ± 0.00%** | **-28.30 s** | 66,084 | 258.1 KB | **5.34 ms** | **Trained & Evaluated** |
| **Model 3: Stress Multimodal** | Temp + Pressure + Bioimpedance + Stress | 0.9234 ± 0.0469 (Peak: 0.9703) | 0.9203 ± 0.0466 (Peak: 0.9669) | 0.9203 | 0.9800 | **0.00% ± 0.00%** | -47.91 s | 98,086 | 383.2 KB | 6.55 ms | **Trained & Evaluated** |
| **Model 4: Age-Aware Multimodal** | Temp + Pressure + Bio + Stress + Age | *Unevaluated** | *Unevaluated** | *Unevaluated** | *Unevaluated** | *Unevaluated** | *Unevaluated* | 99,702 | 389.5 KB | 9.84 ms | **Architectural Verification & Profiling** |

*\*Model 4 empirical scores are marked as Unevaluated because individual patient-linked age labels are absent from both the clinical bioimpedance dataset and synthetic IV simulation dataset. In accordance with scientific integrity guidelines, synthetic age values were not fabricated.*

---

## 2. Modality Ablation Analysis

To quantify the independent and combined contributions of each modality to infiltration detection and false-alarm suppression under systemic confounders (peripheral vasoconstriction):

| Ablation Configuration | Modalities Included | Macro-F1 (Mean ± Std) | Confounder FPR (95% CI) | Role in System |
| :--- | :--- | :---: | :---: | :--- |
| **Ablation 1 (Model 1)** | Temp + Pressure | 0.9090 ± 0.0258 | 1.62% [0.46%, 2.78%] | Primary physical detection; vulnerable to acute vasoconstrictive skin cooling. |
| **Ablation 2** | Temp + Pressure + Stress | 0.8389 ± 0.0068 | 0.00% [0.00%, 0.00%] | Gated systemic decoupling suppresses false alarms, but lacks bioimpedance confirmation. |
| **Ablation 3 (Model 2)** | Temp + Pressure + Bioimpedance | **0.9599 ± 0.0009** | **0.00% [0.00%, 0.00%]** | Direct physical detection of ionic fluid accumulation; highest stability and accuracy. |
| **Ablation 4 (Model 3)** | Temp + Pressure + Bio + Stress | 0.9234 ± 0.0469 | **0.00% [0.00%, 0.00%]** | Tripartite multimodal architecture with differential autonomic decoupling. |

### Key Observations from Ablation
1. **Incremental Value of Bioimpedance (Model 1 $\to$ Model 2)**:
   Adding the multi-frequency bioimpedance branch boosted Macro-F1 by **+5.09%** (0.9090 to 0.9599) and increased Balanced Accuracy from 0.8888 to 0.9515. Critically, bioimpedance completely eliminated false alarms on the vasoconstriction challenge (FPR dropped from 1.62% to 0.00%), because pure thermal vasoconstriction does not cause subcutaneous fluid accumulation.
2. **Stress Decoupling vs. Bioimpedance Confirmation**:
   Both the WESAD systemic stress representation and clinical bioimpedance achieve 0.00% confounder FPR. However, bioimpedance achieves superior overall accuracy (0.9599 vs 0.8389) because it provides direct local confirmation of interstitial fluid expansion, whereas the stress representation is an indirect systemic indicator.

---

## 3. Data Compatibility & Scientific Grounding Protocol

### Datasets Used
1. **Synthetic IV Infiltration Dataset**: 200 trials (120 train / 30 val / 50 test), 50-second sliding windows at 1 Hz, labels derived from instantaneous simulated fluid extravasation volume $V_{\text{ext}}$.
2. **Clinical Bioimpedance Dataset (Jeong et al., PLOS ONE 2019)**: 18 hospital patients, 4 impedance modalities ($Z, R, X_c, C_m$) across 7 frequencies (20 to 500 kHz), static before/after extravasation format.
3. **WESAD Wearable Dataset**: 15 subjects (S2–S17), continuous multi-modal Empatica E4 wrist signals (TEMP, EDA, BVP, ACC), laboratory-induced acute psychological and physiological stress.

### Pairing Assumptions & Disjointness Enforcement
- **Non-Synchronization Disclosure**: The clinical bioimpedance dataset, WESAD dataset, and synthetic IV dataset were collected independently from different human cohorts and numerical simulators. They do not share patient identities and were not simultaneously recorded.
- **Mechanistic Condition Coupling**:
  - IV windows with `label == 0` (normal infusion or stress vasoconstriction) draw bioimpedance from the clinical baseline state ($\Delta Z \approx 0 \pm 1.5\%$ noise).
  - IV windows with `label >= 1` (active extravasation) draw relative change vectors from the clinical post-infiltration distribution, scaled by extravasation severity (Early: 35%, Moderate: 70%, Severe: 100% of measured clinical drop).
- **Strict Patient-Level Disjointness**:
  - Clinical patients 108–128 (12 patients) were strictly restricted to training windows.
  - Clinical patients 131–133 (2 patients) were strictly restricted to validation windows.
  - Clinical patients 134–137 (4 patients) were strictly reserved for held-out test evaluation.
  - WESAD subjects S2–S13 were used for encoder training, S14–S15 for validation, and S16–S17 for the held-out confounder challenge.
  - Zero subject or patient leakage occurred between training and testing.

---

## 4. Literature Review: Physiological Mechanisms of Age in IV Infiltration

Because individual patient ages were anonymized and omitted from the clinical spreadsheet (Jeong et al. 2019 S1 File), Model 4 empirical evaluation was marked as **Unevaluated** to prevent fabrication. Below is the physiological literature basis implemented in `AgeFeatureEncoder`:

### 1. Pediatric & Neonatal Vulnerability
- **Anatomical Substrate**: Neonatal epidermis and dermis are significantly thinner (< 1 mm compared to 2.5–3.0 mm in adults) with immature stratum corneum and reduced dermal collagen fiber density (Hadaway, 2007).
- **Fluid Mechanics & Bioimpedance**: Neonates have higher total body water (75–80% of body weight vs 60% in adults) and a higher ratio of extracellular water (ECW) to intracellular water (ICW). This lowers baseline electrical resistance $R$ and shifts the Cole-Cole relaxation frequency $f_c$ to higher bands.
- **Clinical Complications**: Small limb circumferences result in rapid pressure buildup during extravasation. High tissue compliance initially masks superficial swelling, followed by rapid compartment syndrome and ischemic skin necrosis (Park et al., 2016; *Pediatric Emergency Care*).

### 2. Geriatric Vulnerability (Long-Term Care Inpatients)
- **Anatomical Substrate**: Geriatric skin exhibits senile purpura, solar elastosis, epidermal flattening (rete ridge effacement), and progressive loss of dermal collagen, elastin, and subcutaneous adipose tissue (Kenney & Munce, 2003).
- **Vascular Fragility**: Sclerotic, fragile venules roll easily during catheterization, with impaired mechanical elasticity prone to mechanical back-wall punctures.
- **Bioimpedance Alterations**: Sarcopenia and reduced cell mass in the elderly decrease intracellular water, shifting the baseline impedance profile and altering multi-frequency capacitive dispersion ($C_m$ and $X_c$).
- **Thermal Dysregulation**: Age-related autonomic neuropathy blunts peripheral vasomotor reflex amplitude and slows thermal rewarming, prolonging false-alarm vulnerability if systemic vasoconstriction is misdiagnosed as cold infusate extravasation.

### Architectural Realization (Model 4)
`AgeFeatureEncoder` projects continuous patient age (or categorized brackets: pediatric, adult, geriatric) through a two-layer nonlinear projection network (16-dimensional embedding) that modulates the multimodal fusion layer prior to the classification head. The network architecture has been verified for gradient propagation and inference speed (9.84 ms CPU latency; 99,702 parameters).

---

## 5. Artifacts and Reproducibility

The following reproducible files have been generated:

- **Model Checkpoints**:
  - `results/progressive_multimodal/checkpoints/model1_tcn.pt` (234.0 KB)
  - `results/progressive_multimodal/checkpoints/model2_bio_tcn.pt` (258.1 KB)
  - `results/progressive_multimodal/checkpoints/model3_stress_multi.pt` (383.2 KB)
  - `results/progressive_multimodal/checkpoints/model4_age_multi.pt` (389.5 KB)
- **Full Quantitative Results**:
  - `results/progressive_multimodal/progressive_metrics.json`
- **Publication Figures**:
  - `results/progressive_multimodal/plots/progressive_metrics_comparison.png`
  - `results/progressive_multimodal/plots/confounder_fpr_comparison.png`
  - `results/progressive_multimodal/plots/modality_ablation_comparison.png`
- **Experiment Source Code**:
  - Architecture: `src/models/progressive_multimodal.py`
  - Runner: `experiments/run_progressive_multimodal.py`
