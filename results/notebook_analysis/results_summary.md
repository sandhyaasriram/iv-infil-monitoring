# Executive Results Summary: IV Infiltration Monitoring

**Generated:** 2026-10-15 06:19  
**Source Notebook:** `notebooks/IV_Infiltration_Complete_Results_Analysis.ipynb`  
**Output Directory:** `results/notebook_analysis/`  

---

## 1. Key Performance Highlights

1. **Model 1 (Primary Validated Causal TCN):**
   - **Inputs:** Local Temperature + Local Compartment Pressure (60s rolling windows, 1 Hz).
   - **Macro-F1:** **0.9007 +/- 0.0183** | **Balanced Accuracy:** **0.8765**.
   - **Confounder Suppression:** Vasoconstriction false positive rate restricted to **1.15%**.
   - **Edge Profile:** **59,908 parameters**, **234.02 KB**, **5.011 ms** CPU latency.

2. **Clinical Bioimpedance Discrepancy Unpacked (18 Inpatients, Jeong et al. 2019):**
   - **Retrospective Paired Diagnostic (Delta Z / Z0):** **97.22% Balanced Accuracy** (1.000 ROC-AUC).
   - **Prospective Blind Single-Point Screening:** **72.22% Balanced Accuracy** (0.793 ROC-AUC).
   - **Physiological Mechanism:** Baseline impedance varies from 418 Ohm to 992 Ohm across patients. 66.7% of post-infiltration measurements overlap with other patients' healthy baselines. Self-referencing is essential for clinical validity.
   - **Sample Size Constraint:** Wilson 95% CI for sensitivity spans [74.2%, 99.0%]; this represents a preliminary proof-of-concept, not clinical validation.

3. **Model 2 (Simulated Bioimpedance Fusion):**
   - **Macro-F1:** **0.9689** | **Confounder FPR:** **0.00%**.
   - **Audit Disclosure:** Forward-simulated bioimpedance is mathematically coupled to the latent ODE fluid volume state variable.

4. **Model 4 (Age-Aware Architecture):**
   - **Status:** **UNEVALUATED (Data Blocked)**. Patient ages are absent from all available data; empirical performance was not fabricated.

---

## 2. Generated Visualizations & Tables
- `figure1_clinical_discrepancy_and_overlap.png`
- `figure2_model1_confusion_and_perclass.png`
- `figure3_model1_vs_model2_comparison.png`
- `figure4_resource_latency_parameters.png`
- `figure5_frequency_ablation_clinical.png`
- `table_clinical_loocv_comparison.csv`
- `table_progressive_models_comparison.csv`
- `table_resource_and_latency.csv`
- `table_metric_provenance_audit.csv`
