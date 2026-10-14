# Experiment Plan: Physiological Representation Learning & Infiltration State Estimation

**Protocol Date:** October 2026  
**Hypothesis:** Physiological representations of autonomic and vasomotor dynamics learned from real wearable data (WESAD) significantly reduce false-alarm rates during systemic physiological confounders and improve infiltration detection under scarce local observations, compared to local-only baselines.

---

## 1. Experimental Architecture & Data Splits

### 1.1 Source Domain (Real Physiological Data - WESAD)
* **Dataset:** 15 subjects ( \dots S17$, excluding $).
* **Subject-Disjoint Split (Strict Leave-Subjects-Out):**
  * **Train:** 11 subjects (, S3, S4, S5, S6, S7, S8, S9, S10, S11, S13$)
  * **Validation:** 2 subjects (, S15$)
  * **Test:** 2 subjects (, S17$)
* **Signals:** Synchronous resampled wrist bundle (\text{ Hz}$):
  * TEMP (skin temperature, \circ\text{C}$)
  * EDA (electrodermal activity, $\mu\text{S}$)
  * BVP_RMS (blood volume pulse envelope/amplitude from 64 Hz PPG)
  * ACC_MAG (3D acceleration magnitude from 32 Hz ACC)
* **Window Size:** 25 seconds (100 samples at 4 Hz), aligned with the 25-second infiltration windowing interval.

### 1.2 Target Domain (IV State Estimation & Confounder Challenge)
* **Standard Test Set:** Disjoint synthetic IV trials (held-out trials).
* **Confounder Challenge Test Set:** Inject real human autonomic vasoconstriction episodes (from WESAD test subjects , S17$) where skin temperature drops .5\text{--}1.5^\circ\text{C}$ in the absence of any fluid leakage.
* **Clinical Impedance Target Test Set:** The 18 human clinical infiltration cases from the PLOS ONE dataset (	est_s1.xlsx / S1 File).

---

## 2. Models to Compare (Baselines vs Proposed)

1. **Baseline 1 (Current Repository Baseline):**
   * Handcrafted features (mean, std, slope, min, max) + Scikit-Learn RandomForestClassifier. Evaluated with proper trial-level holdout.
2. **Baseline 2 (Linear / Regularized):**
   * Same handcrafted features + LogisticRegression(penalty='l2').
3. **Baseline 3 (Standard 1D-CNN Temporal Baseline):**
   * End-to-end 1D convolutional neural network trained solely on local signals without pre-training or systemic inputs.
4. **Proposed Model (Autonomic-Decoupled Dual-Stream Network):**
   * Pre-trained temporal encoder for systemic autonomic state $\mathbf{z}_{sys}$ from real signals.
   * Differential gating head evaluating whether observed temperature decrease matches systemic vasomotor expectation or indicates localized extravasation.

---

## 3. Evaluation Metrics

1. **Macro-Averaged F1 Score:** Harmonic mean of precision and recall across the 4 severity classes (
one, early, moderate, severe).
2. **False Positive Rate (FPR) under Systemic Confounding:** Proportion of non-infiltrated windows during stress/cooling confounder trials misclassified as infiltration ($\ge$ early).
3. **Early-Stage Detection Sensitivity (Recall on Class 1):** Detection ability before fluid leakage reaches $>1.5\text{ mL}$.
4. **Time-to-Detection Delay ({delay}$):** Elapsed seconds from true onset of extravasation ({onset}$) until the model first issues sustained infiltration predictions ($\ge 2$ consecutive windows).
5. **Edge Footprint:** Inference latency (ms), parameter count, and C export memory budget for ESP32.

---

## 4. Hypothesis Acceptance / Rejection Criteria

* **Accept Hypothesis IF:**
  1. Under systemic vasoconstriction confounders, the proposed representation reduces false positive alarms by $\ge 40\%$ relative to the local-only Random Forest baseline (e.g. FPR drops from $>30\%$ to $<15\%$), **AND**
  2. In the low-data regime ($\le 25\%$ of available IV training trials), the transferred representation achieves higher macro-F1 than the randomly initialized temporal model ( < 0.05$ over 5 random seeds).
* **Reject Hypothesis IF:**
  1. The systemic representation fails to lower false alarm rates during non-infiltrated stress/cooling episodes, OR
  2. Direct end-to-end local training matches or outperforms the transfer architecture in sample efficiency and detection accuracy.
