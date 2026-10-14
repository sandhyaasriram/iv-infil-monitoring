# Research Options: Cross-Domain Physiological Representation Learning for IV Infiltration

**Evaluation Date:** October 2026  
**Objective:** Formulate defensible, technically substantial research contributions that connect real physiological representations with IV-specific state estimation under severe label scarcity.

---

## 1. Literature Reality & Novelty Landscape

A rigorous review of the biomedical monitoring and TinyML literature reveals the following boundaries:

1. **What is Already Done in IV Infiltration:**
   * Single or multi-sensor local thresholding/heuristics (DS18B20 temperature drop, FSR force/displacement, bioimpedance magnitude).
   * Supervised standard machine learning (Logistic Regression, SVM, Random Forest) trained purely on synthetic ODE simulations or small bench phantom arrays.
   * Porcine preclinical studies testing bioimpedance and strain gauges (Bicen et al., 2018).
   * Static clinical impedance comparisons before/after infiltration (Jeong et al., 2019).
2. **What is Already Done in General Physiological Wearables:**
   * Self-supervised contrastive learning (SimCLR, TS-TCC, CPC) on multi-channel physiological data (ECG, PPG/BVP, EDA, skin temperature) to capture autonomic arousal and circadian rhythms.
   * Anomaly detection algorithms isolating motion artifacts from true physiological events.
3. **The Unsolved Technical Challenge in IV Monitoring:**
   * **The Systemic Confounding Problem:** Peripheral skin temperature and microvascular tone fluctuate naturally due to systemic autonomic events (stress, anxiety, cold ambient exposure, shivering, pain). An edge monitor relying on local temperature or baseline contact strain suffers frequent false alarms when a patient experiences systemic vasoconstriction.
   * **The Target Data Scarcity Problem:** Real human continuous IV infiltration records cannot be collected in vast quantities due to patient safety ethics (accidental extravasation is a medical complication).

---

## 2. Candidate Research Contributions

### Option A: Autonomic-Aware Dual-Stream State Estimator (Systemic Confounder Decoupling)
* **Mechanistic Hypothesis:**
  A representation of systemic autonomic state learned from real multi-modal physiological time series (WESAD wrist BVP, EDA, TEMP, ACC) enables a monitoring model to distinguish systemic skin-temperature cooling (vasoconstriction) from true local fluid infiltration (convective cooling + local edema strain).
* **Technical Architecture:**
  * *Stream 1 (Systemic Prior Encoder):* Temporal 1D-CNN or GRU pre-trained on real WESAD wrist recordings. Produces a 16-dimensional systemic state vector $\mathbf{z}_{sys}(t)$ capturing autonomic arousal and peripheral vasomotor tone.
  * *Stream 2 (Local Infiltration Monitor):* Edge-friendly feature processor ingestive of local IV site readings (temperature slope, local mechanical strain).
  * *Fusion & Differential Decision Head:* Computes the discrepancy between expected systemic peripheral cooling and observed local cooling/strain.
* **Evaluation & Baselines:**
  * *Baseline 1:* Existing Local-Only Random Forest (from 	rain_and_export.py).
  * *Baseline 2:* Naive concatenation of raw physiological features.
  * *Proposed Model:* Systemic-decoupled differential estimator.
  * *Key Test:* Stress/Vasoconstriction False Alarm Injection. When a patient undergoes sudden sympathetic cooling without leakage, does the proposed model suppress false positive alarms?
* **Feasibility in 48 Hours:** **High.** WESAD data is already downloaded locally; model components are straightforward PyTorch modules exportable to ONNX / TinyML.

---

### Option B: Self-Supervised Masked Temporal Autoencoder (Pre-training on Real Wearables)
* **Mechanistic Hypothesis:**
  A 1D temporal autoencoder pre-trained with masked autoencoding on unlabelled real peripheral time series (WESAD wrist temperature, microvascular pulse, accelerometry) learns universal dynamics of human skin-sensor coupling, accelerating adaptation and boosting sample efficiency when fine-tuned on scarce local IV infiltration data.
* **Technical Architecture:**
  * Pre-train a lightweight 1D Patch-TimeNet or 1D-CNN encoder on WESAD (15 subjects, unlabelled) to reconstruct masked segments of real peripheral signals.
  * Freeze or fine-tune the encoder head with a low-rank adapter on small-sample IV scenarios.
* **Novelty & Limitations:**
  * Scientifically rigorous, but requires evaluating target performance across varying fractions of scarce target data (e.g., 5%, 10%, 20% training trials).
  * Implementation involves slightly more training overhead.

---

### Option C: Calibrated Biomechanical Surrogate Transfer (Anchor-Tuned Simulation + Real Clinical Benchmarking)
* **Mechanistic Hypothesis:**
  Grounding the synthetic simulator parameters (, K_P, K_L$) in published porcine strain data (Bicen 2018) and validating the learned representation against the real clinical human dataset (PLOS ONE S1 File) produces a clinically defensible domain-adapted estimator rather than an arbitrarily tuned toy.
* **Feasibility:** High as an essential validation benchmark, and directly complements Option A.

---

## 3. Recommended Research Direction

**Select Option A (Autonomic-Aware Dual-Stream State Estimator) supported by Option C's real clinical benchmark.**

### Why this is defensible and technically substantial:
1. **Solves a Real Clinical Failure Mode:** Clinicians know that patients get cold, stressed, or move their arms. A naive sensor calls this an infiltration alarm. Decoupling systemic physiology from local extravasation is an explicit, novel systems contribution in edge biomedical monitoring.
2. **Strict Data Integrity:** It does NOT falsely label WESAD stress as infiltration. WESAD is used explicitly for what it is: *real systemic autonomic physiology*.
3. **Dual Validation:**
   * Real Data Validation: Evaluates representation quality and false-positive suppression during real human sympathetic arousal (WESAD).
   * Real Clinical Validation: Confirms impedance/biophysical sensitivity using the real human clinical dataset (PLOS ONE S1 File).
