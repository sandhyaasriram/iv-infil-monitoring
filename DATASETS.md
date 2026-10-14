# Dataset Assessment: Real & Physiological Datasets for IV Monitoring

**Assessment Date:** October 2026  
**Project:** Low-Cost Edge-Based IV Infiltration Monitoring

---

## 1. Verified Local Dataset: WESAD (Wearable Stress and Affect Detection)

* **Official Source:** UCI Machine Learning Repository (DOI: 10.24432/C55W31) / ACM Ubicomp 2018 (Schmidt et al.).
* **Local Location:** c:/mer/iot projecttt/WESAD/ (Verified: 15 subjects S2 to S17, excluding S12 per study protocol).
* **Licence & Access:** Academic open use (Creative Commons Attribution 4.0 International). No login or credentials needed.
* **Subjects & Demographics:**
  * 15 healthy adult subjects (12 male, 3 female, mean age 27.5 ± 2.4 years).
  * Metadata per subject (quest.csv, 
eadme.txt): Age, height, weight, gender, dominant hand, caffeine/tobacco intake.
* **Modalities, Channels & Exact Sampling Rates:**
  * **Chest (RespiBAN Professional, 700 Hz synchronous):**
    * ECG: 1-channel electrocardiogram (mV).
    * EDA: Electrodermal activity ($\mu	ext{S}$).
    * EMG: Electromyogram (trapezius).
    * Temp: Skin temperature (\circ	ext{C}$).
    * Resp: Respiration chest expansion.
    * ACC: 3-axis acceleration ($).
  * **Wrist (Empatica E4, asynchronous multi-rate):**
    * BVP: Blood volume pulse (Photoplethysmography, PPG) at **64 Hz**.
    * EDA: Electrodermal activity at **4 Hz**.
    * TEMP: Skin surface temperature (\circ	ext{C}$) at **4 Hz**.
    * ACC: 3-axis accelerometer at **32 Hz**.
    * HR: Derived heart rate at **1 Hz**.
    * IBI: Inter-beat intervals (sparse events).
* **Labels:**
  * Protocol-based ground-truth labels at 700 Hz:
    * 0 = Baseline (20 min neutral)
    * 1 = Stress (Trier Social Stress Test: public speaking + mental arithmetic)
    * 2 = Amusement (funny video clips)
    * 3 = Meditation / De-excitation
    * 4 = Transient / Protocol transition
    * 6/7 = Additional unscored conditions
* **Relevance & Physical Limitation for IV Monitoring:**
  * *Strength:* Contains real, synchronous wrist skin temperature (TEMP), microvascular pulse (BVP/PPG), autonomic arousal (EDA), and motion (ACC).
  * *Critical Limitation:* **Contains ZERO IV infiltration events.** WESAD measures systemic autonomic nervous system responses (sympathetic activation causing peripheral vasoconstriction and wrist temperature drops).
  * *Strict Integrity Rule:* Under no circumstances will WESAD stress labels be substituted as IV infiltration labels.

---

## 2. Dataset B: Autonomic Aging (Cardiovascular Recordings)

* **Official Source:** PhysioNet (utonomic-aging-cardiovascular/1.0.0/) (Schumann et al., Scientific Data 2021).
* **Access Requirements:** Open access under PhysioNet Credentialed / Open Health Data licence (Requires ODC-BY 1.0; no credentialed PhysioNet sign-off needed for public bucket).
* **Signals & Formats:**
  * WFDB binary .dat + .hea headers.
  * Continuous Non-Invasive Blood Pressure (CNBP) at **100 Hz**.
  * Continuous Lead II ECG at **1000 Hz**.
  * Demographics (subject-info.csv): Age brackets (e.g., 20–24, 60–64), generalized BMI, sex ($-anonymized).
* **Feasibility & Role:**
  * Excellent for learning age-dependent cardiovascular baseline priors (heart rate variability, pulse transit time, autonomic tone changes with age).
  * Does not contain local limb swelling, limb temperature, or IV infiltration events.

---

## 3. Dataset C: IV-Specific Research Data

### 3.1 PLOS ONE Clinical Study (Jeong et al., 2019)
* **Title:** *Detection of intravenous infiltration using impedance parameters in patients in a long-term care hospital* (PLOS ONE 14(3): e0213585).
* **Data Availability Status:** **Verified & Downloadable.**
  * Available directly via publisher supplementary file: journal.pone.0213585.s001 (.xlsx).
* **Actual Contents (Direct Inspection):**
  * Contains numerical clinical records from 18 elderly hospital patients experiencing real IV infiltration.
  * Measurements: Bioelectrical impedance parameters at multiple frequencies (20 kHz, 50 kHz, 100 kHz, 150 kHz, 200 kHz, 300 kHz, 500 kHz).
  * Measured matrices: $ (impedance magnitude, $\Omega$), $ (resistance, $\Omega$), $ (reactance, $\Omega$), $ (membrane capacitance, nF).
  * Format: Static paired readings (**Before infiltration vs. After infiltration**).
  * *Limitation:* It is **not a high-frequency continuous time series**. It provides static pre/post physiological feature vectors of genuine human infiltration.

### 3.2 Porcine Model Study (Bicen, Inan et al., IEEE JTEHM 2018)
* **Title:** *Toward Non-Invasive and Automatic Intravenous Infiltration Detection: Evaluation of Bioimpedance and Skin Strain in a Pig Model*.
* **Data Availability Status:** **Restricted / Not Openly Deposited.**
  * No public Zenodo, GitHub, or IEEE DataPort repository exists for this paper.
  * Raw continuous strain and bioimpedance recordings require formal author request.
  * Quantitative response curves (e.g. strain $\Delta \varepsilon$ vs. leaked volume $, ROC curves for \text{ mL}, 5\text{ mL}, 10\text{ mL}$) can be extracted via high-fidelity digitization for benchmark calibration.

---

## 4. Dataset D: MIMIC-IV-ED (Emergency Department)

* **Source:** PhysioNet (mimic-iv-ed/2.2/).
* **Access Requirements:** Requires CITI Human Subjects Training and signed PhysioNet Data Use Agreement (DUA).
* **Suitability:** Contains vital signs (triage heart rate, blood pressure, peripheral SpO2, temperature) across 400,000+ patient visits. However, IV infiltration events are not systematically coded as structured alarm times. Given access onboarding delays (2–7 days), it is excluded from the immediate 48-hour timeline.

---

## 5. Summary Matrix & Viable Pipeline Roles

| Dataset | Local Availability | Modality / Signals | Ground Truth | Role in Proposed Research Pipeline |
| :--- | :--- | :--- | :--- | :--- |
| **WESAD** | **100% Downloaded** (15 subj, ~18 GB) | Wrist TEMP (4Hz), BVP (64Hz), EDA (4Hz), ACC (32Hz); Chest ECG, Resp | Autonomic state (Neutral, Stress, Amusement) | **Source Domain:** Learn real physiological autonomic/microvascular representation to separate systemic arousal from local anomalies. |
| **PLOS ONE (Jeong 2019)** | **Direct Download Verified** (.xlsx) | Multi-frequency bioimpedance (, R, X_c, C_m$) | Real human clinical infiltration (Before vs After) | **Target Clinical Benchmark:** Ground-truth validation of real tissue parameter shifts during actual human infiltration. |
| **Porcine Model (Bicen 2018)** | Documented in paper | Strain ($\mu\varepsilon$), Bioresistance ($\Omega$) vs Leaked Volume (	ext{--}10	ext{ mL}$) | Animal infusion vs infiltration (2, 5, 10 mL) | **Tissue Calibration Anchor:** Physical calibration curve to replace arbitrary , K_P$ simulation values. |
| **Synthetic Simulator** | **Local Python** (simulator/) | Forearm Temp (	ext{ Hz}$), Local Pressure/Strain (	ext{ Hz}$) | Dynamic leaked volume (	ext{--}15	ext{ mL}$) | **Target Emulation:** Test bed for edge detection delay, battery/compute constraints, and real-time streaming inference. |
