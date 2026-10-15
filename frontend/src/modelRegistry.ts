/**
 * modelRegistry.ts
 * ================
 * Centralized registry for IV Infiltration Detection Models.
 * Contains validated parameter counts, CPU latencies, empirical evaluation statuses,
 * input channels, checkpoints, and benchmark metrics across all progressive architectures.
 */

export interface ModelSpecification {
  id: string;
  name: string;
  shortName: string;
  category: 'primary' | 'multimodal' | 'exploratory' | 'baseline';
  status: 'Primary Validated' | 'Bioimpedance Fusion' | 'Dual-Stream Stress Aware' | 'Age Architecture (Data Blocked)' | 'Trained Baseline';
  description: string;
  inputModalities: string[];
  totalParameters: number;
  cpuLatencyMs: number;
  sizeKb: number;
  macroF1: number | null;
  balancedAccuracy: number | null;
  confounderFpr: number | null;
  detectionDelayS: number | null;
  checkpointPath: string;
  liveInferenceAvailable: boolean;
  notes: string;
}

export const MODEL_REGISTRY: Record<string, ModelSpecification> = {
  model1_tcn: {
    id: 'model1_tcn',
    name: 'Model 1: Baseline Causal TCN',
    shortName: 'Model 1 (TCN)',
    category: 'primary',
    status: 'Primary Validated',
    description: 'Causal dilated Temporal Convolutional Network (receptive field 61s) operating on local temperature and pressure time series.',
    inputModalities: ['Local Skin Temperature (DS18B20)', 'Local Tissue Pressure (FSR)'],
    totalParameters: 59908,
    cpuLatencyMs: 5.01,
    sizeKb: 234.0,
    macroF1: 0.9008,
    balancedAccuracy: 0.8766,
    confounderFpr: 0.0115,
    detectionDelayS: 38.45,
    checkpointPath: 'results/progressive_multimodal_corrected/checkpoints/model1_baseline_tcn.pt',
    liveInferenceAvailable: true,
    notes: 'Primary benchmark model. 4-layer dilated causal convolutions with residual blocks. Verified zero target leakage and strict trial disjointness.'
  },
  model2_bio_tcn: {
    id: 'model2_bio_tcn',
    name: 'Model 2: Bioimpedance-Augmented TCN',
    shortName: 'Model 2 (TCN + Bio)',
    category: 'multimodal',
    status: 'Bioimpedance Fusion',
    description: 'Dual-branch neural fusion combining Causal TCN temporal stream with multi-frequency bioimpedance encoder (20–500 kHz).',
    inputModalities: ['Local Temperature', 'Local Pressure', 'Multi-Frequency Bioimpedance (28 features)'],
    totalParameters: 66084,
    cpuLatencyMs: 10.34,
    sizeKb: 258.1,
    macroF1: 0.9689,
    balancedAccuracy: 0.9622,
    confounderFpr: 0.0000,
    detectionDelayS: -35.05,
    checkpointPath: 'results/progressive_multimodal_corrected/checkpoints/model2_bio_tcn_simulated.pt',
    liveInferenceAvailable: true,
    notes: 'Bioimpedance generated via forward biophysical simulation (admittance change from extravasated volume). Completely rejects vasoconstriction cooling.'
  },
  model3_stress: {
    id: 'model3_stress',
    name: 'Model 3: Stress-Aware Multimodal Model',
    shortName: 'Model 3 (Stress-Aware)',
    category: 'exploratory',
    status: 'Dual-Stream Stress Aware',
    description: 'Triple-branch architecture fusing local transducers, bioimpedance, and a systemic autonomic stress representation (z_sys) trained on WESAD wearable data.',
    inputModalities: ['Local Temperature', 'Local Pressure', 'Bioimpedance', 'WESAD Autonomic Stress Vector (32-dim)'],
    totalParameters: 95820,
    cpuLatencyMs: 16.52,
    sizeKb: 374.2,
    macroF1: 0.9710,
    balancedAccuracy: 0.9650,
    confounderFpr: 0.0000,
    detectionDelayS: 32.10,
    checkpointPath: 'results/systemic_encoder_wesad.pt',
    liveInferenceAvailable: false,
    notes: 'Evaluates stress-confounded false alarm suppression. When sympathetic vasoconstriction causes cold skin without edema, the stress branch vetoes the false alert.'
  },
  model4_age: {
    id: 'model4_age',
    name: 'Model 4: Age-Aware Multimodal Model',
    shortName: 'Model 4 (Age-Aware)',
    category: 'exploratory',
    status: 'Age Architecture (Data Blocked)',
    description: 'Multi-branch architecture integrating modular patient age embedding with local temporal, bioimpedance, and stress encoders.',
    inputModalities: ['Local Temperature', 'Local Pressure', 'Bioimpedance', 'Stress Representation', 'Patient Age Embedding (16-dim)'],
    totalParameters: 99702,
    cpuLatencyMs: 22.34,
    sizeKb: 389.5,
    macroF1: null,
    balancedAccuracy: null,
    confounderFpr: null,
    detectionDelayS: null,
    checkpointPath: 'results/progressive_multimodal_corrected/checkpoints/model4_age_multi_architecture.pt',
    liveInferenceAvailable: false,
    notes: 'Architectural implementation verified and profiled. Marked Data Blocked because individual patient-linked age data are absent from public datasets, preventing unverified data fabrication.'
  },
  rf: {
    id: 'rf',
    name: 'Baseline 1: Random Forest Classifier',
    shortName: 'Random Forest',
    category: 'baseline',
    status: 'Trained Baseline',
    description: 'Ensemble of 100 decision trees trained on 10 statistical moment features (mean, std, min, max, slope) over 50-second sliding windows.',
    inputModalities: ['Local Temperature (Moments)', 'Local Pressure (Moments)'],
    totalParameters: 18450,
    cpuLatencyMs: 1.15,
    sizeKb: 742.0,
    macroF1: 0.8845,
    balancedAccuracy: 0.8620,
    confounderFpr: 0.0480,
    detectionDelayS: 42.00,
    checkpointPath: 'models/random_forest_iv.joblib',
    liveInferenceAvailable: true,
    notes: 'Standard machine learning tabular baseline with non-linear feature split boundaries. Susceptible to sudden thermal drift artifacts.'
  },
  logreg: {
    id: 'logreg',
    name: 'Baseline 2: Logistic Regression',
    shortName: 'Logistic Regression',
    category: 'baseline',
    status: 'Trained Baseline',
    description: 'Multinomial linear classifier over 10 normalized statistical moment features. Suitable for direct C-code deployment on resource-constrained microcontrollers.',
    inputModalities: ['Local Temperature (Moments)', 'Local Pressure (Moments)'],
    totalParameters: 44,
    cpuLatencyMs: 0.04,
    sizeKb: 2.1,
    macroF1: 0.7410,
    balancedAccuracy: 0.7225,
    confounderFpr: 0.1250,
    detectionDelayS: 55.00,
    checkpointPath: 'models/logistic_regression_iv.joblib',
    liveInferenceAvailable: true,
    notes: 'Ultra-lightweight edge linear baseline. Low computational footprint allows microsecond execution on ESP32, but has lower sensitivity to gradual extravasation.'
  }
};
