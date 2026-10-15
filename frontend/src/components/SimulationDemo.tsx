import React, { useState, useEffect, useRef } from 'react';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine
} from 'recharts';
import {
  Play, Pause, RotateCcw, AlertTriangle,
  Sliders, Activity, RefreshCw, Cpu, ShieldAlert
} from 'lucide-react';
import { MODEL_REGISTRY, type ModelSpecification } from '../modelRegistry';

export interface DataPoint {
  time_s: number;
  temperature_c: number;
  pressure_adc: number;
  bioimpedance_ohm?: number;
  bioimpedance_rel_drop?: number;
  ground_truth_class: number;
  ground_truth_name: string;
  pred_class: number | null;
  pred_name: string | null;
  probability: number;
  probs: number[];
  alert: boolean;
}

export interface SimulationResult {
  scenario: string;
  ground_truth: string;
  is_infiltration_ground_truth: boolean;
  duration_s: number;
  window_len_s: number;
  dt: number;
  model_type: string;
  model_name: string;
  model_loaded: boolean;
  data: DataPoint[];
  summary: {
    first_detection_time_s: number | null;
    onset_time_s: number | null;
    detection_delay_s: number | null;
    alarm_triggered: boolean;
    total_windows: number;
    max_probability: number;
  };
}

interface ScenarioConfig {
  name: string;
  id: string;
  description: string;
  baseTemp: number;
  tempChange: number;
  basePress: number;
  pressChange: number;
  duration: number;
  noiseLevel: 'normal' | 'high';
}

const SCENARIOS: Record<string, ScenarioConfig> = {
  normal: {
    name: '1. Normal Infusion',
    id: 'normal',
    description: 'Continuous IV delivery into patent lumen. Normal baseline temperature and tissue compartment pressure.',
    baseTemp: 33.0,
    tempChange: 0.0,
    basePress: 1160.0,
    pressChange: 0.0,
    duration: 120,
    noiseLevel: 'normal',
  },
  infiltration: {
    name: '2. Simulated Infiltration',
    id: 'infiltration',
    description: 'Catheter dislodgement causing cold saline extravasation into subcutaneous space. Rapid cooling, compartment pressure rise, and impedance drop.',
    baseTemp: 33.0,
    tempChange: -1.8,
    basePress: 1160.0,
    pressChange: 80.0,
    duration: 180,
    noiseLevel: 'normal',
  },
  stress: {
    name: '3. Stress Vasoconstriction',
    id: 'stress',
    description: 'Autonomic sympathetic cooling without fluid extravasation. Skin cools by >1.2°C, but tissue pressure and bioimpedance remain strictly at baseline.',
    baseTemp: 33.0,
    tempChange: -1.4,
    basePress: 1160.0,
    pressChange: 0.0,
    duration: 180,
    noiseLevel: 'normal',
  },
  noisy: {
    name: '4. Noisy Sensor Readings',
    id: 'noisy',
    description: 'Baseline infusion with elevated thermal noise and mechanical motion artifacts (patient arm shifting, sensor cable jostling).',
    baseTemp: 33.0,
    tempChange: 0.0,
    basePress: 1160.0,
    pressChange: 0.0,
    duration: 120,
    noiseLevel: 'high',
  },
};

export const SimulationDemo: React.FC = () => {
  // Scenario and parameter state
  const [selectedScenario, setSelectedScenario] = useState<string>('infiltration');
  const [baseTemp, setBaseTemp] = useState<number>(33.0);
  const [tempChange, setTempChange] = useState<number>(-1.8);
  const [basePress, setBasePress] = useState<number>(1160.0);
  const [pressChange, setPressChange] = useState<number>(80.0);
  const [duration, setDuration] = useState<number>(180);
  const [selectedModelId, setSelectedModelId] = useState<string>('model1_tcn');

  // Simulation execution state
  const [simulationData, setSimulationData] = useState<SimulationResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [serverError, setServerError] = useState<string | null>(null);
  const [serverOnline, setServerOnline] = useState<boolean | null>(null);

  // Playback state
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(2); // 1x, 2x, 5x
  const playbackTimer = useRef<number | null>(null);

  const activeModelSpec: ModelSpecification = MODEL_REGISTRY[selectedModelId] || MODEL_REGISTRY.model1_tcn;

  // Check server health on mount
  useEffect(() => {
    fetch('/api/health')
      .then(res => res.json())
      .then(data => {
        if (data.status === 'ok') {
          setServerOnline(true);
        }
      })
      .catch(() => {
        setServerOnline(false);
      });
  }, []);

  // Update parameters when user picks a scenario preset
  const handleSelectScenario = (key: string) => {
    setSelectedScenario(key);
    const cfg = SCENARIOS[key];
    if (cfg) {
      setBaseTemp(cfg.baseTemp);
      setTempChange(cfg.tempChange);
      setBasePress(cfg.basePress);
      setPressChange(cfg.pressChange);
      setDuration(cfg.duration);
    }
  };

  // Run simulation via API (with client-side fallback if server offline)
  const handleRunSimulation = async () => {
    setLoading(true);
    setServerError(null);
    setIsPlaying(false);
    setCurrentTime(0);

    const payload = {
      scenario: selectedScenario,
      duration_s: duration,
      base_temp: baseTemp,
      temp_change: tempChange,
      base_press: basePress,
      press_change: pressChange,
      noise_level: SCENARIOS[selectedScenario]?.noiseLevel || 'normal',
      model_type: selectedModelId,
    };

    try {
      const response = await fetch('/api/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }

      const result: SimulationResult = await response.json();
      setSimulationData(result);
      setServerOnline(true);
      setCurrentTime(result.data.length > 0 ? 1 : 0);
      setIsPlaying(true);
    } catch (err: any) {
      console.warn('API error, falling back to simulator-only client mode:', err);
      setServerOnline(false);
      setServerError(`Backend server not reachable at http://127.0.0.1:8000 (${err.message}). Showing simulator waveforms in offline fallback mode.`);

      const fallbackResult = generateClientSimulatorTrace(payload);
      setSimulationData(fallbackResult);
      setCurrentTime(1);
      setIsPlaying(true);
    } finally {
      setLoading(false);
    }
  };

  // Playback timer loop
  useEffect(() => {
    if (isPlaying && simulationData && simulationData.data.length > 0) {
      playbackTimer.current = window.setInterval(() => {
        setCurrentTime(prev => {
          const next = prev + 1;
          if (next >= simulationData.data.length) {
            setIsPlaying(false);
            return simulationData.data.length - 1;
          }
          return next;
        });
      }, 1000 / playbackSpeed);
    } else {
      if (playbackTimer.current) clearInterval(playbackTimer.current);
    }

    return () => {
      if (playbackTimer.current) clearInterval(playbackTimer.current);
    };
  }, [isPlaying, simulationData, playbackSpeed]);

  const handleReset = () => {
    setIsPlaying(false);
    setCurrentTime(0);
    setSimulationData(null);
    setServerError(null);
  };

  // Current point in playback
  const currentPoint: DataPoint | null =
    simulationData && simulationData.data.length > 0 && currentTime < simulationData.data.length
      ? simulationData.data[currentTime]
      : null;

  // Progressive data history up to current playback cursor
  const progressiveData =
    simulationData && simulationData.data.length > 0
      ? simulationData.data.slice(0, Math.max(1, currentTime + 1))
      : [];

  const getSeverityBadge = (cls: number | null | undefined, name: string | null | undefined) => {
    switch (cls) {
      case 0:
        return <span style={{ background: 'rgba(6, 182, 212, 0.15)', color: 'var(--accent-cyan)', border: '1px solid rgba(6, 182, 212, 0.4)', padding: '2px 8px', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 600 }}>Normal (None)</span>;
      case 1:
        return <span style={{ background: 'rgba(234, 179, 8, 0.15)', color: '#facc15', border: '1px solid rgba(234, 179, 8, 0.4)', padding: '2px 8px', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 600 }}>Early Infiltration</span>;
      case 2:
        return <span style={{ background: 'rgba(249, 115, 22, 0.15)', color: '#fb923c', border: '1px solid rgba(249, 115, 22, 0.4)', padding: '2px 8px', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 600 }}>Moderate Infiltration</span>;
      case 3:
        return <span style={{ background: 'rgba(239, 68, 68, 0.2)', color: '#f87171', border: '1px solid rgba(239, 68, 68, 0.5)', padding: '2px 8px', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 700 }}>Severe Infiltration</span>;
      default:
        return <span style={{ background: 'rgba(100, 116, 139, 0.2)', color: '#94a3b8', border: '1px solid rgba(100, 116, 139, 0.3)', padding: '2px 8px', borderRadius: '4px', fontSize: '0.75rem' }}>{name || 'Buffering Window'}</span>;
    }
  };

  return (
    <div style={{ width: '100%', maxWidth: '1360px', margin: '0 auto' }}>
      {/* Streamlined Top Banner */}
      <div style={{
        background: 'rgba(15, 23, 42, 0.45)',
        border: '1px solid var(--border-subtle)',
        borderRadius: '6px',
        padding: '10px 18px',
        marginBottom: '20px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '12px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Activity size={18} color="var(--accent-cyan)" />
          <span style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-primary)' }}>
            Physiological Extravasation Simulator
          </span>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)' }}>
            • Pennes bioheat, interstitial compliance &amp; bioimpedance forward dynamics
          </span>
        </div>

        <div style={{
          fontSize: '0.68rem',
          padding: '3px 8px',
          borderRadius: '4px',
          background: serverOnline ? 'rgba(74, 222, 128, 0.12)' : 'rgba(239, 68, 68, 0.12)',
          color: serverOnline ? '#4ade80' : '#f87171',
          border: `1px solid ${serverOnline ? 'rgba(74, 222, 128, 0.25)' : 'rgba(239, 68, 68, 0.25)'}`,
          fontWeight: 500
        }}>
          {serverOnline === null ? 'Checking Backend...' : serverOnline ? 'Inference Engine Active (:8000)' : 'Client Fallback'}
        </div>
      </div>

      {serverError && (
        <div style={{
          background: 'rgba(234, 179, 8, 0.1)',
          border: '1px solid rgba(234, 179, 8, 0.3)',
          borderRadius: '8px',
          padding: '10px 16px',
          marginBottom: '18px',
          fontSize: '0.78rem',
          color: '#fbbf24',
          display: 'flex',
          alignItems: 'center',
          gap: '10px'
        }}>
          <AlertTriangle size={16} />
          <div>{serverError}</div>
        </div>
      )}

      {/* Main Grid: Controls on Left (340px), Waveforms & Telemetry on Right */}
      <div style={{ display: 'grid', gridTemplateColumns: '340px 1fr', gap: '20px', alignItems: 'start' }}>
        
        {/* Left Column: Model Progression Selector & Parameters */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          
          {/* 1. Model Architecture Selector */}
          <div className="glass-card" style={{ padding: '18px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
              <div className="text-caption" style={{ color: 'var(--accent-cyan)' }}>1. Model Architecture</div>
              <Cpu size={14} color="var(--accent-cyan)" />
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '16px' }}>
              {Object.keys(MODEL_REGISTRY).map(key => {
                const spec = MODEL_REGISTRY[key];
                const isSelected = selectedModelId === key;
                return (
                  <button
                    key={key}
                    onClick={() => setSelectedModelId(key)}
                    style={{
                      background: isSelected ? 'rgba(6, 182, 212, 0.12)' : 'rgba(255, 255, 255, 0.02)',
                      border: `1px solid ${isSelected ? 'var(--accent-cyan)' : 'var(--border-subtle)'}`,
                      borderRadius: '6px',
                      padding: '10px 12px',
                      textAlign: 'left',
                      cursor: 'pointer',
                      transition: 'all 0.15s ease',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                      <span style={{ fontSize: '0.8rem', fontWeight: 600, color: isSelected ? 'var(--accent-cyan)' : 'var(--text-primary)' }}>
                        {spec.shortName}
                      </span>
                      <span style={{ fontSize: '0.65rem', color: isSelected ? 'var(--accent-cyan)' : 'var(--text-tertiary)', fontFamily: 'JetBrains Mono, monospace' }}>
                        {spec.totalParameters.toLocaleString()}p
                      </span>
                    </div>
                    <div style={{ fontSize: '0.68rem', color: 'var(--text-secondary)', marginTop: '2px', lineHeight: 1.3 }}>
                      {spec.status}
                    </div>
                  </button>
                );
              })}
            </div>

            {/* Selected Model Profile Card */}
            <div style={{
              background: 'rgba(15, 23, 42, 0.6)',
              border: '1px solid var(--border-subtle)',
              borderRadius: '6px',
              padding: '12px',
              fontSize: '0.72rem'
            }}>
              <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: '4px' }}>
                {activeModelSpec.name}
              </div>
              <div style={{ color: 'var(--text-secondary)', marginBottom: '8px', lineHeight: 1.4 }}>
                {activeModelSpec.description}
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', borderTop: '1px solid var(--border-subtle)', paddingTop: '8px' }}>
                <div>
                  <span style={{ color: 'var(--text-tertiary)' }}>CPU Latency: </span>
                  <span className="text-numeric" style={{ color: 'var(--accent-cyan)' }}>{activeModelSpec.cpuLatencyMs} ms</span>
                </div>
                <div>
                  <span style={{ color: 'var(--text-tertiary)' }}>Macro-F1: </span>
                  <span className="text-numeric" style={{ color: activeModelSpec.macroF1 ? '#4ade80' : '#94a3b8' }}>
                    {activeModelSpec.macroF1 ? activeModelSpec.macroF1.toFixed(4) : 'N/A'}
                  </span>
                </div>
                <div>
                  <span style={{ color: 'var(--text-tertiary)' }}>Model Size: </span>
                  <span className="text-numeric">{activeModelSpec.sizeKb} KB</span>
                </div>
                <div>
                  <span style={{ color: 'var(--text-tertiary)' }}>Confounder FPR: </span>
                  <span className="text-numeric" style={{ color: activeModelSpec.confounderFpr === 0 ? '#4ade80' : 'var(--text-secondary)' }}>
                    {activeModelSpec.confounderFpr !== null ? `${(activeModelSpec.confounderFpr * 100).toFixed(1)}%` : 'N/A'}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* 2. Scenario Presets */}
          <div className="glass-card" style={{ padding: '18px' }}>
            <div className="text-caption" style={{ marginBottom: '10px', color: 'var(--accent-cyan)' }}>
              2. Select Scenario
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {Object.keys(SCENARIOS).map(key => {
                const sc = SCENARIOS[key];
                const isSelected = selectedScenario === key;
                return (
                  <button
                    key={key}
                    onClick={() => handleSelectScenario(key)}
                    style={{
                      background: isSelected ? 'rgba(6, 182, 212, 0.12)' : 'rgba(255, 255, 255, 0.02)',
                      border: `1px solid ${isSelected ? 'var(--accent-cyan)' : 'var(--border-subtle)'}`,
                      borderRadius: '6px',
                      padding: '10px 12px',
                      textAlign: 'left',
                      cursor: 'pointer',
                      transition: 'all 0.15s',
                    }}
                  >
                    <div style={{ fontSize: '0.8rem', fontWeight: 600, color: isSelected ? 'var(--accent-cyan)' : 'var(--text-primary)' }}>
                      {sc.name}
                    </div>
                    <div style={{ fontSize: '0.68rem', color: 'var(--text-secondary)', marginTop: '2px', lineHeight: 1.3 }}>
                      {sc.description}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* 3. Physical Parameters & Controls */}
          <div className="glass-card" style={{ padding: '18px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
              <div className="text-caption" style={{ color: 'var(--accent-cyan)' }}>3. Physical Controls</div>
              <Sliders size={14} color="var(--text-secondary)" />
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {/* Baseline Temp */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', marginBottom: '3px' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Baseline Skin Temperature:</span>
                  <span className="text-numeric" style={{ color: 'var(--accent-cyan)' }}>{baseTemp.toFixed(1)} °C</span>
                </div>
                <input
                  type="range" min="30.0" max="36.0" step="0.1" value={baseTemp}
                  onChange={e => setBaseTemp(parseFloat(e.target.value))}
                  style={{ width: '100%', accentColor: 'var(--accent-cyan)' }}
                />
              </div>

              {/* Temp Delta */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', marginBottom: '3px' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Temperature Delta (ΔT):</span>
                  <span className="text-numeric" style={{ color: tempChange < 0 ? '#67e8f9' : 'var(--text-primary)' }}>
                    {tempChange > 0 ? `+${tempChange.toFixed(1)}` : tempChange.toFixed(1)} °C
                  </span>
                </div>
                <input
                  type="range" min="-3.5" max="0.5" step="0.1" value={tempChange}
                  onChange={e => setTempChange(parseFloat(e.target.value))}
                  style={{ width: '100%', accentColor: 'var(--accent-cyan)' }}
                />
              </div>

              {/* Baseline Pressure */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', marginBottom: '3px' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Baseline Tissue Pressure:</span>
                  <span className="text-numeric" style={{ color: 'var(--accent-cyan)' }}>{basePress.toFixed(0)} ADC</span>
                </div>
                <input
                  type="range" min="1000" max="1300" step="10" value={basePress}
                  onChange={e => setBasePress(parseFloat(e.target.value))}
                  style={{ width: '100%', accentColor: 'var(--accent-cyan)' }}
                />
              </div>

              {/* Pressure Delta */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', marginBottom: '3px' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Pressure Swelling Delta (ΔP):</span>
                  <span className="text-numeric" style={{ color: pressChange > 0 ? '#f87171' : 'var(--text-primary)' }}>
                    +{pressChange.toFixed(0)} ADC
                  </span>
                </div>
                <input
                  type="range" min="0" max="250" step="5" value={pressChange}
                  onChange={e => setPressChange(parseFloat(e.target.value))}
                  style={{ width: '100%', accentColor: 'var(--accent-cyan)' }}
                />
              </div>

              {/* Duration */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', marginBottom: '3px' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Scenario Duration:</span>
                  <span className="text-numeric">{duration}s</span>
                </div>
                <input
                  type="range" min="60" max="300" step="30" value={duration}
                  onChange={e => setDuration(parseInt(e.target.value))}
                  style={{ width: '100%', accentColor: 'var(--accent-cyan)' }}
                />
              </div>
            </div>

            {/* Run & Reset Buttons */}
            <div style={{ display: 'flex', gap: '8px', marginTop: '16px' }}>
              <button
                className="btn active"
                onClick={handleRunSimulation}
                disabled={loading}
                style={{
                  flex: 2,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '8px',
                  background: 'var(--accent-cyan)',
                  color: '#000',
                  fontWeight: 600,
                  padding: '10px 14px',
                  fontSize: '0.8rem'
                }}
              >
                {loading ? <RefreshCw size={15} className="spin" /> : <Play size={15} />}
                Run Simulation
              </button>
              <button
                className="btn"
                onClick={handleReset}
                disabled={loading || !simulationData}
                style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px', padding: '10px 12px', fontSize: '0.8rem' }}
              >
                <RotateCcw size={14} />
                Reset
              </button>
            </div>
          </div>

        </div>

        {/* Right Column: Inference Monitor, Playback Controls, & Multi-Channel Waveforms */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          
          {/* Top Panel: Ground Truth vs Real-Time Prediction Card */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
            
            {/* Ground Truth Status */}
            <div className="glass-card" style={{ padding: '16px' }}>
              <div className="text-caption" style={{ marginBottom: '8px', color: 'var(--text-secondary)' }}>
                Ground Truth Scenario
              </div>
              <div style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '8px', lineHeight: 1.4 }}>
                {simulationData?.ground_truth || SCENARIOS[selectedScenario].description}
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)' }}>Current State:</span>
                {currentPoint ? getSeverityBadge(currentPoint.ground_truth_class, currentPoint.ground_truth_name) : <span style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)' }}>Idle</span>}
              </div>
            </div>

            {/* Real-Time Model Prediction */}
            <div className="glass-card" style={{ padding: '16px', position: 'relative' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                <div className="text-caption" style={{ color: 'var(--accent-cyan)' }}>Real-Time Model Inference</div>
                <div style={{ fontSize: '0.68rem', color: 'var(--text-tertiary)' }}>
                  {simulationData?.model_name || activeModelSpec.name}
                </div>
              </div>

              {simulationData ? (
                <div>
                  <div style={{ display: 'flex', alignItems: 'baseline', gap: '12px', marginBottom: '8px' }}>
                    <div className="text-numeric" style={{
                      fontSize: '1.8rem',
                      fontWeight: 600,
                      color: currentPoint?.alert ? 'var(--accent-red)' : (currentPoint && currentPoint.probability > 0.4) ? '#f59e0b' : 'var(--accent-cyan)'
                    }}>
                      {currentPoint ? (currentPoint.probability * 100).toFixed(1) : '0.0'}%
                    </div>
                    <div>
                      {currentPoint ? getSeverityBadge(currentPoint.pred_class, currentPoint.pred_name) : null}
                    </div>
                  </div>

                  {currentPoint?.alert && (
                    <div style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '6px',
                      background: 'rgba(239, 68, 68, 0.2)',
                      border: '1px solid rgba(239, 68, 68, 0.5)',
                      borderRadius: '4px',
                      padding: '3px 8px',
                      fontSize: '0.72rem',
                      color: '#f87171',
                      fontWeight: 600,
                      marginBottom: '8px'
                    }}>
                      <ShieldAlert size={14} /> SUSTAINED EXTRAVASATION ALARM
                    </div>
                  )}

                  {/* Class Probabilities Bar */}
                  {currentPoint && currentPoint.probs && (
                    <div style={{ marginTop: '4px' }}>
                      <div style={{ display: 'flex', height: '6px', borderRadius: '3px', overflow: 'hidden', background: 'rgba(255,255,255,0.05)' }}>
                        <div style={{ width: `${currentPoint.probs[0] * 100}%`, background: 'var(--accent-cyan)' }} title={`None: ${(currentPoint.probs[0]*100).toFixed(1)}%`} />
                        <div style={{ width: `${currentPoint.probs[1] * 100}%`, background: '#facc15' }} title={`Early: ${(currentPoint.probs[1]*100).toFixed(1)}%`} />
                        <div style={{ width: `${currentPoint.probs[2] * 100}%`, background: '#fb923c' }} title={`Moderate: ${(currentPoint.probs[2]*100).toFixed(1)}%`} />
                        <div style={{ width: `${currentPoint.probs[3] * 100}%`, background: '#f87171' }} title={`Severe: ${(currentPoint.probs[3]*100).toFixed(1)}%`} />
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', color: 'var(--text-tertiary)', marginTop: '4px' }}>
                        <span>None</span>
                        <span>Early</span>
                        <span>Moderate</span>
                        <span>Severe</span>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', padding: '16px 0' }}>
                  Click &ldquo;Run Simulation&rdquo; to execute time series and trigger sliding-window model predictions.
                </div>
              )}
            </div>

          </div>

          {/* Waveforms Panel */}
          <div className="glass-card" style={{ padding: '18px' }}>
            
            {/* Playback Controls & Time Cursor Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px', flexWrap: 'wrap', gap: '10px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <button
                  className="btn"
                  onClick={() => setIsPlaying(!isPlaying)}
                  disabled={!simulationData}
                  style={{ padding: '6px 14px', fontSize: '0.75rem', display: 'flex', alignItems: 'center', gap: '6px' }}
                >
                  {isPlaying ? <Pause size={13} /> : <Play size={13} />}
                  {isPlaying ? 'Pause' : 'Play'}
                </button>
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                  {[1, 2, 5].map(spd => (
                    <button
                      key={spd}
                      onClick={() => setPlaybackSpeed(spd)}
                      style={{
                        background: playbackSpeed === spd ? 'rgba(6, 182, 212, 0.2)' : 'transparent',
                        border: `1px solid ${playbackSpeed === spd ? 'var(--accent-cyan)' : 'var(--border-subtle)'}`,
                        color: playbackSpeed === spd ? 'var(--accent-cyan)' : 'var(--text-secondary)',
                        borderRadius: '4px',
                        padding: '3px 8px',
                        fontSize: '0.7rem',
                        cursor: 'pointer'
                      }}
                    >
                      {spd}x
                    </button>
                  ))}
                </div>
              </div>

              {simulationData && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <span className="text-numeric" style={{ fontSize: '0.75rem', color: 'var(--accent-cyan)' }}>
                    Cursor: t = {currentTime}s / {simulationData.duration_s}s
                  </span>
                  {simulationData.summary.first_detection_time_s !== null && (
                    <span style={{ fontSize: '0.7rem', color: '#f87171', background: 'rgba(239, 68, 68, 0.1)', padding: '2px 8px', borderRadius: '4px' }}>
                      Alert at t = {simulationData.summary.first_detection_time_s}s
                    </span>
                  )}
                </div>
              )}
            </div>

            {/* Time Scrubber Slider */}
            {simulationData && (
              <div style={{ marginBottom: '16px' }}>
                <input
                  type="range"
                  min="0"
                  max={simulationData.data.length - 1}
                  value={currentTime}
                  onChange={e => {
                    setIsPlaying(false);
                    setCurrentTime(parseInt(e.target.value));
                  }}
                  style={{ width: '100%', accentColor: 'var(--accent-cyan)' }}
                />
              </div>
            )}

            {/* Waveform 1: Temperature */}
            <div style={{ marginBottom: '18px' }}>
              <div className="data-row" style={{ marginBottom: '4px' }}>
                <span style={{ fontSize: '0.72rem', color: 'var(--accent-cyan)', fontWeight: 500 }}>
                  ● Local Temperature (DS18B20 Bead, °C)
                </span>
                <span className="text-numeric" style={{ fontSize: '0.85rem', color: 'var(--accent-cyan)' }}>
                  {currentPoint ? `${currentPoint.temperature_c.toFixed(2)} °C` : '--'}
                </span>
              </div>
              <div style={{ height: '100px', width: '100%' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={simulationData ? progressiveData : []}>
                    <XAxis dataKey="time_s" hide domain={[0, duration]} />
                    <YAxis domain={['auto', 'auto']} stroke="var(--text-tertiary)" fontSize={10} width={40} />
                    <Tooltip contentStyle={{ background: '#0a0d14', border: '1px solid var(--border-active)', fontSize: '0.75rem' }} />
                    <Line type="monotone" dataKey="temperature_c" stroke="var(--accent-cyan)" strokeWidth={2} dot={false} isAnimationActive={false} />
                    {simulationData && <ReferenceLine x={currentTime} stroke="#ffffff" strokeDasharray="3 3" />}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Waveform 2: Pressure */}
            <div style={{ marginBottom: selectedModelId === 'model2_bio_tcn' || (simulationData?.data?.[0]?.bioimpedance_ohm !== undefined) ? '18px' : '0' }}>
              <div className="data-row" style={{ marginBottom: '4px' }}>
                <span style={{ fontSize: '0.72rem', color: '#c084fc', fontWeight: 500 }}>
                  ● Interstitial Compartment Pressure (FSR 402, ADC Counts)
                </span>
                <span className="text-numeric" style={{ fontSize: '0.85rem', color: '#c084fc' }}>
                  {currentPoint ? `${currentPoint.pressure_adc.toFixed(1)} ADC` : '--'}
                </span>
              </div>
              <div style={{ height: '100px', width: '100%' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={simulationData ? progressiveData : []}>
                    <XAxis dataKey="time_s" stroke="var(--text-tertiary)" fontSize={10} domain={[0, duration]} />
                    <YAxis domain={['auto', 'auto']} stroke="var(--text-tertiary)" fontSize={10} width={40} />
                    <Tooltip contentStyle={{ background: '#0a0d14', border: '1px solid var(--border-active)', fontSize: '0.75rem' }} />
                    <Line type="monotone" dataKey="pressure_adc" stroke="#c084fc" strokeWidth={2} dot={false} isAnimationActive={false} />
                    {simulationData && <ReferenceLine x={currentTime} stroke="#ffffff" strokeDasharray="3 3" />}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Waveform 3: Bioimpedance (when Model 2 or multi-frequency data available) */}
            {(selectedModelId === 'model2_bio_tcn' || (simulationData?.data?.[0]?.bioimpedance_ohm !== undefined)) && (
              <div>
                <div className="data-row" style={{ marginBottom: '4px' }}>
                  <span style={{ fontSize: '0.72rem', color: '#4ade80', fontWeight: 500 }}>
                    ● Multi-Frequency Bioimpedance (50 kHz Admittance Channel, Ω)
                  </span>
                  <span className="text-numeric" style={{ fontSize: '0.85rem', color: '#4ade80' }}>
                    {currentPoint?.bioimpedance_ohm ? `${currentPoint.bioimpedance_ohm.toFixed(1)} Ω (${currentPoint.bioimpedance_rel_drop?.toFixed(1)}%)` : '--'}
                  </span>
                </div>
                <div style={{ height: '100px', width: '100%' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={simulationData ? progressiveData : []}>
                      <XAxis dataKey="time_s" stroke="var(--text-tertiary)" fontSize={10} domain={[0, duration]} />
                      <YAxis domain={['auto', 'auto']} stroke="var(--text-tertiary)" fontSize={10} width={40} />
                      <Tooltip contentStyle={{ background: '#0a0d14', border: '1px solid var(--border-active)', fontSize: '0.75rem' }} />
                      <Line type="monotone" dataKey="bioimpedance_ohm" stroke="#4ade80" strokeWidth={2} dot={false} isAnimationActive={false} />
                      {simulationData && <ReferenceLine x={currentTime} stroke="#ffffff" strokeDasharray="3 3" />}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

          </div>

        </div>

      </div>
    </div>
  );
};

// --- Client-Side Simulator Fallback ---
function generateClientSimulatorTrace(params: any): SimulationResult {
  const duration = params.duration_s || 120;
  const baseT = params.base_temp || 33.0;
  const baseP = params.base_press || 1160.0;
  const deltaT = params.temp_change || 0.0;
  const deltaP = params.press_change || 0.0;
  const isNoisy = params.noise_level === 'high';
  const isStress = params.scenario === 'stress';
  const isInfil = params.scenario === 'infiltration';

  const data: DataPoint[] = [];
  const onset = Math.floor(duration * 0.2);

  for (let t = 0; t < duration; t++) {
    const progress = t > onset ? (t - onset) / (duration - onset) : 0;
    
    // Thermal curve
    let temp = baseT;
    if (isInfil || isStress) {
      temp = baseT + deltaT * Math.min(1.0, progress * 1.5);
    }
    temp += (Math.random() - 0.5) * (isNoisy ? 0.45 : 0.15);

    // Pressure curve
    let press = baseP;
    if (isInfil) {
      press = baseP + deltaP * Math.min(1.0, progress * 1.8);
    }
    press += (Math.random() - 0.5) * (isNoisy ? 15.0 : 4.0);

    // Forward bioimpedance drop (saline conductivity expansion)
    let bioOhm = 580.0;
    let bioDrop = 0.0;
    if (isInfil && t > onset) {
      bioDrop = -Math.min(32.0, progress * 28.0);
      bioOhm = 580.0 * (1.0 + bioDrop / 100.0);
    }
    bioOhm += (Math.random() - 0.5) * (isNoisy ? 6.0 : 1.5);

    // Ground truth
    let gt = 0;
    let gtName = 'None';
    if (isInfil && t > onset) {
      if (progress > 0.6) { gt = 2; gtName = 'Moderate'; }
      else { gt = 1; gtName = 'Early'; }
    }

    data.push({
      time_s: t,
      temperature_c: parseFloat(temp.toFixed(2)),
      pressure_adc: parseFloat(press.toFixed(1)),
      bioimpedance_ohm: parseFloat(bioOhm.toFixed(1)),
      bioimpedance_rel_drop: parseFloat(bioDrop.toFixed(2)),
      ground_truth_class: gt,
      ground_truth_name: gtName,
      pred_class: null,
      pred_name: null,
      probability: 0.0,
      probs: [1.0, 0.0, 0.0, 0.0],
      alert: false,
    });
  }

  return {
    scenario: params.scenario,
    ground_truth: isInfil ? `Simulated Infiltration (Onset at t = ${onset}s)` : (isStress ? 'Stress Vasoconstriction (No fluid leakage)' : 'Normal Infusion'),
    is_infiltration_ground_truth: isInfil,
    duration_s: duration,
    window_len_s: 50,
    dt: 1.0,
    model_type: params.model_type || 'model1_tcn',
    model_name: 'Simulator-Only Mode (Offline Fallback — Predictions Not Fabricated)',
    model_loaded: false,
    data,
    summary: {
      first_detection_time_s: null,
      onset_time_s: isInfil ? onset : null,
      detection_delay_s: null,
      alarm_triggered: false,
      total_windows: Math.max(0, duration - 50),
      max_probability: 0.0,
    }
  };
}

export default SimulationDemo;
