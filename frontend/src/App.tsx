import { useState, useEffect, useRef, useMemo, Suspense } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Html, useGLTF } from '@react-three/drei';
import * as THREE from 'three';
import { 
  LineChart, Line, YAxis, Tooltip, ResponsiveContainer 
} from 'recharts';
import { Activity, X } from 'lucide-react';

// --- Types ---
interface SensorData {
  time_ms: number;
  temperature_c: number;
  pressure_adc: number;
  condition: string;
  probability: number;
  severity_class: number;
  alert: number;
}

interface RobustnessResult {
  condition: string;
  logreg_f1: number;
  rf_f1: number;
}

// --- 3D Holographic Arm & IV Setup ---
const ArmGLTF = () => {
  // This expects arm.glb in frontend/public/.
  // We apply a custom holographic material to its meshes.
  const { scene } = useGLTF('/arm.glb');
  
  const holoMaterial = useMemo(() => new THREE.MeshStandardMaterial({
    color: '#06b6d4',
    emissive: '#001133',
    emissiveIntensity: 0.3,
    roughness: 0.4,
    metalness: 0.5,
  }), []);

  useEffect(() => {
    scene.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        (child as THREE.Mesh).material = holoMaterial;
      }
    });
  }, [scene, holoMaterial]);

  return <primitive object={scene} scale={[0.08, 0.08, 0.08]} />;
};

// Fallback if no GLTF provided yet
const FallbackArm = () => {
  const holoMat = new THREE.MeshStandardMaterial({
    color: '#06b6d4', emissive: '#002244', transparent: true, opacity: 0.6
  });
  return (
    <mesh position={[0,0,0]} rotation={[0,0,-0.2]}>
      <cylinderGeometry args={[0.6, 0.4, 4, 32]} />
      <primitive object={holoMat} attach="material" />
    </mesh>
  );
};

// The hardware itself, un-rotated, centered at its own local origin
const IVHardware = ({ severity, isFocused, onFocus, onInspect }: any) => {
  const [hovered, setHovered] = useState(false);
  const hotspotColor = severity === 0 ? '#06b6d4' : severity === 1 ? '#eab308' : severity === 2 ? '#f97316' : '#ef4444';
  const hotspotRef = useRef<THREE.Mesh>(null);
  const swellingRef = useRef<THREE.Mesh>(null);
  const leakageRef = useRef<THREE.Mesh>(null);
  
  useFrame(({ clock }) => {
    if (hotspotRef.current) {
      const pulse = 1 + Math.sin(clock.elapsedTime * 4) * 0.15 * (severity > 0 ? 1 : 0);
      hotspotRef.current.scale.setScalar(pulse);
    }
    
    // Smoothly animate swelling and leakage based on severity
    if (swellingRef.current && leakageRef.current) {
      const targetScale = severity === 0 ? 0.001 : 1 + (severity * 0.3);
      const currentScale = swellingRef.current.scale.x;
      const newScale = THREE.MathUtils.lerp(currentScale, targetScale, 0.05);
      
      swellingRef.current.scale.set(newScale, newScale, newScale * 0.3); // Flattened bump
      leakageRef.current.scale.set(newScale, newScale, 1);
      
      // Pulse opacity based on severity, but keep it subtle
      const leakMat = leakageRef.current.material as THREE.MeshPhysicalMaterial;
      const swellMat = swellingRef.current.material as THREE.MeshPhysicalMaterial;
      
      const targetOpacity = severity > 0 ? 0.3 + (Math.sin(clock.elapsedTime * 2) * 0.05) : 0;
      leakMat.opacity = THREE.MathUtils.lerp(leakMat.opacity, targetOpacity, 0.1);
      swellMat.opacity = THREE.MathUtils.lerp(swellMat.opacity, severity > 0 ? 0.5 : 0, 0.1);
    }
  });

  return (
    <group>
      {/* A. SWELLING & LEAKAGE EFFECTS (Placed under the tape at insertion site) */}
      <group position={[0, 0.5, -0.02]}>
        {/* The physical bump (subcutaneous fluid) */}
        <mesh ref={swellingRef}>
          <sphereGeometry args={[0.15, 32, 32]} />
          <meshPhysicalMaterial 
            color={severity >= 2 ? '#ef4444' : '#0ea5e9'} 
            emissive={severity >= 2 ? '#991b1b' : '#0284c7'}
            emissiveIntensity={0.6}
            transmission={0.4} 
            roughness={0.4} 
            transparent 
            opacity={0} 
          />
        </mesh>
        
        {/* The surface leakage/pooling effect (Subtle circular aura) */}
        <mesh ref={leakageRef} position={[0, 0, 0.01]}>
          <circleGeometry args={[0.25, 32]} />
          <meshPhysicalMaterial 
            color="#ef4444" 
            transmission={0.9}
            roughness={0.2}
            transparent 
            opacity={0}
            depthWrite={false}
          />
        </mesh>
      </group>

      {/* 1. Medical Adhesive Patch (Tape) - flush with skin */}
      <mesh position={[0, 0, 0.0]}>
        {/* Horizontal adhesive patch */}
        <boxGeometry args={[1.4, 0.9, 0.01]} />
        <meshPhysicalMaterial color="#ffffff" transmission={0.2} opacity={0.6} transparent side={THREE.DoubleSide} roughness={0.8} />
      </mesh>

      {/* 2. IV Catheter Tube - entering at the top into the skin */}
      <mesh position={[0, 0.75, 0.02]} rotation={[0.5, 0, 0]}>
        <cylinderGeometry args={[0.015, 0.015, 0.8, 8]} />
        <meshPhysicalMaterial color="#e0f2fe" transmission={0.9} opacity={0.7} transparent />
      </mesh>
      
      {/* 3. Compact ESP32 Module (distinct dark color) */}
      <mesh position={[0, -0.1, 0.02]}>
        {/* Thin, flat, horizontal sensor module sitting on the tape */}
        <boxGeometry args={[0.8, 0.5, 0.05]} />
        <meshStandardMaterial color="#0f172a" metalness={0.8} roughness={0.2} />
      </mesh>

      {/* ESP32 Chip Detail (Silver) */}
      <mesh position={[0, 0.1, 0.05]}>
        <boxGeometry args={[0.14, 0.14, 0.02]} />
        <meshStandardMaterial color="#94a3b8" metalness={0.9} roughness={0.1} />
      </mesh>

      {/* 4. Sensors & Status Indicator */}
      {/* Temperature sensor bead */}
      <mesh position={[-0.15, 0.35, 0.02]}>
        <sphereGeometry args={[0.04, 16, 16]} />
        <meshStandardMaterial color="#cbd5e1" metalness={0.9} />
      </mesh>
      {/* Pressure sensor disc */}
      <mesh position={[0.15, 0.35, 0.01]} rotation={[Math.PI/2, 0, 0]}>
        <cylinderGeometry args={[0.06, 0.06, 0.02, 16]} />
        <meshStandardMaterial color="#334155" />
      </mesh>

      {/* LED Indicator */}
      <mesh position={[0, -0.3, 0.05]}>
        <sphereGeometry args={[0.02, 16, 16]} />
        <meshBasicMaterial color={hotspotColor} />
      </mesh>

      {/* 5. Thin neatly routed wiring */}
      <mesh position={[0, 0.5, 0.01]}>
        <cylinderGeometry args={[0.008, 0.008, 0.6, 8]} />
        <meshStandardMaterial color="#ef4444" />
      </mesh>

      {/* 6. Infiltration Hotspot & Hitbox - placed exactly at catheter insertion point */}
      <mesh 
        position={[0, 0.5, 0.02]}
        onPointerOver={(e) => { e.stopPropagation(); setHovered(true); document.body.style.cursor = 'pointer'; }}
        onPointerOut={(e) => { e.stopPropagation(); setHovered(false); document.body.style.cursor = 'auto'; }}
        onClick={(e) => { 
          e.stopPropagation(); 
          if (!isFocused) onFocus(); 
        }}
        onDoubleClick={(e) => {
          e.stopPropagation();
          if (isFocused) onInspect();
        }}
      >
        {/* Invisible Hitbox (Large) */}
        <sphereGeometry args={[0.3, 16, 16]} />
        <meshBasicMaterial transparent opacity={0} depthWrite={false} />

        {/* Visible glowing dot at insertion site */}
        <mesh ref={hotspotRef}>
          <sphereGeometry args={[0.05, 16, 16]} />
          <meshBasicMaterial color={hotspotColor} transparent opacity={0.8} />
          <mesh>
            <sphereGeometry args={[0.02, 16, 16]} />
            <meshBasicMaterial color="#ffffff" />
          </mesh>
        </mesh>

        {/* Hover Information Panel */}
        {hovered && !isFocused && (
          <Html position={[0.4, 0, 0]} center style={{ pointerEvents: 'none', zIndex: 10 }}>
            <div className="hotspot-panel">
              <div className="text-caption" style={{ marginBottom: '12px', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '8px' }}>
                Infiltration Site
              </div>
              <div className="data-row">
                <span className="text-secondary">Probability</span>
                <span className="text-numeric" style={{ color: hotspotColor, fontWeight: 500 }}>{severity > 0 ? '> 70%' : '< 10%'}</span>
              </div>
              <div className="data-row">
                <span className="text-secondary">Severity</span>
                <span style={{ color: hotspotColor, fontWeight: 500 }}>{severity > 0 ? `Level ${severity}` : 'Normal'}</span>
              </div>
              <div className="text-caption" style={{ marginTop: '16px', color: 'var(--accent-cyan)' }}>
                Click to Focus
              </div>
            </div>
          </Html>
        )}
      </mesh>
    </group>
  );
};

// --- Camera Controller ---
const CameraController = ({ isFocused }: { isFocused: boolean }) => {
  const { camera } = useThree();
  const controlsRef = useRef<any>(null);
  const [animating, setAnimating] = useState(false);

  // Default state matches Reference 2: framing the arm clearly
  const targetPos = useMemo(() => isFocused ? new THREE.Vector3(2.8, -3.4, 8.0) : new THREE.Vector3(0, -1, 16), [isFocused]);
  // Look directly at the IVSiteAnchor world coordinates
  const targetLookAt = useMemo(() => isFocused ? new THREE.Vector3(2.8, -3.4, 1.1) : new THREE.Vector3(1, -2, 0), [isFocused]);

  useEffect(() => {
    setAnimating(true);
  }, [isFocused]);

  useFrame(() => {
    if (animating && controlsRef.current) {
      camera.position.lerp(targetPos, 0.05);
      controlsRef.current.target.lerp(targetLookAt, 0.05);
      controlsRef.current.update();
      
      if (camera.position.distanceTo(targetPos) < 0.1) {
        setAnimating(false);
      }
    }
  });

  return (
    <OrbitControls 
      ref={controlsRef} 
      enablePan={true} 
      maxPolarAngle={Math.PI / 1.2} 
      minDistance={1.5} 
      maxDistance={25} 
      onStart={() => setAnimating(false)}
    />
  );
};


// --- Main App ---
function App() {
  const [dataHistory, setDataHistory] = useState<SensorData[]>([]);
  const [currentData, setCurrentData] = useState<SensorData | null>(null);
  
  const [isSimulating, setIsSimulating] = useState(false);
  const [condition, setCondition] = useState('normal');
  const [robustness, setRobustness] = useState<RobustnessResult[]>([]);
  
  const [serialConnected, setSerialConnected] = useState(false);
  const [logs, setLogs] = useState<{time:string, msg:string}[]>([]);
  
  const [isFocused, setIsFocused] = useState(false);
  const [isInspecting, setIsInspecting] = useState(false);
  
  const [activeTab, setActiveTab] = useState('monitor');

  const simulationTimer = useRef<number | null>(null);
  const replayTrace = useRef<SensorData[]>([]);
  const replayIdx = useRef(0);

  const addLog = (msg: string) => {
    setLogs(prev => [{ time: new Date().toLocaleTimeString(), msg }, ...prev].slice(0, 50));
  };

  useEffect(() => {
    const handleScroll = () => {
      const sections = ['monitor', 'analysis', 'hardware'];
      let current = 'monitor';
      for (const section of sections) {
        const el = document.getElementById(section);
        if (el) {
          const rect = el.getBoundingClientRect();
          if (rect.top <= window.innerHeight / 2) {
            current = section;
          }
        }
      }
      setActiveTab(current);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    fetch('/robustness_results.json')
      .then(r => r.json())
      .then(data => setRobustness(data))
      .catch(e => console.error("Failed to load robustness", e));

    fetch('/replay_trace.csv')
      .then(r => r.text())
      .then(text => {
        const lines = text.split('\n').slice(1);
        const trace = lines.map(line => {
          const [t, temp, p, c, prob, sev, alert] = line.split(',');
          if (!t) return null;
          return {
            time_ms: parseInt(t),
            temperature_c: parseFloat(temp),
            pressure_adc: parseFloat(p),
            condition: c,
            probability: parseFloat(prob),
            severity_class: parseInt(sev),
            alert: parseInt(alert)
          };
        }).filter(Boolean) as SensorData[];
        replayTrace.current = trace;
        addLog(`Loaded trace with ${trace.length} rows`);
      });
  }, []);

  useEffect(() => {
    if (isSimulating && !serialConnected) {
      simulationTimer.current = window.setInterval(() => {
        if (replayTrace.current.length === 0) return;
        let nextData = replayTrace.current[replayIdx.current];
        setCurrentData(nextData);
        setDataHistory(prev => {
          const next = [...prev, nextData];
          if (next.length > 50) next.shift();
          return next;
        });
        if (nextData.alert) {
          addLog(`ALERT! Infiltration detected (Sev: ${nextData.severity_class})`);
        }
        replayIdx.current = (replayIdx.current + 1) % replayTrace.current.length;
      }, 1000);
    } else {
      if (simulationTimer.current) clearInterval(simulationTimer.current);
    }
    return () => {
      if (simulationTimer.current) clearInterval(simulationTimer.current);
    };
  }, [isSimulating, serialConnected]);

  const connectSerial = async () => {
    try {
      const port = await (navigator as any).serial.requestPort();
      await port.open({ baudRate: 115200 });
      setSerialConnected(true);
      setIsSimulating(false);
      addLog("Connected to ESP32 via Serial");
      
      const textDecoder = new TextDecoderStream();
      port.readable.pipeTo(textDecoder.writable);
      const reader = textDecoder.readable.getReader();
      
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += value;
        const lines = buffer.split('\n');
        buffer = lines.pop() || "";
        
        for (const line of lines) {
          if (line.includes(',')) {
            const parts = line.split(',');
            if (parts.length >= 7) {
              const d = {
                time_ms: parseInt(parts[0]),
                temperature_c: parseFloat(parts[1]),
                pressure_adc: parseFloat(parts[2]),
                condition: parts[3],
                probability: parseFloat(parts[4]),
                severity_class: parseInt(parts[5]),
                alert: parseInt(parts[6])
              };
              setCurrentData(d);
              setDataHistory(prev => {
                const next = [...prev, d];
                if (next.length > 50) next.shift();
                return next;
              });
            }
          }
        }
      }
    } catch (err) {
      console.error(err);
      addLog(`Serial error: ${err}`);
      setSerialConnected(false);
    }
  };

  const handleScrollTo = (id: string) => {
    setActiveTab(id);
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
  };

  const sevClass = currentData ? currentData.severity_class : 0;
  
  // Handle 'nan' probability from ESP32 cleanly
  let prob = '--';
  if (currentData) {
    if (!isNaN(currentData.probability)) {
      prob = (currentData.probability * 100).toFixed(1);
    } else {
      // Fallback display if ESP32 sends literal 'nan' string
      prob = currentData.alert ? '>95.0' : '<10.0';
    }
  }

  const alertStatusColor = currentData?.alert ? 'var(--accent-red)' : 'var(--accent-cyan)';

  return (
    <div className="app-container">
      {/* Top Navigation */}
      <nav className="top-nav">
        <div style={{ display: 'flex', alignItems: 'center', gap: '24px' }}>
          <Activity size={24} color={alertStatusColor} />
          <div>
            <div className="title-primary" style={{ fontSize: '1rem', letterSpacing: '0.2em' }}>INFILTRATE</div>
            <div className="text-caption" style={{ color: 'var(--text-tertiary)' }}>IV MONITORING SYSTEM</div>
          </div>
        </div>
        
        <div className="nav-links">
          {['monitor', 'analysis', 'hardware'].map(tab => (
            <div key={tab} className={`nav-link ${activeTab === tab ? 'active' : ''}`} onClick={() => handleScrollTo(tab)}>
              {tab}
            </div>
          ))}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{ width: 8, height: 8, borderRadius: '50%', backgroundColor: serialConnected ? '#4ade80' : 'var(--text-tertiary)' }} />
          <div>
            <div className="text-caption" style={{ color: 'var(--text-primary)' }}>
              {serialConnected ? 'ESP32 Connected' : isSimulating ? 'SIMULATED' : 'SYSTEM IDLE'}
            </div>
            <div className="text-caption" style={{ fontSize: '0.65rem', color: 'var(--text-tertiary)' }}>
              {serialConnected ? 'Live Inference' : 'Offline Trace'}
            </div>
          </div>
        </div>
      </nav>

      {/* Section 01: MONITOR (Side-by-Side Composition) */}
      <section id="monitor" className="section hero-grid">
        
        {/* Left: 3D Viewport */}
        <div className="viewport-container">
          <Canvas shadows camera={{ position: [0, 0, 18], fov: 45 }}>
            <color attach="background" args={['#06070a']} />
            <ambientLight intensity={1.5} />
            <directionalLight position={[5, 10, 5]} intensity={3} color="#00ffff" />
            <directionalLight position={[-5, -10, -5]} intensity={2} color="#0055ff" />
            <pointLight position={[0, -2, 5]} intensity={2} />
            
            {/* The Unified Assembly anchoring everything together */}
            {/* Master group: rotates the whole assembly to match Reference 2 (hand top-left, forearm bottom-right) */}
            <group position={[1, -1, 0]} rotation={[0, 0, 0.0]}>
              
              {/* ARM */}
              <group position={[0, -5, 0]}>
                <Suspense fallback={<FallbackArm />}>
                  <ArmGLTF />
                </Suspense>
              </group>

              {/* IV HARDWARE mounted flush to the forearm surface */}
              {/* Moved +X to center it on the broad dorsal surface so it lays flat, rather than sticking out the edge */}
              <group position={[0.12, -2.7, 3.0]} rotation={[2.5, 1.6825, -3.0]} scale={[1.4, 1.4, 1.4]}>
                <IVHardware 
                  severity={sevClass} 
                  isFocused={isFocused} 
                  onFocus={() => { setIsFocused(true); setIsInspecting(false); }} 
                  onInspect={() => setIsInspecting(true)}
                />
              </group>

            </group>
            
            <CameraController isFocused={isFocused} />
          </Canvas>

          {/* Overlays on 3D Viewport */}
          <div className="interaction-hints">
            <div className="text-caption" style={{ color: 'var(--text-tertiary)' }}>Interaction</div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.6, marginTop: '8px' }}>
              Drag to rotate<br/>
              Scroll to zoom<br/>
              Click hotspot to focus<br/>
              Double-click to inspect
            </div>
          </div>

          {isFocused && (
            <button className="btn" style={{ position: 'absolute', top: '120px', left: '64px' }} onClick={() => { setIsFocused(false); setIsInspecting(false); }}>
              Reset View
            </button>
          )}

          {/* Deep Inspection Overlay */}
          {isInspecting && (
            <div className="inspect-overlay">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
                <div className="text-caption">Detailed Inspection</div>
                <X size={16} style={{ cursor: 'pointer', color: 'var(--text-secondary)' }} onClick={() => setIsInspecting(false)} />
              </div>

              <div className="data-row" style={{ marginBottom: '24px' }}>
                <span className="text-secondary">P(Infiltration)</span>
                <span className="text-numeric" style={{ fontSize: '1.5rem', color: alertStatusColor }}>{prob}%</span>
              </div>

              <div className="data-row">
                <span className="text-secondary">Temperature</span>
                <span className="text-numeric">{currentData?.temperature_c?.toFixed(1) || '--'} °C</span>
              </div>
              <div className="data-row">
                <span className="text-secondary">Pressure</span>
                <span className="text-numeric">{currentData?.pressure_adc?.toFixed(0) || '--'} ADC</span>
              </div>
              
              <div className="divider" style={{ margin: '16px 0' }} />
              
              <div className="data-row">
                <span className="text-secondary">Condition</span>
                <span style={{ textTransform: 'capitalize' }}>{currentData?.condition.replace('_', ' ') || 'Normal'}</span>
              </div>
              <div className="data-row">
                <span className="text-secondary">Model</span>
                <span>Logistic Regression</span>
              </div>
            </div>
          )}
        </div>

        {/* Right: Monitoring Data (Above Fold) */}
        <div className="data-panel">
          <div style={{ marginBottom: '64px' }}>
            <div className="text-numeric value-massive" style={{ color: alertStatusColor, marginBottom: '8px' }}>
              {prob}<span style={{ fontSize: '2rem' }}>%</span>
            </div>
            <div className="title-primary" style={{ marginBottom: '8px' }}>INFILTRATION<br/>PROBABILITY</div>
            {currentData?.alert ? (
              <div style={{ color: 'var(--bg-base)', background: 'var(--accent-red)', padding: '4px 12px', display: 'inline-block', fontSize: '0.85rem', fontWeight: 600, letterSpacing: '0.1em' }}>
                ● ALERT DETECTED
              </div>
            ) : (
              <div className="text-secondary" style={{ fontSize: '0.85rem' }}>Monitoring baseline</div>
            )}
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
            {/* Temp Waveform */}
            <div>
              <div className="data-row">
                <div className="text-caption">Temperature</div>
                <div className="text-numeric" style={{ fontSize: '1.1rem' }}>{currentData?.temperature_c?.toFixed(1) || '--'} °C</div>
              </div>
              <div className="sparkline-container">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={dataHistory}>
                    <YAxis domain={['dataMin - 1', 'dataMax + 1']} hide />
                    <Tooltip contentStyle={{ background: 'var(--bg-surface)', border: '1px solid var(--border-active)' }} />
                    <Line type="monotone" dataKey="temperature_c" stroke="var(--accent-cyan)" strokeWidth={2} dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Pressure Waveform */}
            <div>
              <div className="data-row">
                <div className="text-caption">Pressure</div>
                <div className="text-numeric" style={{ fontSize: '1.1rem' }}>{currentData?.pressure_adc?.toFixed(0) || '--'} ADC</div>
              </div>
              <div className="sparkline-container">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={dataHistory}>
                    <YAxis domain={['auto', 'auto']} hide />
                    <Tooltip contentStyle={{ background: 'var(--bg-surface)', border: '1px solid var(--border-active)' }} />
                    <Line type="monotone" dataKey="pressure_adc" stroke="var(--text-secondary)" strokeWidth={2} dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Section 02: ANALYSIS */}
      <section id="analysis" className="analysis-section">
        <div>
          <div className="title-primary" style={{ marginBottom: '8px' }}>ANALYSIS</div>
          <div className="text-secondary" style={{ marginBottom: '48px', fontSize: '0.85rem' }}>Signal integrity & fault evaluation.</div>
          
          <div className="text-caption" style={{ marginBottom: '16px' }}>Signal Condition</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {['normal', 'noisy_temp', 'noisy_pressure', 'missing_temp', 'missing_pressure'].map(c => (
              <button key={c} className={`btn ${condition === c ? 'active' : ''}`} onClick={() => setCondition(c)} style={{ textAlign: 'left' }}>
                {c.replace('_', ' ').toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        <div>
          <div className="text-caption" style={{ marginBottom: '16px' }}>Robustness Matrix (F1)</div>
          <div className="glass-card" style={{ padding: 0 }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-secondary)' }}>
                  <th style={{ padding: '16px', textAlign: 'left', fontWeight: 'normal' }}>Condition</th>
                  <th style={{ padding: '16px', textAlign: 'right', fontWeight: 'normal' }}>LogReg</th>
                  <th style={{ padding: '16px', textAlign: 'right', fontWeight: 'normal' }}>Random Forest</th>
                </tr>
              </thead>
              <tbody>
                {robustness.map(r => (
                  <tr key={r.condition} style={{ background: r.condition === condition ? 'rgba(255,255,255,0.05)' : 'transparent', borderBottom: '1px solid var(--border-subtle)' }}>
                    <td style={{ padding: '16px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{r.condition.replace('_',' ')}</td>
                    <td className="text-numeric" style={{ padding: '16px', textAlign: 'right' }}>{r.logreg_f1.toFixed(3)}</td>
                    <td className="text-numeric" style={{ padding: '16px', textAlign: 'right', color: 'var(--text-tertiary)' }}>{r.rf_f1.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* Section 03: HARDWARE */}
      <section id="hardware" className="analysis-section" style={{ background: '#020305' }}>
        <div>
          <div className="title-primary" style={{ marginBottom: '8px' }}>HARDWARE</div>
          <div className="text-secondary" style={{ marginBottom: '48px', fontSize: '0.85rem' }}>Edge inference serial stream.</div>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <button className={`btn ${serialConnected ? 'active' : ''}`} onClick={connectSerial}>
              {serialConnected ? 'ESP32 Connected' : 'Connect ESP32 (Web Serial)'}
            </button>
            <button className={`btn ${isSimulating ? 'active' : ''}`} onClick={() => setIsSimulating(!isSimulating)} disabled={serialConnected}>
              {isSimulating ? 'Stop Simulator' : 'Start Offline Simulator'}
            </button>
          </div>
        </div>

        <div>
          <div className="text-caption" style={{ marginBottom: '16px' }}>Raw Serial Event Stream</div>
          <div className="glass-card" style={{ height: '300px', overflowY: 'auto', fontFamily: 'JetBrains Mono, monospace', fontSize: '0.75rem', background: '#000' }}>
            {logs.map((l, i) => (
              <div key={i} style={{ marginBottom: '12px', color: l.msg.includes('ALERT') ? 'var(--accent-red)' : 'var(--text-secondary)' }}>
                <span style={{ opacity: 0.4, marginRight: '16px' }}>{l.time}</span>
                {l.msg}
              </div>
            ))}
            {logs.length === 0 && <div style={{ opacity: 0.5 }}>Waiting for data stream...</div>}
          </div>
        </div>
      </section>
    </div>
  );
}

export default App;
