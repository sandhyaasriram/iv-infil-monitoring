import { useState, useEffect, useRef, useMemo, Suspense, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Html, useGLTF } from '@react-three/drei';
import * as THREE from 'three';
import { 
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, ReferenceLine 
} from 'recharts';
import { 
  Activity, X, Cpu, Thermometer, Gauge, Waves, ArrowRight,
  Layers, Battery, ShieldCheck, RotateCcw, Play, Pause
} from 'lucide-react';
import SimulationDemo from './components/SimulationDemo';
import { MODEL_REGISTRY } from './modelRegistry';

// --- Types ---
interface SensorData {
  time_ms: number;
  time_s?: number;
  temperature_c: number;
  pressure_adc: number;
  bioimpedance_ohm?: number;
  bioimpedance_rel_drop?: number;
  condition: string;
  probability: number | null;
  severity_class: number;
  alert: number;
  pred_class?: number | null;
  pred_name?: string | null;
  probs?: number[] | null;
}

// --- 3D Holographic Arm ---
const ArmGLTF = () => {
  const { scene } = useGLTF('/arm.glb');
  
  const holoMaterial = useMemo(() => new THREE.MeshStandardMaterial({
    color: '#06b6d4',
    emissive: '#001133',
    emissiveIntensity: 0.35,
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

// Helper to create smooth rounded stadium / capsule shapes
const createStadiumShape = (width: number, height: number, radius: number) => {
  const shape = new THREE.Shape();
  const x = -width / 2;
  const y = -height / 2;
  const r = Math.min(radius, width / 2, height / 2);
  shape.moveTo(x + r, y);
  shape.lineTo(x + width - r, y);
  shape.quadraticCurveTo(x + width, y, x + width, y + r);
  shape.lineTo(x + width, y + height - r);
  shape.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
  shape.lineTo(x + r, y + height);
  shape.quadraticCurveTo(x, y + height, x, y + height - r);
  shape.lineTo(x, y + r);
  shape.quadraticCurveTo(x, y, x + r, y);
  return shape;
};

// --- Slim Medical Sensor Enclosure & Exploded Hardware Stack ---
// Target 3D appearance matches reference image 1: Smooth, closed white medical enclosure on soft beige adhesive dressing.
// Interactive pressed state matches reference image 2: Animated exploded cutaway view revealing internal sensors, ESP32, battery, and adhesive base.
const IVHardware = ({
  severity,
  isFocused,
  isInspecting,
  onFocus,
  onInspect,
  activeComponent = 'all',
  onSelectComponent
}: {
  severity: number;
  isFocused: boolean;
  isInspecting: boolean;
  onFocus: () => void;
  onInspect: () => void;
  activeComponent?: string;
  onSelectComponent?: (comp: any) => void;
}) => {
  const [hovered, setHovered] = useState(false);
  const hotspotColor = severity === 0 ? '#06b6d4' : severity === 1 ? '#eab308' : severity === 2 ? '#f97316' : '#ef4444';
  
  const podRef = useRef<THREE.Group>(null);
  const internalsRef = useRef<THREE.Group>(null);
  const swellingRef = useRef<THREE.Mesh>(null);
  const leakageRef = useRef<THREE.Mesh>(null);
  const ledRef = useRef<THREE.Mesh>(null);

  // Reusable shapes and extrude geometries
  const adhesiveShape = useMemo(() => createStadiumShape(0.88, 0.48, 0.22), []);
  const podShape = useMemo(() => createStadiumShape(0.68, 0.36, 0.16), []);
  const contourShape = useMemo(() => createStadiumShape(0.48, 0.22, 0.10), []);

  const adhesiveGeometry = useMemo(() => new THREE.ExtrudeGeometry(adhesiveShape, {
    depth: 0.006,
    bevelEnabled: true,
    bevelThickness: 0.002,
    bevelSize: 0.003,
    bevelSegments: 3,
  }), [adhesiveShape]);

  const podGeometry = useMemo(() => new THREE.ExtrudeGeometry(podShape, {
    depth: 0.052,
    bevelEnabled: true,
    bevelThickness: 0.022,
    bevelSize: 0.022,
    bevelSegments: 8,
  }), [podShape]);

  const contourGeometry = useMemo(() => new THREE.ExtrudeGeometry(contourShape, {
    depth: 0.002,
    bevelEnabled: false,
  }), [contourShape]);

  // Animated Z elevation for exploded inspection view
  const currentZ = useRef(0);
  const currentScale = useRef(0.001);

  useFrame(({ clock }, delta) => {
    // Pulse indicator LED on the patch
    if (ledRef.current) {
      const pulse = 1 + Math.sin(clock.elapsedTime * 4) * 0.2 * (severity > 0 ? 1 : 0);
      ledRef.current.scale.setScalar(pulse);
    }

    // Smoothly elevate the enclosure pod upward during inspection (exploded cutaway)
    const targetZ = isInspecting ? 0.46 : 0;
    currentZ.current = THREE.MathUtils.damp(currentZ.current, targetZ, 7, delta);
    if (podRef.current) {
      podRef.current.position.z = currentZ.current;
    }

    // Smoothly scale/fade the internal components when exploded
    const targetScale = isInspecting ? 1 : 0.001;
    currentScale.current = THREE.MathUtils.damp(currentScale.current, targetScale, 8, delta);
    if (internalsRef.current) {
      internalsRef.current.scale.setScalar(currentScale.current);
      internalsRef.current.visible = currentZ.current > 0.02;
    }

    // Subcutaneous fluid extravasation swelling & leakage pooling
    if (swellingRef.current && leakageRef.current) {
      const targetSwell = severity === 0 ? 0.001 : 1 + (severity * 0.35);
      const currSwell = swellingRef.current.scale.x;
      const newSwell = THREE.MathUtils.lerp(currSwell, targetSwell, 0.05);

      swellingRef.current.scale.set(newSwell, newSwell, newSwell * 0.3);
      leakageRef.current.scale.set(newSwell, newSwell, 1);

      const leakMat = leakageRef.current.material as THREE.MeshPhysicalMaterial;
      const swellMat = swellingRef.current.material as THREE.MeshPhysicalMaterial;

      const targetOpacity = severity > 0 ? 0.35 + (Math.sin(clock.elapsedTime * 2) * 0.05) : 0;
      leakMat.opacity = THREE.MathUtils.lerp(leakMat.opacity, targetOpacity, 0.1);
      swellMat.opacity = THREE.MathUtils.lerp(swellMat.opacity, severity > 0 ? 0.55 : 0, 0.1);
    }
  });

  return (
    <group>
      {/* A. Subcutaneous Fluid Extravasation Swelling & Pooling (under skin near catheter tip) */}
      <group position={[0, 0.36, -0.02]}>
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
        
        <mesh ref={leakageRef} position={[0, 0, 0.01]}>
          <circleGeometry args={[0.24, 32]} />
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

      {/* B. Transparent Medical Cannula Entry Line */}
      <mesh position={[0, 0.58, 0.014]} rotation={[0.38, 0, 0]}>
        <cylinderGeometry args={[0.010, 0.010, 0.55, 12]} />
        <meshPhysicalMaterial color="#e0f2fe" transmission={0.92} opacity={0.75} transparent roughness={0.15} />
      </mesh>

      {/* Low-Profile Butterfly Fixation Wing Hub */}
      <mesh position={[0, 0.42, 0.012]}>
        <boxGeometry args={[0.15, 0.045, 0.008]} />
        <meshStandardMaterial color="#0284c7" roughness={0.3} transparent opacity={0.85} />
      </mesh>

      {/* 1. SOFT BEIGE SILICONE ADHESIVE DRESSING (Skin-Friendly Base) */}
      <group position={[0, 0, 0.003]}>
        {/* Main Soft Skin-Compatible Adhesive Dressing Body */}
        <mesh geometry={adhesiveGeometry}>
          <meshPhysicalMaterial 
            color="#e5ded4" 
            roughness={0.82} 
            metalness={0.02} 
            transmission={0.12} 
            transparent 
            opacity={0.96} 
          />
        </mesh>

        {/* Subtle Adhesive Fabric Margin Rim */}
        <mesh position={[0, 0, 0.004]}>
          <ringGeometry args={[0.42, 0.44, 32]} />
          <meshBasicMaterial color="#d4cbbd" transparent opacity={0.4} side={THREE.DoubleSide} />
        </mesh>

        {/* 4 Silver Bioimpedance Hydrogel Contact Discs (Resting on skin under adhesive - clearly flagged as Proposed Extension) */}
        <group position={[0, 0, 0.005]}>
          <mesh position={[-0.26, 0.02, 0]}>
            <cylinderGeometry args={[0.025, 0.025, 0.002, 16]} />
            <meshStandardMaterial color="#e2e8f0" metalness={0.9} roughness={0.2} />
          </mesh>
          <mesh position={[-0.14, 0.02, 0]}>
            <cylinderGeometry args={[0.025, 0.025, 0.002, 16]} />
            <meshStandardMaterial color="#e2e8f0" metalness={0.9} roughness={0.2} />
          </mesh>
          <mesh position={[0.14, -0.04, 0]}>
            <cylinderGeometry args={[0.025, 0.025, 0.002, 16]} />
            <meshStandardMaterial color="#e2e8f0" metalness={0.9} roughness={0.2} />
          </mesh>
          <mesh position={[0.26, -0.04, 0]}>
            <cylinderGeometry args={[0.025, 0.025, 0.002, 16]} />
            <meshStandardMaterial color="#e2e8f0" metalness={0.9} roughness={0.2} />
          </mesh>
        </group>
      </group>

      {/* 2. INTERNAL EXPLODED HARDWARE STACK (Revealed Only When Inspected - Target Image 2) */}
      <group ref={internalsRef} visible={false}>
        {/* Layer A: Rechargeable Li-Po Battery Cell */}
        <group 
          position={[0.01, -0.02, 0.04]}
          onClick={(e) => { e.stopPropagation(); onSelectComponent?.('battery'); }}
        >
          {/* Silver Aluminum Pouch Cell Body */}
          <mesh>
            <boxGeometry args={[0.32, 0.18, 0.032]} />
            <meshStandardMaterial 
              color={activeComponent === 'battery' ? '#38bdf8' : '#cbd5e1'} 
              metalness={0.85} 
              roughness={0.25} 
            />
          </mesh>
          {/* Amber Kapton Polyimide Tape Edge Wraps */}
          <mesh position={[-0.145, 0, 0]}>
            <boxGeometry args={[0.03, 0.182, 0.034]} />
            <meshStandardMaterial color="#d97706" roughness={0.3} metalness={0.1} />
          </mesh>
          <mesh position={[0.145, 0, 0]}>
            <boxGeometry args={[0.03, 0.182, 0.034]} />
            <meshStandardMaterial color="#d97706" roughness={0.3} metalness={0.1} />
          </mesh>
          {/* Power Lead Pins */}
          <mesh position={[0.04, 0.10, 0.008]} rotation={[Math.PI / 2, 0, 0]}>
            <cylinderGeometry args={[0.003, 0.003, 0.025, 8]} />
            <meshBasicMaterial color="#ef4444" />
          </mesh>
          <mesh position={[-0.04, 0.10, 0.008]} rotation={[Math.PI / 2, 0, 0]}>
            <cylinderGeometry args={[0.003, 0.003, 0.025, 8]} />
            <meshBasicMaterial color="#0f172a" />
          </mesh>
        </group>

        {/* Layer B: ESP32 Processing Board (Data processing + WiFi/BLE) */}
        <group 
          position={[0.01, -0.02, 0.125]}
          onClick={(e) => { e.stopPropagation(); onSelectComponent?.('esp32'); }}
        >
          {/* Green Solder Mask FR4 PCB */}
          <mesh>
            <boxGeometry args={[0.36, 0.22, 0.008]} />
            <meshStandardMaterial 
              color={activeComponent === 'esp32' ? '#059669' : '#064e3b'} 
              roughness={0.4} 
              metalness={0.1} 
            />
          </mesh>
          {/* Gold Edge Castellated Pads & Mounting Holes */}
          <mesh position={[0, 0.105, 0.001]}>
            <boxGeometry args={[0.34, 0.006, 0.009]} />
            <meshStandardMaterial color="#f59e0b" metalness={0.9} roughness={0.15} />
          </mesh>
          <mesh position={[0, -0.105, 0.001]}>
            <boxGeometry args={[0.34, 0.006, 0.009]} />
            <meshStandardMaterial color="#f59e0b" metalness={0.9} roughness={0.15} />
          </mesh>
          {/* ESP32 Stamped Silver Metal RF Shield Can */}
          <mesh position={[0.01, 0.01, 0.011]}>
            <boxGeometry args={[0.17, 0.17, 0.014]} />
            <meshStandardMaterial color="#e2e8f0" metalness={0.85} roughness={0.2} />
          </mesh>
          {/* Ceramic Resonator / Crystal */}
          <mesh position={[-0.12, 0.04, 0.008]}>
            <boxGeometry args={[0.04, 0.03, 0.008]} />
            <meshStandardMaterial color="#94a3b8" metalness={0.9} />
          </mesh>
          {/* Micro SMD Resistors & Capacitors */}
          <mesh position={[-0.12, -0.03, 0.007]}>
            <boxGeometry args={[0.025, 0.015, 0.006]} />
            <meshStandardMaterial color="#0f172a" />
          </mesh>
          <mesh position={[-0.12, -0.06, 0.007]}>
            <boxGeometry args={[0.025, 0.015, 0.006]} />
            <meshStandardMaterial color="#0f172a" />
          </mesh>
          {/* Printed Meander Antenna Trace */}
          <mesh position={[0.13, 0, 0.006]}>
            <boxGeometry args={[0.04, 0.14, 0.004]} />
            <meshStandardMaterial color="#f59e0b" metalness={0.85} roughness={0.2} />
          </mesh>
        </group>

        {/* Layer C: Temperature Sensor (Skin Surface Contact Probe) */}
        <group 
          position={[-0.14, 0.13, 0.22]}
          onClick={(e) => { e.stopPropagation(); onSelectComponent?.('temp'); }}
        >
          {/* Blue Breakout PCB */}
          <mesh>
            <boxGeometry args={[0.09, 0.06, 0.006]} />
            <meshStandardMaterial color={activeComponent === 'temp' ? '#38bdf8' : '#1d4ed8'} roughness={0.3} />
          </mesh>
          {/* Thermal Sensor IC Package */}
          <mesh position={[0, 0, 0.005]}>
            <boxGeometry args={[0.03, 0.03, 0.005]} />
            <meshStandardMaterial color="#0f172a" />
          </mesh>
          {/* Thermal Probe Needle extending straight down to skin surface at catheter tip */}
          <mesh position={[0, 0, -0.11]}>
            <cylinderGeometry args={[0.004, 0.004, 0.22, 12]} />
            <meshStandardMaterial color="#cbd5e1" metalness={0.95} roughness={0.1} />
          </mesh>
        </group>

        {/* Layer D: Pressure Sensor (Under Patch - Subcutaneous Micro-Strain) */}
        <group 
          position={[0.14, 0.13, 0.22]}
          onClick={(e) => { e.stopPropagation(); onSelectComponent?.('pressure'); }}
        >
          {/* Blue Breakout PCB */}
          <mesh>
            <boxGeometry args={[0.08, 0.08, 0.006]} />
            <meshStandardMaterial color={activeComponent === 'pressure' ? '#c084fc' : '#1d4ed8'} roughness={0.3} />
          </mesh>
          {/* Piezoresistive Sensing Disc underneath */}
          <mesh position={[0, 0, -0.004]}>
            <cylinderGeometry args={[0.032, 0.032, 0.003, 20]} />
            <meshStandardMaterial color="#0f172a" roughness={0.6} />
          </mesh>
        </group>

        {/* Exploded Alignment Guide Lines (Colored dashed vertical lines matching Image 2) */}
        {/* Orange line to Temperature Sensor */}
        <mesh position={[-0.14, 0.13, 0.11]}>
          <cylinderGeometry args={[0.0015, 0.0015, 0.22, 6]} />
          <meshBasicMaterial color="#f97316" transparent opacity={0.6} />
        </mesh>
        {/* Cyan line to Pressure Sensor */}
        <mesh position={[0.14, 0.13, 0.11]}>
          <cylinderGeometry args={[0.0015, 0.0015, 0.22, 6]} />
          <meshBasicMaterial color="#06b6d4" transparent opacity={0.6} />
        </mesh>
        {/* Green line to ESP32 / Battery */}
        <mesh position={[0.01, -0.02, 0.08]}>
          <cylinderGeometry args={[0.0015, 0.0015, 0.09, 6]} />
          <meshBasicMaterial color="#10b981" transparent opacity={0.5} />
        </mesh>
      </group>

      {/* 3. SMOOTH WHITE MEDICAL ENCLOSURE POD (Target Image 1 Default / Elevated when Inspected) */}
      <group 
        ref={podRef} 
        position={[0, 0, 0]}
        onClick={(e) => {
          e.stopPropagation();
          if (!isFocused) onFocus();
          onInspect();
        }}
        onPointerOver={(e) => {
          e.stopPropagation();
          setHovered(true);
          document.body.style.cursor = 'pointer';
        }}
        onPointerOut={(e) => {
          e.stopPropagation();
          setHovered(false);
          document.body.style.cursor = 'auto';
        }}
      >
        {/* Main Solid White Biocompatible Molded Enclosure Shell */}
        <mesh geometry={podGeometry} position={[0, 0, 0.004]}>
          <meshPhysicalMaterial 
            color="#fcfcfd" 
            roughness={0.26} 
            metalness={0.05} 
            clearcoat={0.35} 
            clearcoatRoughness={0.2} 
          />
        </mesh>

        {/* Top Face Details (Image 1 Features) */}
        {/* A. Recessed Debossed Oval Contour Groove */}
        <mesh geometry={contourGeometry} position={[0, 0, 0.076]}>
          <meshBasicMaterial color="#e5e7eb" />
        </mesh>

        {/* B. Central Stylized Cyan/Teal Diamond Logo Emblem */}
        <group position={[0, 0, 0.078]} scale={[0.12, 0.12, 0.12]}>
          {/* Left Teal Chevron Ribbon */}
          <mesh position={[-0.22, 0, 0]} rotation={[0, 0, Math.PI / 4]}>
            <boxGeometry args={[0.55, 0.16, 0.01]} />
            <meshStandardMaterial color="#06b6d4" emissive="#0891b2" emissiveIntensity={0.4} roughness={0.2} />
          </mesh>
          {/* Right Cyan Chevron Ribbon */}
          <mesh position={[0.22, 0, 0]} rotation={[0, 0, Math.PI / 4]}>
            <boxGeometry args={[0.55, 0.16, 0.01]} />
            <meshStandardMaterial color="#0ea5e9" emissive="#0284c7" emissiveIntensity={0.4} roughness={0.2} />
          </mesh>
          {/* Central Connecting Node */}
          <mesh position={[0, 0, 0.002]} rotation={[0, 0, Math.PI / 4]}>
            <boxGeometry args={[0.22, 0.22, 0.01]} />
            <meshStandardMaterial color="#38bdf8" emissive="#06b6d4" emissiveIntensity={0.5} roughness={0.2} />
          </mesh>
        </group>

        {/* C. Upper-Right Optical Sensor Lens Aperture Dot (Image 1) */}
        <group position={[0.21, 0.10, 0.078]}>
          {/* Dark Circular Aperture Rim */}
          <mesh rotation={[Math.PI / 2, 0, 0]}>
            <cylinderGeometry args={[0.016, 0.016, 0.004, 16]} />
            <meshStandardMaterial color="#0f172a" roughness={0.15} metalness={0.8} />
          </mesh>
          {/* Active Internal Micro-LED Optical Indicator */}
          <mesh ref={ledRef} position={[0, 0, 0.003]}>
            <sphereGeometry args={[0.007, 12, 12]} />
            <meshBasicMaterial color={hotspotColor} />
          </mesh>
        </group>

        {/* Interactive Click Hitbox */}
        <mesh position={[0, 0, 0.04]}>
          <boxGeometry args={[0.74, 0.42, 0.1]} />
          <meshBasicMaterial transparent opacity={0} depthWrite={false} />
        </mesh>

        {/* Hover Information Tooltip when not yet inspecting */}
        {hovered && !isInspecting && (
          <Html position={[0.42, 0, 0.05]} center style={{ pointerEvents: 'none', zIndex: 10 }}>
            <div className="hotspot-panel">
              <div className="text-caption" style={{ marginBottom: '6px', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '4px', color: 'var(--accent-cyan)' }}>
                IV Infiltration Patch
              </div>
              <div className="data-row">
                <span className="text-secondary" style={{ fontSize: '0.72rem' }}>Status</span>
                <span style={{ color: hotspotColor, fontWeight: 600, fontSize: '0.8rem' }}>
                  {severity === 0 ? 'Normal Baseline' : `Level ${severity} Extravasation`}
                </span>
              </div>
              <div className="text-caption" style={{ marginTop: '6px', color: 'var(--accent-cyan)', fontSize: '0.66rem' }}>
                Click to Open Exploded View
              </div>
            </div>
          </Html>
        )}
      </group>
    </group>
  );
};

// --- Camera Controller ---
const CameraController = ({ isFocused }: { isFocused: boolean }) => {
  const { camera } = useThree();
  const controlsRef = useRef<any>(null);
  const [animating, setAnimating] = useState(false);

  // Framed to keep the forearm centered in the viewport
  const targetPos = useMemo(() => isFocused ? new THREE.Vector3(0.9, -0.3, 6.8) : new THREE.Vector3(0, 0, 14.0), [isFocused]);
  const targetLookAt = useMemo(() => isFocused ? new THREE.Vector3(0.12, 0.25, 3.0) : new THREE.Vector3(0.15, 0.2, 1.0), [isFocused]);

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
      minDistance={2.0} 
      maxDistance={22} 
      onStart={() => setAnimating(false)}
    />
  );
};


// --- Main Application ---
function App() {
  const [dataHistory, setDataHistory] = useState<SensorData[]>([]);
  const [currentData, setCurrentData] = useState<SensorData | null>(null);
  
  const [isSimulating, setIsSimulating] = useState(false);
  const [monitorScenario, setMonitorScenario] = useState<'infiltration' | 'normal' | 'stress' | 'noisy'>('infiltration');
  const [monitorModel, setMonitorModel] = useState<string>('model1_tcn');
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);

  const [activeAnalysisTab, setActiveAnalysisTab] = useState<'models' | 'confounder' | 'discrepancy'>('models');
  
  const [serialConnected, setSerialConnected] = useState(false);
  const [logs, setLogs] = useState<{time:string, msg:string}[]>([]);
  
  const [isFocused, setIsFocused] = useState(false);
  const [isInspecting, setIsInspecting] = useState(false);
  const [activeSensorComponent, setActiveSensorComponent] = useState<'all' | 'enclosure' | 'temp' | 'pressure' | 'esp32' | 'battery' | 'adhesive' | 'bio'>('all');
  
  const [activeTab, setActiveTab] = useState('monitor');

  const simulationTimer = useRef<number | null>(null);
  const replayTrace = useRef<SensorData[]>([]);
  const replayIdx = useRef(0);

  const addLog = (msg: string) => {
    setLogs(prev => [{ time: new Date().toLocaleTimeString(), msg }, ...prev].slice(0, 50));
  };

  useEffect(() => {
    const handleScroll = () => {
      const sections = ['monitor', 'demo', 'patch', 'analysis', 'hardware'];
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

  // Dynamic Replay Loader: Invokes trained inference engine (/api/simulate) or falls back to neutral biophysical trace
  const loadReplayData = useCallback(async (scenario: string, modelId: string) => {
    try {
      const res = await fetch('/api/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scenario: scenario,
          model_id: modelId,
          duration_s: 180,
          sampling_hz: 1
        })
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      setBackendOnline(true);

      const trace: SensorData[] = (json.data || []).map((d: any) => ({
        time_ms: (d.time_s ?? 0) * 1000,
        time_s: d.time_s,
        temperature_c: d.temperature_c,
        pressure_adc: d.pressure_adc,
        bioimpedance_ohm: d.bioimpedance_ohm,
        bioimpedance_rel_drop: d.bioimpedance_rel_drop,
        condition: d.ground_truth_name || scenario,
        probability: d.probability !== undefined ? d.probability : null,
        severity_class: d.pred_class !== null && d.pred_class !== undefined ? d.pred_class : (d.probability !== null && d.probability >= 0.7 ? 2 : 0),
        alert: d.alert ? 1 : 0,
        pred_class: d.pred_class,
        pred_name: d.pred_name,
        probs: d.probs
      }));

      replayTrace.current = trace;
      replayIdx.current = 0;
      if (trace.length > 0) {
        setCurrentData(trace[0]);
        // Seed dataHistory with initial samples so charts render real numeric baseline immediately
        setDataHistory(trace.slice(0, Math.min(2, trace.length)));
      }
      addLog(`Loaded ${json.model_name || modelId} inference for ${scenario} (${trace.length}s)`);
    } catch (err: any) {
      setBackendOnline(false);
      console.warn("Backend /api/simulate unavailable, generating biophysical waveform without inference:", err);
      // Offline fallback: purely biophysical waveform, explicitly setting probability to null
      const fallbackTrace: SensorData[] = [];
      const n = 180;
      const isInf = scenario === 'infiltration';
      const isStress = scenario === 'stress';
      const isNoise = scenario === 'noisy';

      for (let i = 0; i < n; i++) {
        let temp = 33.0;
        let pres = 1160.0;
        let bio = 580.0;

        if (isInf && i >= 40) {
          const dt = i - 40;
          temp -= Math.min(2.2, dt * 0.035) + (Math.random() - 0.5) * 0.05;
          pres += Math.min(110, dt * 1.8) + (Math.random() - 0.5) * 2.0;
          bio -= Math.min(145, dt * 2.4) + (Math.random() - 0.5) * 1.5;
        } else if (isStress && i >= 30) {
          const dt = i - 30;
          temp -= Math.min(1.8, dt * 0.03) + (Math.random() - 0.5) * 0.05;
          pres += (Math.random() - 0.5) * 2.0;
          bio += (Math.random() - 0.5) * 1.5;
        } else if (isNoise) {
          temp += (Math.random() - 0.5) * 0.3;
          pres += (Math.random() - 0.5) * 45.0;
          bio += (Math.random() - 0.5) * 15.0;
        } else {
          temp += (Math.random() - 0.5) * 0.05;
          pres += (Math.random() - 0.5) * 2.0;
          bio += (Math.random() - 0.5) * 1.5;
        }

        fallbackTrace.push({
          time_ms: i * 1000,
          time_s: i,
          temperature_c: Math.round(temp * 100) / 100,
          pressure_adc: Math.round(pres * 10) / 10,
          bioimpedance_ohm: Math.round(bio * 10) / 10,
          condition: scenario,
          probability: null, // NO FAKE 0.0%: Model inference not running on client offline fallback
          severity_class: 0,
          alert: 0,
          pred_class: null,
          pred_name: 'Inference Unavailable (Offline)'
        });
      }

      replayTrace.current = fallbackTrace;
      replayIdx.current = 0;
      if (fallbackTrace.length > 0) {
        setCurrentData(fallbackTrace[0]);
        setDataHistory(fallbackTrace.slice(0, Math.min(2, fallbackTrace.length)));
      }
      addLog(`Offline trace loaded for ${scenario} (Inference unavailable)`);
    }
  }, []);

  useEffect(() => {
    loadReplayData(monitorScenario, monitorModel);
  }, [monitorScenario, monitorModel, loadReplayData]);

  const restartReplay = () => {
    replayIdx.current = 0;
    if (replayTrace.current.length > 0) {
      setCurrentData(replayTrace.current[0]);
      setDataHistory(replayTrace.current.slice(0, Math.min(2, replayTrace.current.length)));
    }
    addLog(`Replay restarted at t=0s`);
  };

  // Offline replay timer
  useEffect(() => {
    if (isSimulating && !serialConnected) {
      simulationTimer.current = window.setInterval(() => {
        if (replayTrace.current.length === 0) return;
        const curIdx = replayIdx.current;
        const nextData = replayTrace.current[curIdx];
        setCurrentData(nextData);
        // Slices all real numeric samples up to the current playback position
        setDataHistory(replayTrace.current.slice(0, Math.max(2, curIdx + 1)));
        if (nextData.alert) {
          addLog(`ALERT! Extravasation detected (t=${nextData.time_s ?? curIdx}s)`);
        }
        replayIdx.current = (curIdx + 1) % replayTrace.current.length;
      }, 1000);
    } else {
      if (simulationTimer.current) clearInterval(simulationTimer.current);
    }
    return () => {
      if (simulationTimer.current) clearInterval(simulationTimer.current);
    };
  }, [isSimulating, serialConnected]);

  // Web Serial Port Connection for physical ESP32
  const connectSerial = async () => {
    try {
      const port = await (navigator as any).serial.requestPort();
      await port.open({ baudRate: 115200 });
      setSerialConnected(true);
      setIsSimulating(false);
      addLog("Connected to ESP32 via Serial (115200 baud)");
      
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
    } catch (err: any) {
      console.warn("Serial connection notice:", err);
      addLog(`Serial notice: ${err.message || 'Port not selected'}`);
      setSerialConnected(false);
    }
  };

  const handleScrollTo = (id: string) => {
    setActiveTab(id);
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
  };

  const sevClass = currentData ? currentData.severity_class : 0;
  
  let probDisplay = '--';
  let probUnit = '%';
  let isBuffering = false;
  let isInferenceUnavailable = false;

  let statusColor = 'var(--accent-cyan)';
  let statusText = 'Normal Baseline';
  let statusBg = 'transparent';
  let statusTextColor = 'var(--text-secondary)';

  if (currentData) {
    if (currentData.probability !== null && !isNaN(currentData.probability)) {
      probDisplay = (currentData.probability * 100).toFixed(1);
      probUnit = '%';

      if (currentData.alert || currentData.probability >= 0.70 || (currentData.pred_class !== null && currentData.pred_class !== undefined && currentData.pred_class >= 2)) {
        statusColor = '#ef4444';
        statusText = 'CRITICAL ALERT: EXTRAVASATION';
        statusBg = '#ef4444';
        statusTextColor = 'var(--bg-base)';
      } else if (currentData.probability >= 0.35 || (currentData.pred_class !== null && currentData.pred_class !== undefined && currentData.pred_class === 1)) {
        statusColor = '#f97316';
        statusText = 'HIGH RISK: INFILTRATION SUSPECTED';
        statusBg = '#f97316';
        statusTextColor = 'var(--bg-base)';
      } else if (currentData.probability >= 0.15) {
        statusColor = '#eab308';
        statusText = 'ELEVATED RISK: EARLY ONSET';
        statusBg = '#eab308';
        statusTextColor = 'var(--bg-base)';
      } else {
        statusColor = 'var(--accent-cyan)';
        statusText = 'Normal Infusion Baseline';
        statusBg = 'transparent';
        statusTextColor = 'var(--accent-cyan)';
      }
    } else {
      // Probability is null
      if (currentData.time_s !== undefined && currentData.time_s < 50) {
        isBuffering = true;
        probDisplay = 'Buffering';
        probUnit = '';
        statusColor = '#94a3b8';
        statusText = `Buffering Window (${currentData.time_s}s / 50s)`;
        statusBg = 'rgba(148, 163, 184, 0.15)';
        statusTextColor = '#94a3b8';
      } else {
        isInferenceUnavailable = true;
        probDisplay = 'N/A';
        probUnit = '';
        statusColor = '#94a3b8';
        statusText = currentData.pred_name || 'Replay Active — No Prediction';
        statusBg = 'rgba(148, 163, 184, 0.15)';
        statusTextColor = '#94a3b8';
      }
    }
  }

  return (
    <div className="app-container">
      {/* Compact Top Navigation */}
      <nav className="top-nav">
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <Activity size={20} color={statusColor} />
          <div>
            <div className="title-primary" style={{ fontSize: '0.9rem', letterSpacing: '0.15em', fontWeight: 600 }}>INFILTRATE</div>
            <div className="text-caption" style={{ color: 'var(--text-tertiary)', fontSize: '0.62rem' }}>IV MONITORING SYSTEM</div>
          </div>
        </div>
        
        <div className="nav-links">
          {[
            { id: 'monitor', label: 'Monitor' },
            { id: 'demo', label: 'Simulation Demo' },
            { id: 'patch', label: 'Patch Architecture' },
            { id: 'analysis', label: 'Analysis' },
            { id: 'hardware', label: 'Hardware' },
          ].map(tab => (
            <div key={tab.id} className={`nav-link ${activeTab === tab.id ? 'active' : ''}`} onClick={() => handleScrollTo(tab.id)}>
              {tab.label}
            </div>
          ))}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{ width: 7, height: 7, borderRadius: '50%', backgroundColor: serialConnected ? '#4ade80' : isSimulating ? 'var(--accent-cyan)' : 'var(--text-tertiary)' }} />
          <div className="text-caption" style={{ color: 'var(--text-primary)', fontSize: '0.7rem' }}>
            {serialConnected ? 'ESP32 Connected' : isSimulating ? 'Replay Stream' : 'Ready'}
          </div>
        </div>
      </nav>

      {/* SECTION 01: MONITOR (Vertically Balanced Hero Viewport + Lower Real-Time Charts) */}
      <section id="monitor" style={{ position: 'relative' }}>
        <div className="hero-grid">
        
        {/* Left: 3D Viewport (Framed & Centered at Eye Level) */}
        <div className="viewport-container">
          <Canvas shadows camera={{ position: [0, 0, 14.5], fov: 40 }}>
            <color attach="background" args={['#06070a']} />
            <ambientLight intensity={1.6} />
            <directionalLight position={[5, 10, 5]} intensity={3} color="#00ffff" />
            <directionalLight position={[-5, -10, -5]} intensity={2} color="#0055ff" />
            <pointLight position={[0, 0, 5]} intensity={2} />
            
            {/* Centered Arm & Slim Wearable Patch Assembly */}
            <group position={[0.4, 0.4, 0]} rotation={[0, 0, 0.0]}>
              <group position={[0, -2.6, 0]}>
                <Suspense fallback={<FallbackArm />}>
                  <ArmGLTF />
                </Suspense>
              </group>

              {/* Slim Integrated Patch mounted flush to the forearm */}
              <group position={[0.12, -0.3, 3.0]} rotation={[2.5, 1.6825, -3.0]} scale={[1.4, 1.4, 1.4]}>
                <IVHardware 
                  severity={sevClass} 
                  isFocused={isFocused} 
                  isInspecting={isInspecting}
                  onFocus={() => { setIsFocused(true); setIsInspecting(true); }} 
                  onInspect={() => {
                    setIsFocused(true);
                    setIsInspecting(prev => !prev);
                  }}
                  activeComponent={activeSensorComponent}
                  onSelectComponent={(c) => setActiveSensorComponent(c)}
                />
              </group>
            </group>
            
            <CameraController isFocused={isFocused} />
          </Canvas>

          {/* Compact Viewport Hints */}
          <div className="interaction-hints">
            <div className="text-caption" style={{ color: 'var(--accent-cyan)', fontSize: '0.65rem' }}>3D Controls</div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', lineHeight: 1.4, marginTop: '2px' }}>
              Drag to rotate | Scroll to zoom | Click patch to toggle exploded view
            </div>
          </div>

          <div style={{ position: 'absolute', top: '16px', left: '24px', display: 'flex', gap: '8px', zIndex: 10 }}>
            {isFocused && (
              <button className="btn" style={{ padding: '6px 12px', fontSize: '0.72rem' }} onClick={() => { setIsFocused(false); setIsInspecting(false); }}>
                Reset View
              </button>
            )}
            <button 
              className={`btn ${isInspecting ? 'active' : ''}`}
              style={{ padding: '6px 12px', fontSize: '0.72rem', display: 'flex', alignItems: 'center', gap: '6px' }}
              onClick={() => {
                if (!isFocused) setIsFocused(true);
                setIsInspecting(prev => !prev);
              }}
            >
              <Layers size={13} />
              <span>{isInspecting ? 'Close Exploded View' : 'Inspect Hardware Stack'}</span>
            </button>
          </div>

          {/* Exploded Hardware Stack Inspection Drawer (Matches Reference Image 2) */}
          {isInspecting && (
            <div className="inspect-overlay" style={{ maxHeight: '88vh', overflowY: 'auto' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Layers size={15} color="var(--accent-cyan)" />
                  <div className="text-caption" style={{ color: 'var(--accent-cyan)', fontWeight: 600 }}>Physical Patch Hardware Stack</div>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span style={{ fontSize: '0.62rem', background: 'rgba(6, 182, 212, 0.15)', color: 'var(--accent-cyan)', padding: '2px 6px', borderRadius: '4px', border: '1px solid rgba(6, 182, 212, 0.3)' }}>
                    Exploded 3D View
                  </span>
                  <X size={15} style={{ cursor: 'pointer', color: 'var(--text-secondary)' }} onClick={() => setIsInspecting(false)} />
                </div>
              </div>

              {/* Exploded Component Selector Tabs (Mirroring Reference Image 2) */}
              <div style={{ display: 'flex', gap: '4px', marginBottom: '14px', flexWrap: 'wrap' }}>
                {[
                  { id: 'all', label: 'Stack Overview' },
                  { id: 'enclosure', label: 'Enclosure' },
                  { id: 'temp', label: 'Temp Sensor' },
                  { id: 'pressure', label: 'Pressure Sensor' },
                  { id: 'esp32', label: 'ESP32 SoC' },
                  { id: 'battery', label: 'Li-Po Battery' },
                  { id: 'adhesive', label: 'Adhesive Base' },
                  { id: 'bio', label: 'Bioimpedance*' }
                ].map(item => (
                  <button
                    key={item.id}
                    onClick={() => setActiveSensorComponent(item.id as any)}
                    style={{
                      background: activeSensorComponent === item.id ? 'rgba(6, 182, 212, 0.2)' : 'rgba(255,255,255,0.03)',
                      border: `1px solid ${activeSensorComponent === item.id ? 'var(--accent-cyan)' : 'var(--border-subtle)'}`,
                      color: activeSensorComponent === item.id ? 'var(--accent-cyan)' : 'var(--text-secondary)',
                      borderRadius: '4px',
                      padding: '3px 7px',
                      fontSize: '0.67rem',
                      cursor: 'pointer',
                      whiteSpace: 'nowrap'
                    }}
                  >
                    {item.label}
                  </button>
                ))}
              </div>

              {/* TAB 1: STACK OVERVIEW (Illustrated Component Stack from Image 2) */}
              {activeSensorComponent === 'all' && (
                <div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginBottom: '10px', lineHeight: 1.4 }}>
                    Exploded physical layers of the wearable infiltration monitor, showing encapsulation, embedded processing, sensing elements, and skin interface.
                  </div>

                  {/* Vertical Layer Breakdown */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginBottom: '14px' }}>
                    {[
                      { id: 'enclosure', title: '1. Enclosure (Biocompatible, molded shell)', desc: 'Smooth white protective cap, IP67 sealed', color: 'var(--text-primary)', badge: 'Top Pod' },
                      { id: 'temp', title: '2. Temperature sensor (Skin surface)', desc: 'Direct thermal probe needle at catheter insertion site', color: 'var(--accent-cyan)', badge: 'Primary' },
                      { id: 'pressure', title: '3. Pressure sensor (Under patch)', desc: 'Piezoresistive compartment micro-strain transducer', color: '#c084fc', badge: 'Primary' },
                      { id: 'esp32', title: '4. ESP32 (Data processing + WiFi/BLE)', desc: 'Xtensa dual-core running 1 Hz sliding-window inference', color: '#34d399', badge: 'Edge SoC' },
                      { id: 'battery', title: '5. Li-Po Battery (Rechargeable)', desc: '3.7V ultra-slim pouch cell with Kapton tape edging', color: '#fbbf24', badge: 'Power' },
                      { id: 'adhesive', title: '6. Silicone adhesive patch (Skin friendly)', desc: 'Conforming beige transdermal dressing', color: '#e2e8f0', badge: 'Interface' },
                      { id: 'bio', title: '7. Tetrapolar Bioimpedance Array*', desc: '4-electrode Kelvin array for tissue admittance', color: '#4ade80', badge: 'Proposed Extension' },
                    ].map(layer => (
                      <div
                        key={layer.id}
                        onClick={() => setActiveSensorComponent(layer.id as any)}
                        style={{
                          background: 'rgba(255,255,255,0.02)',
                          border: '1px solid var(--border-subtle)',
                          borderRadius: '5px',
                          padding: '7px 10px',
                          cursor: 'pointer',
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                          transition: 'border-color 0.15s ease'
                        }}
                      >
                        <div>
                          <div style={{ fontSize: '0.72rem', fontWeight: 600, color: layer.color }}>{layer.title}</div>
                          <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)' }}>{layer.desc}</div>
                        </div>
                        <span style={{ fontSize: '0.6rem', color: 'var(--text-tertiary)', background: 'rgba(255,255,255,0.04)', padding: '2px 5px', borderRadius: '3px', whiteSpace: 'nowrap' }}>
                          {layer.badge}
                        </span>
                      </div>
                    ))}
                  </div>

                  {/* Bedside Metric Summary */}
                  <div className="divider" style={{ margin: '10px 0' }} />
                  <div className="data-row">
                    <span className="text-secondary" style={{ fontSize: '0.72rem' }}>Risk Assessment</span>
                    <span className="text-numeric" style={{ color: statusColor, fontWeight: 600 }}>{probDisplay}{probUnit ? ` ${probUnit}` : ''} ({statusText})</span>
                  </div>
                  <div className="data-row">
                    <span className="text-secondary" style={{ fontSize: '0.72rem' }}>Skin Temperature</span>
                    <span className="text-numeric">{currentData?.temperature_c?.toFixed(1) || '33.0'} °C</span>
                  </div>
                  <div className="data-row">
                    <span className="text-secondary" style={{ fontSize: '0.72rem' }}>Tissue Strain</span>
                    <span className="text-numeric">{currentData?.pressure_adc?.toFixed(0) || '1160'} ADC</span>
                  </div>
                </div>
              )}

              {/* TAB 2: ENCLOSURE */}
              {activeSensorComponent === 'enclosure' && (
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-primary)', fontWeight: 600, marginBottom: '6px' }}>
                    <ShieldCheck size={16} color="var(--accent-cyan)" />
                    <span>Enclosure (Biocompatible, 3D Printed / Molded)</span>
                  </div>
                  <p style={{ margin: '0 0 8px 0' }}>
                    Compact, smooth white medical-grade enclosure pod featuring a rounded stadium profile, debossed oval groove channel, cyan emblem, and upper-right optical lens aperture. Sealed to IP67 standards to withstand clinical fluids and cleaning protocols.
                  </p>
                  <div className="divider" style={{ margin: '8px 0' }} />
                  <div className="data-row">
                    <span>Material:</span>
                    <span style={{ color: 'var(--text-primary)' }}>Medical White Polycarbonate / ABS</span>
                  </div>
                  <div className="data-row">
                    <span>Dimensions:</span>
                    <span style={{ color: 'var(--text-primary)' }}>48 mm × 26 mm × 5.2 mm</span>
                  </div>
                  <div className="data-row">
                    <span>Ergonomics:</span>
                    <span style={{ color: '#4ade80' }}>Low-profile flush skin contour</span>
                  </div>
                </div>
              )}

              {/* TAB 3: TEMPERATURE SENSOR */}
              {activeSensorComponent === 'temp' && (
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-primary)', fontWeight: 600, marginBottom: '6px' }}>
                    <Thermometer size={16} color="var(--accent-cyan)" />
                    <span>Temperature Sensor (Skin Surface Contact)</span>
                  </div>
                  <p style={{ margin: '0 0 8px 0' }}>
                    Sub-millimeter digital thermal probe contacting the skin surface adjacent to the IV catheter insertion point. Detects acute localized tissue cooling caused by the extravasation of room-temperature or refrigerated infusate.
                  </p>
                  <div className="divider" style={{ margin: '8px 0' }} />
                  <div className="data-row">
                    <span>Live Cannula Tip Temp:</span>
                    <span className="text-numeric" style={{ color: 'var(--accent-cyan)', fontWeight: 600 }}>{currentData?.temperature_c?.toFixed(2) || '33.00'} °C</span>
                  </div>
                  <div className="data-row">
                    <span>Resolution & Accuracy:</span>
                    <span style={{ color: 'var(--text-primary)' }}>0.0625 °C (12-bit ADC, ±0.1 °C)</span>
                  </div>
                  <div className="data-row">
                    <span>Detection Response:</span>
                    <span style={{ color: '#4ade80' }}>&lt; 15 seconds from onset</span>
                  </div>
                </div>
              )}

              {/* TAB 4: PRESSURE SENSOR */}
              {activeSensorComponent === 'pressure' && (
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-primary)', fontWeight: 600, marginBottom: '6px' }}>
                    <Gauge size={16} color="#c084fc" />
                    <span>Pressure Sensor (Under Patch - Compartment Strain)</span>
                  </div>
                  <p style={{ margin: '0 0 8px 0' }}>
                    Ultra-thin piezoresistive micro-disc positioned directly on the underside of the patch over the subcutaneous compartment. Quantifies mechanical skin tension and micro-strain as interstitial fluid pooling causes localized swelling and edema.
                  </p>
                  <div className="divider" style={{ margin: '8px 0' }} />
                  <div className="data-row">
                    <span>Live Compartment Pressure:</span>
                    <span className="text-numeric" style={{ color: '#c084fc', fontWeight: 600 }}>{currentData?.pressure_adc?.toFixed(0) || '1160'} ADC</span>
                  </div>
                  <div className="data-row">
                    <span>Transducer Principle:</span>
                    <span style={{ color: 'var(--text-primary)' }}>Piezoresistive semi-conductive polymer</span>
                  </div>
                  <div className="data-row">
                    <span>Dynamic Range:</span>
                    <span style={{ color: 'var(--text-primary)' }}>0.1 N to 10.0 N (0–50 mmHg compartment)</span>
                  </div>
                </div>
              )}

              {/* TAB 5: ESP32 SOC */}
              {activeSensorComponent === 'esp32' && (
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-primary)', fontWeight: 600, marginBottom: '6px' }}>
                    <Cpu size={16} color="#34d399" />
                    <span>ESP32 (Data Processing + WiFi/BLE)</span>
                  </div>
                  <p style={{ margin: '0 0 8px 0' }}>
                    Low-power dual-core microcontroller module with stamped RF metal shield. Buffers continuous sensor streams at 1.0 Hz in a 50s sliding window, executes on-device neural inference (Model 1 Causal TCN), and handles encrypted telemetry.
                  </p>
                  <div className="divider" style={{ margin: '8px 0' }} />
                  <div className="data-row">
                    <span>Neural Inference Model:</span>
                    <span style={{ color: 'var(--accent-cyan)' }}>Causal Dilated TCN (4,994 params)</span>
                  </div>
                  <div className="data-row">
                    <span>Inference Latency:</span>
                    <span className="text-numeric" style={{ color: '#4ade80' }}>5.01 ms / step</span>
                  </div>
                  <div className="data-row">
                    <span>Telemetry Protocol:</span>
                    <span style={{ color: 'var(--text-primary)' }}>BLE 4.2 Low Energy / 802.11 b/g/n</span>
                  </div>
                </div>
              )}

              {/* TAB 6: BATTERY */}
              {activeSensorComponent === 'battery' && (
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-primary)', fontWeight: 600, marginBottom: '6px' }}>
                    <Battery size={16} color="#fbbf24" />
                    <span>Li-Po Battery (Rechargeable Pouch Cell)</span>
                  </div>
                  <p style={{ margin: '0 0 8px 0' }}>
                    Ultra-compact rechargeable lithium-polymer pouch cell wrapped in protective polyimide Kapton tape with an integrated PCM safety circuit. Powers the sensor front-end and wireless telemetry throughout continuous clinical infusions.
                  </p>
                  <div className="divider" style={{ margin: '8px 0' }} />
                  <div className="data-row">
                    <span>Cell Voltage & Capacity:</span>
                    <span className="text-numeric" style={{ color: '#fbbf24' }}>3.7V Nominal, 180 mAh</span>
                  </div>
                  <div className="data-row">
                    <span>Remaining Charge:</span>
                    <span className="text-numeric" style={{ color: '#4ade80', fontWeight: 600 }}>94% (~28 hrs operating life)</span>
                  </div>
                  <div className="data-row">
                    <span>Charging Interface:</span>
                    <span style={{ color: 'var(--text-primary)' }}>Magnetic 2-pin pogo dock (5V / 100mA)</span>
                  </div>
                </div>
              )}

              {/* TAB 7: ADHESIVE BASE */}
              {activeSensorComponent === 'adhesive' && (
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-primary)', fontWeight: 600, marginBottom: '6px' }}>
                    <ShieldCheck size={16} color="#e2e8f0" />
                    <span>Silicone Adhesive Patch (Skin-Friendly Base)</span>
                  </div>
                  <p style={{ margin: '0 0 8px 0' }}>
                    Hypoallergenic medical silicone adhesive backing with a soft beige tone. Conforms dynamically to forearm skin contours, isolates the catheter site against external moisture and contamination, and secures the sensor probes flush against the tissue.
                  </p>
                  <div className="divider" style={{ margin: '8px 0' }} />
                  <div className="data-row">
                    <span>Skin Compatibility:</span>
                    <span style={{ color: '#4ade80' }}>ISO 10993 Biocompatible Silicone</span>
                  </div>
                  <div className="data-row">
                    <span>Breathability:</span>
                    <span style={{ color: 'var(--text-primary)' }}>Microporous MVTR &gt; 1200 g/m²/day</span>
                  </div>
                  <div className="data-row">
                    <span>Wear Duration:</span>
                    <span style={{ color: 'var(--text-primary)' }}>Up to 72 hours painless removal</span>
                  </div>
                </div>
              )}

              {/* TAB 8: BIOIMPEDANCE (PROPOSED EXTENSION) */}
              {activeSensorComponent === 'bio' && (
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-primary)', fontWeight: 600, marginBottom: '6px' }}>
                    <Waves size={16} color="#4ade80" />
                    <span>Tetrapolar Bioimpedance Array (Proposed Extension)</span>
                  </div>
                  <p style={{ margin: '0 0 8px 0' }}>
                    Four Ag/AgCl hydrogel contact pads providing 4-wire Kelvin admittance sensing across 20–500 kHz. Evaluated as an exploratory multimodal extension to track ionic conductivity shifts as infusate displaces tissue fluid.
                  </p>
                  <div style={{ background: 'rgba(74, 222, 128, 0.08)', border: '1px solid rgba(74, 222, 128, 0.25)', borderRadius: '4px', padding: '8px', margin: '8px 0', fontSize: '0.7rem' }}>
                    <strong style={{ color: '#4ade80' }}>Research Audit Finding:</strong> Paired clinical analysis achieves 97.2% retrospective accuracy. In this prototype, real-time edge detection relies primarily on continuous low-latency temperature and pressure dynamics.
                  </div>
                  <div className="data-row">
                    <span>Impedance Reading:</span>
                    <span className="text-numeric" style={{ color: '#4ade80', fontWeight: 600 }}>{currentData?.bioimpedance_ohm ? `${currentData.bioimpedance_ohm.toFixed(1)} Ω` : '580.0 Ω'}</span>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right: Consolidated Bedside Monitoring Card (Fits Above Fold) */}
        <div className="data-panel">
          <div className="glass-card" style={{ padding: '18px 20px' }}>
            
            {/* Header & Status */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
              <div>
                <div className="text-caption" style={{ color: 'var(--text-tertiary)', fontSize: '0.65rem' }}>REAL-TIME BED MONITORING</div>
                <div style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-primary)' }}>Infiltration Surveillance</div>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                {backendOnline !== null && (
                  <span style={{
                    fontSize: '0.62rem',
                    padding: '2px 6px',
                    borderRadius: '4px',
                    background: backendOnline ? 'rgba(74, 222, 128, 0.1)' : 'rgba(148, 163, 184, 0.1)',
                    color: backendOnline ? '#4ade80' : '#94a3b8',
                    border: `1px solid ${backendOnline ? 'rgba(74, 222, 128, 0.25)' : 'rgba(148, 163, 184, 0.25)'}`
                  }}>
                    {backendOnline ? 'Inference Engine Active' : 'Offline Trace Mode'}
                  </span>
                )}
                <div style={{
                  fontSize: '0.68rem',
                  padding: '3px 8px',
                  borderRadius: '4px',
                  fontWeight: 600,
                  background: statusBg !== 'transparent' ? statusBg : 'rgba(6, 182, 212, 0.12)',
                  color: statusBg !== 'transparent' ? statusTextColor : 'var(--accent-cyan)',
                  border: `1px solid ${statusBg !== 'transparent' ? 'transparent' : 'rgba(6, 182, 212, 0.25)'}`
                }}>
                  {statusText}
                </div>
              </div>
            </div>

            {/* Scenario & Model Selection Controls */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '14px', background: 'rgba(255,255,255,0.02)', padding: '10px', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
              <div>
                <div style={{ fontSize: '0.63rem', color: 'var(--text-tertiary)', marginBottom: '4px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Replay Scenario:
                </div>
                <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
                  {[
                    { id: 'infiltration', label: 'Infiltration' },
                    { id: 'normal', label: 'Normal Infusion' },
                    { id: 'stress', label: 'Vasoconstriction' },
                    { id: 'noisy', label: 'Motion Noise' }
                  ].map(sc => (
                    <button
                      key={sc.id}
                      onClick={() => setMonitorScenario(sc.id as any)}
                      style={{
                        padding: '3px 8px',
                        fontSize: '0.68rem',
                        borderRadius: '4px',
                        border: `1px solid ${monitorScenario === sc.id ? 'var(--accent-cyan)' : 'var(--border-subtle)'}`,
                        background: monitorScenario === sc.id ? 'rgba(6, 182, 212, 0.2)' : 'rgba(255,255,255,0.02)',
                        color: monitorScenario === sc.id ? 'var(--accent-cyan)' : 'var(--text-secondary)',
                        cursor: 'pointer',
                        fontWeight: monitorScenario === sc.id ? 600 : 400
                      }}
                    >
                      {sc.label}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <div style={{ fontSize: '0.63rem', color: 'var(--text-tertiary)', marginBottom: '4px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Active Inference Model:
                </div>
                <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
                  {[
                    { id: 'model1_tcn', label: 'Model 1 (TCN)' },
                    { id: 'model2_bio_tcn', label: 'Model 2 (Bio)' },
                    { id: 'rf', label: 'Random Forest' },
                    { id: 'logreg', label: 'LogReg' }
                  ].map(m => (
                    <button
                      key={m.id}
                      onClick={() => setMonitorModel(m.id)}
                      style={{
                        padding: '3px 8px',
                        fontSize: '0.68rem',
                        borderRadius: '4px',
                        border: `1px solid ${monitorModel === m.id ? 'var(--accent-cyan)' : 'var(--border-subtle)'}`,
                        background: monitorModel === m.id ? 'rgba(6, 182, 212, 0.2)' : 'rgba(255,255,255,0.02)',
                        color: monitorModel === m.id ? 'var(--accent-cyan)' : 'var(--text-secondary)',
                        cursor: 'pointer',
                        fontWeight: monitorModel === m.id ? 600 : 400
                      }}
                    >
                      {m.label}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Extravasation Probability & Progress */}
            <div style={{ marginBottom: '14px', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px' }}>
                  <div className="text-numeric value-massive" style={{ color: statusColor }}>
                    {probDisplay}<span style={{ fontSize: '1.3rem' }}>{probUnit}</span>
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', fontWeight: 500 }}>
                    {isBuffering ? 'Receptive Field Buffering' : isInferenceUnavailable ? 'Trace Replay (No Model)' : 'Extravasation Risk'}
                  </div>
                </div>
                {currentData?.pred_name && !isBuffering && (
                  <div style={{ fontSize: '0.68rem', color: 'var(--text-tertiary)', marginTop: '2px' }}>
                    Predicted State: <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{currentData.pred_name}</span>
                  </div>
                )}
                {isBuffering && (
                  <div style={{ fontSize: '0.68rem', color: '#94a3b8', marginTop: '2px' }}>
                    Accumulating 50s receptive field window ({currentData?.time_s ?? 0}s / 50s)
                  </div>
                )}
              </div>

              {/* Replay Timestamp Counter */}
              <div style={{ textAlign: 'right', fontSize: '0.72rem', color: 'var(--text-tertiary)' }}>
                <span>Replay Time: </span>
                <span className="text-numeric" style={{ color: 'var(--text-primary)', fontWeight: 600 }}>
                  {currentData?.time_s ?? 0}s
                </span>
                <span> / 180s</span>
              </div>
            </div>

            {/* Biosignal Live Telemetry Readouts */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '14px' }}>
              {/* Temperature */}
              <div style={{ background: 'rgba(255,255,255,0.03)', padding: '8px 12px', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
                <div className="data-row" style={{ marginBottom: '2px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Thermometer size={13} color="var(--accent-cyan)" />
                    <span style={{ fontSize: '0.72rem', color: 'var(--accent-cyan)', fontWeight: 600 }}>Local Temperature</span>
                  </div>
                  <span className="text-numeric" style={{ fontSize: '0.92rem', color: 'var(--accent-cyan)', fontWeight: 600 }}>
                    {currentData?.temperature_c !== undefined ? currentData.temperature_c.toFixed(2) : '33.00'} °C
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', color: 'var(--text-tertiary)' }}>
                  <span>Baseline: 33.00 °C</span>
                  <span>ΔT: {currentData?.temperature_c !== undefined ? (currentData.temperature_c - 33.0).toFixed(2) : '+0.00'} °C</span>
                </div>
              </div>

              {/* Pressure */}
              <div style={{ background: 'rgba(255,255,255,0.03)', padding: '8px 12px', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
                <div className="data-row" style={{ marginBottom: '2px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Gauge size={13} color="#c084fc" />
                    <span style={{ fontSize: '0.72rem', color: '#c084fc', fontWeight: 600 }}>Tissue Pressure</span>
                  </div>
                  <span className="text-numeric" style={{ fontSize: '0.92rem', color: '#c084fc', fontWeight: 600 }}>
                    {currentData?.pressure_adc !== undefined ? currentData.pressure_adc.toFixed(0) : '1160'} ADC
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', color: 'var(--text-tertiary)' }}>
                  <span>Baseline: 1160 ADC</span>
                  <span>ΔP: {currentData?.pressure_adc !== undefined ? (currentData.pressure_adc - 1160).toFixed(0) : '+0'} ADC</span>
                </div>
              </div>

              {/* Bioimpedance (if present) */}
              {currentData?.bioimpedance_ohm !== undefined && (
                <div style={{ background: 'rgba(255,255,255,0.03)', padding: '8px 12px', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
                  <div className="data-row" style={{ marginBottom: '2px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <Waves size={13} color="#4ade80" />
                      <span style={{ fontSize: '0.72rem', color: '#4ade80', fontWeight: 600 }}>Tissue Bioimpedance</span>
                    </div>
                    <span className="text-numeric" style={{ fontSize: '0.92rem', color: '#4ade80', fontWeight: 600 }}>
                      {currentData.bioimpedance_ohm.toFixed(1)} Ω
                    </span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', color: 'var(--text-tertiary)' }}>
                    <span>Baseline: 700.0 Ω</span>
                    <span>ΔZ: {(currentData.bioimpedance_ohm - 700.0).toFixed(1)} Ω</span>
                  </div>
                </div>
              )}
            </div>

            {/* Quick Action Controls */}
            <div style={{ display: 'flex', gap: '8px' }}>
              <button 
                className={`btn ${isSimulating ? 'active' : ''}`} 
                onClick={() => setIsSimulating(!isSimulating)}
                style={{ flex: 1.2, padding: '7px 10px', fontSize: '0.72rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}
              >
                {isSimulating ? <Pause size={13} /> : <Play size={13} />}
                <span>{isSimulating ? 'Pause Replay' : 'Run Offline Replay'}</span>
              </button>
              <button 
                className="btn" 
                onClick={restartReplay}
                style={{ padding: '7px 10px', fontSize: '0.72rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '4px' }}
                title="Restart Replay to t=0s"
              >
                <RotateCcw size={13} />
                <span>Restart</span>
              </button>
              <button 
                className="btn" 
                onClick={() => handleScrollTo('demo')}
                style={{ flex: 1.2, padding: '7px 10px', fontSize: '0.72rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '4px' }}
              >
                <span>Interactive Demo</span>
                <ArrowRight size={12} />
              </button>
            </div>

          </div>
        </div>
      </div>

      {/* Restored Lower-Page Real-Time Biosignal Charts */}
      <div className="section-wrapper" style={{ paddingTop: '28px', paddingBottom: '52px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '20px', flexWrap: 'wrap', gap: '12px' }}>
          <div>
            <div className="text-caption" style={{ color: 'var(--accent-cyan)', marginBottom: '4px', letterSpacing: '0.12em' }}>
              REAL-TIME BIOSIGNAL TELEMETRY
            </div>
            <div className="title-primary" style={{ fontSize: '1.25rem' }}>
              LOCAL SKIN TEMPERATURE & TISSUE PRESSURE SENSOR TRACES
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px', fontSize: '0.75rem', color: 'var(--text-tertiary)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--accent-cyan)' }}></span>
              <span>Skin Temperature (°C)</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#c084fc' }}></span>
              <span>Tissue Pressure (ADC)</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: isSimulating ? '#4ade80' : 'var(--text-tertiary)' }}></span>
              <span>{isSimulating ? 'Live Telemetry (1 Hz)' : 'Replay Paused'}</span>
            </div>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(460px, 1fr))', gap: '24px' }}>
          {/* Chart 1: Local Skin Temperature */}
          <div className="glass-card" style={{ padding: '22px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '2px' }}>
                  <Thermometer size={16} color="var(--accent-cyan)" />
                  <span style={{ fontWeight: 600, fontSize: '0.9rem', color: 'var(--text-primary)' }}>Local Skin Temperature</span>
                </div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-tertiary)' }}>
                  Infusion site contact probe (Baseline ~33.0 °C)
                </div>
              </div>
              <div style={{ textAlign: 'right' }}>
                <span className="text-numeric" style={{ fontSize: '1.35rem', fontWeight: 700, color: 'var(--accent-cyan)' }}>
                  {currentData?.temperature_c !== undefined ? currentData.temperature_c.toFixed(2) : '33.00'}
                </span>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-tertiary)', marginLeft: '4px' }}>°C</span>
              </div>
            </div>

            <div style={{ height: '220px', width: '100%' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={dataHistory} margin={{ top: 12, right: 24, left: 0, bottom: 20 }}>
                  <CartesianGrid stroke="rgba(255,255,255,0.05)" strokeDasharray="3 3" vertical={false} />
                  <XAxis 
                    dataKey="time_s" 
                    type="number"
                    domain={[0, 180]} 
                    tickCount={10}
                    tickFormatter={(v) => `${v}s`} 
                    stroke="var(--text-tertiary)"
                    tick={{ fill: 'var(--text-tertiary)', fontSize: 11 }}
                    axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
                    tickLine={false}
                  />
                  <YAxis 
                    domain={['dataMin - 0.5', 'dataMax + 0.5']} 
                    tickFormatter={(v) => `${Number(v).toFixed(1)}°`} 
                    stroke="var(--text-tertiary)"
                    tick={{ fill: 'var(--text-tertiary)', fontSize: 11 }}
                    axisLine={false}
                    tickLine={false}
                    width={44}
                  />
                  <Tooltip 
                    contentStyle={{ background: '#0a0d14', border: '1px solid var(--border-active)', borderRadius: '6px', fontSize: '0.75rem', color: '#fff' }}
                    formatter={(val: any) => [`${Number(val).toFixed(2)} °C`, 'Temperature']}
                    labelFormatter={(label: any) => `Replay Time: ${label}s`}
                  />
                  <ReferenceLine y={33.0} stroke="rgba(255,255,255,0.18)" strokeDasharray="4 4" label={{ value: 'Baseline 33.0°C', position: 'insideTopLeft', fill: 'var(--text-tertiary)', fontSize: 10 }} />
                  {currentData?.time_s !== undefined && (
                    <ReferenceLine x={currentData.time_s} stroke="#ffffff" strokeDasharray="2 2" strokeOpacity={0.6} />
                  )}
                  <Line 
                    type="monotone" 
                    dataKey="temperature_c" 
                    stroke="var(--accent-cyan)" 
                    strokeWidth={2.2} 
                    dot={false} 
                    isAnimationActive={false} 
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '12px', fontSize: '0.7rem', color: 'var(--text-tertiary)' }}>
              <span>Infusion cooling / physiological vasoconstriction tracking</span>
              <span>Window: 0–180s (1 Hz sampling)</span>
            </div>
          </div>

          {/* Chart 2: Compartment Tissue Pressure */}
          <div className="glass-card" style={{ padding: '22px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '2px' }}>
                  <Gauge size={16} color="#c084fc" />
                  <span style={{ fontWeight: 600, fontSize: '0.9rem', color: 'var(--text-primary)' }}>Compartment Tissue Pressure</span>
                </div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-tertiary)' }}>
                  Piezo-resistive pressure sensor (Baseline ~1160 ADC)
                </div>
              </div>
              <div style={{ textAlign: 'right' }}>
                <span className="text-numeric" style={{ fontSize: '1.35rem', fontWeight: 700, color: '#c084fc' }}>
                  {currentData?.pressure_adc !== undefined ? Math.round(currentData.pressure_adc) : '1160'}
                </span>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-tertiary)', marginLeft: '4px' }}>ADC</span>
              </div>
            </div>

            <div style={{ height: '220px', width: '100%' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={dataHistory} margin={{ top: 12, right: 24, left: 0, bottom: 20 }}>
                  <CartesianGrid stroke="rgba(255,255,255,0.05)" strokeDasharray="3 3" vertical={false} />
                  <XAxis 
                    dataKey="time_s" 
                    type="number"
                    domain={[0, 180]} 
                    tickCount={10}
                    tickFormatter={(v) => `${v}s`} 
                    stroke="var(--text-tertiary)"
                    tick={{ fill: 'var(--text-tertiary)', fontSize: 11 }}
                    axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
                    tickLine={false}
                  />
                  <YAxis 
                    domain={['dataMin - 20', 'dataMax + 20']} 
                    tickFormatter={(v) => `${Math.round(Number(v))}`} 
                    stroke="var(--text-tertiary)"
                    tick={{ fill: 'var(--text-tertiary)', fontSize: 11 }}
                    axisLine={false}
                    tickLine={false}
                    width={48}
                  />
                  <Tooltip 
                    contentStyle={{ background: '#0a0d14', border: '1px solid var(--border-active)', borderRadius: '6px', fontSize: '0.75rem', color: '#fff' }}
                    formatter={(val: any) => [`${Math.round(Number(val))} ADC`, 'Pressure']}
                    labelFormatter={(label: any) => `Replay Time: ${label}s`}
                  />
                  <ReferenceLine y={1160} stroke="rgba(255,255,255,0.18)" strokeDasharray="4 4" label={{ value: 'Baseline 1160 ADC', position: 'insideTopLeft', fill: 'var(--text-tertiary)', fontSize: 10 }} />
                  {currentData?.time_s !== undefined && (
                    <ReferenceLine x={currentData.time_s} stroke="#ffffff" strokeDasharray="2 2" strokeOpacity={0.6} />
                  )}
                  <Line 
                    type="monotone" 
                    dataKey="pressure_adc" 
                    stroke="#c084fc" 
                    strokeWidth={2.2} 
                    dot={false} 
                    isAnimationActive={false} 
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '12px', fontSize: '0.7rem', color: 'var(--text-tertiary)' }}>
              <span>Subcutaneous swelling & tissue compartment distension</span>
              <span>Window: 0–180s (1 Hz sampling)</span>
            </div>
          </div>
        </div>
      </div>
    </section>

      {/* SECTION 02: SIMULATION DEMO & BENCHMARK */}
      <section id="demo" style={{ borderTop: '1px solid var(--border-subtle)', background: '#020305' }}>
        <div className="section-wrapper">
          <div style={{ marginBottom: '20px' }}>
            <div className="title-primary" style={{ fontSize: '1.2rem', marginBottom: '4px' }}>SIMULATION DEMO & BENCHMARK</div>
            <div className="text-secondary" style={{ fontSize: '0.82rem' }}>
              Configure physiological scenarios and run real-time inference across Model 1 (Causal TCN), Model 2 (Bio-TCN), and baselines.
            </div>
          </div>
          <SimulationDemo />
        </div>
      </section>

      {/* SECTION 03: PATCH ARCHITECTURE */}
      <section id="patch" style={{ borderTop: '1px solid var(--border-subtle)', background: '#040508' }}>
        <div className="section-wrapper">
          <div style={{ marginBottom: '28px' }}>
            <div className="text-caption" style={{ color: 'var(--accent-cyan)', marginBottom: '4px' }}>Physical Device Architecture</div>
            <div className="title-primary" style={{ fontSize: '1.25rem', marginBottom: '6px' }}>INTEGRATED WEARABLE SENSING PATCH</div>
            <div className="text-secondary" style={{ fontSize: '0.82rem', maxWidth: '780px', lineHeight: 1.5 }}>
              The multi-sensor skin patch integrates localized thermal tracking, micro-strain compartment pressure measurement, and multi-frequency bioimpedance inside a biocompatible molded enclosure attached to a soft skin-compatible adhesive base.
            </div>
          </div>

          {/* 5 Exploded Hardware Layers (Reflecting Reference Image 2) */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '16px', marginBottom: '32px' }}>
            
            {/* Layer 1: Biocompatible Enclosure & Adhesive Dressing */}
            <div className="glass-card" style={{ padding: '18px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                <span style={{ fontSize: '0.68rem', color: 'var(--text-primary)', fontWeight: 600, letterSpacing: '0.08em' }}>LAYER 1: ENCLOSURE & BASE</span>
                <ShieldCheck size={16} color="var(--accent-cyan)" />
              </div>
              <div style={{ fontSize: '0.9rem', fontWeight: 600, marginBottom: '4px' }}>Molded Pod &amp; Silicone Patch</div>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                Compact white polycarbonate enclosure pod (IP67 sealed) mounted flush on a soft, skin-compatible hypoallergenic silicone adhesive dressing.
              </p>
            </div>

            {/* Layer 2: Micro Temperature Sensor */}
            <div className="glass-card" style={{ padding: '18px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                <span style={{ fontSize: '0.68rem', color: 'var(--accent-cyan)', fontWeight: 600, letterSpacing: '0.08em' }}>LAYER 2: TEMPERATURE PROBE</span>
                <Thermometer size={16} color="var(--accent-cyan)" />
              </div>
              <div style={{ fontSize: '0.9rem', fontWeight: 600, marginBottom: '4px' }}>Skin Surface Contact Sensor</div>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                Sub-millimeter thermal probe needle positioned directly at the catheter puncture point. Detects acute localized tissue cooling as cold infusate extravasates.
              </p>
            </div>

            {/* Layer 3: Piezoresistive Strain Pad */}
            <div className="glass-card" style={{ padding: '18px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                <span style={{ fontSize: '0.68rem', color: '#c084fc', fontWeight: 600, letterSpacing: '0.08em' }}>LAYER 3: STRAIN TRANSDUCER</span>
                <Gauge size={16} color="#c084fc" />
              </div>
              <div style={{ fontSize: '0.9rem', fontWeight: 600, marginBottom: '4px' }}>Under-Patch Pressure Disc</div>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                Piezoresistive sensing disc mounted beneath the patch over the subcutaneous compartment. Quantifies mechanical skin tension and swelling micro-strain.
              </p>
            </div>

            {/* Layer 4: Embedded ESP32 SoC & Li-Po Battery */}
            <div className="glass-card" style={{ padding: '18px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                <span style={{ fontSize: '0.68rem', color: '#34d399', fontWeight: 600, letterSpacing: '0.08em' }}>LAYER 4: EDGE SOC &amp; POWER</span>
                <Cpu size={16} color="#34d399" />
              </div>
              <div style={{ fontSize: '0.9rem', fontWeight: 600, marginBottom: '4px' }}>ESP32 + 3.7V Li-Po Cell</div>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                Dual-core Xtensa module with RF shield buffering a 50s sliding window at 1.0 Hz, paired with an ultra-slim 180 mAh rechargeable Li-Po battery.
              </p>
            </div>

            {/* Layer 5: Bioimpedance Electrodes (Proposed Extension) */}
            <div className="glass-card" style={{ padding: '18px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                <span style={{ fontSize: '0.68rem', color: '#4ade80', fontWeight: 600, letterSpacing: '0.08em' }}>LAYER 5: BIOIMPEDANCE (PROPOSED)</span>
                <Waves size={16} color="#4ade80" />
              </div>
              <div style={{ fontSize: '0.9rem', fontWeight: 600, marginBottom: '4px' }}>Tetrapolar Contact Array</div>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                4 Ag/AgCl hydrogel contacts providing 4-wire Kelvin sensing across 20–500 kHz. Proposed extension based on clinical study calibration.
              </p>
            </div>

          </div>

          {/* Data Flow Pipeline */}
          <div className="glass-card" style={{ padding: '20px', marginBottom: '24px' }}>
            <div className="text-caption" style={{ color: 'var(--accent-cyan)', marginBottom: '12px' }}>
              Signal Conditioning &amp; Edge Processing Pipeline
            </div>
            
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
              gap: '12px',
              alignItems: 'center'
            }}>
              <div style={{ background: 'rgba(255,255,255,0.02)', padding: '12px', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: '0.65rem', color: 'var(--text-tertiary)', marginBottom: '2px' }}>01. SENSING</div>
                <div style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--accent-cyan)' }}>Multi-Sensor Patch</div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Temp, Pressure, Bioimpedance</div>
              </div>

              <div style={{ background: 'rgba(255,255,255,0.02)', padding: '12px', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: '0.65rem', color: 'var(--text-tertiary)', marginBottom: '2px' }}>02. CONDITIONING</div>
                <div style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-primary)' }}>Analog Front-End</div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>1-Wire bus + 10kΩ divider + AFE filter</div>
              </div>

              <div style={{ background: 'rgba(255,255,255,0.02)', padding: '12px', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: '0.65rem', color: 'var(--text-tertiary)', marginBottom: '2px' }}>03. EDGE BUFFER</div>
                <div style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-primary)' }}>50s Ring Buffer</div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>1.0 Hz sampling with normalization</div>
              </div>

              <div style={{ background: 'rgba(255,255,255,0.02)', padding: '12px', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: '0.65rem', color: 'var(--text-tertiary)', marginBottom: '2px' }}>04. NEURAL INFERENCE</div>
                <div style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--accent-cyan)' }}>Causal Dilated TCN</div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>5.01 ms CPU latency, 61s receptive field</div>
              </div>

              <div style={{ background: 'rgba(255,255,255,0.02)', padding: '12px', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: '0.65rem', color: 'var(--text-tertiary)', marginBottom: '2px' }}>05. ALARM</div>
                <div style={{ fontSize: '0.78rem', fontWeight: 600, color: '#f87171' }}>Dual-Stage Threshold</div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Sustained alarm on ≥2 consecutive windows</div>
              </div>
            </div>
          </div>


        </div>
      </section>

      {/* SECTION 04: ANALYSIS */}
      <section id="analysis" style={{ borderTop: '1px solid var(--border-subtle)', background: '#030407' }}>
        <div className="section-wrapper">
          <div style={{ marginBottom: '24px' }}>
            <div className="text-caption" style={{ color: 'var(--accent-cyan)', marginBottom: '4px' }}>Empirical Benchmarks &amp; Analysis</div>
            <div className="title-primary" style={{ fontSize: '1.25rem', marginBottom: '6px' }}>RESEARCH RESULTS &amp; METRIC COMPARISONS</div>
            <div className="text-secondary" style={{ fontSize: '0.82rem', maxWidth: '800px', lineHeight: 1.5 }}>
              Comprehensive performance comparison across progressive model architectures, autonomic confounder stress rejection, and clinical bioimpedance evaluation.
            </div>
          </div>

          {/* Analysis View Tabs */}
          <div style={{ display: 'flex', gap: '8px', marginBottom: '20px' }}>
            <button
              className={`btn ${activeAnalysisTab === 'models' ? 'active' : ''}`}
              onClick={() => setActiveAnalysisTab('models')}
              style={{ fontSize: '0.75rem' }}
            >
              Model Progression (1–4)
            </button>
            <button
              className={`btn ${activeAnalysisTab === 'confounder' ? 'active' : ''}`}
              onClick={() => setActiveAnalysisTab('confounder')}
              style={{ fontSize: '0.75rem' }}
            >
              Confounder Rejection
            </button>
            <button
              className={`btn ${activeAnalysisTab === 'discrepancy' ? 'active' : ''}`}
              onClick={() => setActiveAnalysisTab('discrepancy')}
              style={{ fontSize: '0.75rem' }}
            >
              Clinical Bioimpedance Audit
            </button>
          </div>

          {/* Tab 1: Progressive Models Table */}
          {activeAnalysisTab === 'models' && (
            <div>
              <div className="glass-card" style={{ padding: 0, overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.78rem' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-tertiary)' }}>
                      <th style={{ padding: '14px 18px', textAlign: 'left', fontWeight: 500 }}>Model Architecture</th>
                      <th style={{ padding: '14px', textAlign: 'left', fontWeight: 500 }}>Input Channels</th>
                      <th style={{ padding: '14px', textAlign: 'right', fontWeight: 500 }}>Parameters</th>
                      <th style={{ padding: '14px', textAlign: 'right', fontWeight: 500 }}>CPU Latency</th>
                      <th style={{ padding: '14px', textAlign: 'right', fontWeight: 500 }}>Macro-F1</th>
                      <th style={{ padding: '14px', textAlign: 'right', fontWeight: 500 }}>Bal. Acc</th>
                      <th style={{ padding: '14px 18px', textAlign: 'right', fontWeight: 500 }}>Confounder FPR</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.keys(MODEL_REGISTRY).map(k => {
                      const m = MODEL_REGISTRY[k];
                      return (
                        <tr key={m.id} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                          <td style={{ padding: '14px 18px', fontWeight: 600, color: 'var(--text-primary)' }}>
                            {m.name}
                          </td>
                          <td style={{ padding: '14px', color: 'var(--text-secondary)', fontSize: '0.72rem' }}>
                            {m.inputModalities.join(' + ')}
                          </td>
                          <td className="text-numeric" style={{ padding: '14px', textAlign: 'right' }}>
                            {m.totalParameters.toLocaleString()}
                          </td>
                          <td className="text-numeric" style={{ padding: '14px', textAlign: 'right', color: 'var(--accent-cyan)' }}>
                            {m.cpuLatencyMs} ms
                          </td>
                          <td className="text-numeric" style={{ padding: '14px', textAlign: 'right', fontWeight: 600, color: m.macroF1 ? '#4ade80' : '#94a3b8' }}>
                            {m.macroF1 !== null ? m.macroF1.toFixed(4) : '—'}
                          </td>
                          <td className="text-numeric" style={{ padding: '14px', textAlign: 'right' }}>
                            {m.balancedAccuracy !== null ? m.balancedAccuracy.toFixed(4) : '—'}
                          </td>
                          <td className="text-numeric" style={{ padding: '14px 18px', textAlign: 'right', color: m.confounderFpr === 0 ? '#4ade80' : m.confounderFpr ? '#f87171' : 'var(--text-secondary)' }}>
                            {m.confounderFpr !== null ? (m.confounderFpr === 0 ? '0.00%' : `${(m.confounderFpr * 100).toFixed(2)}%`) : '—'}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <div style={{ marginTop: '12px', padding: '10px 14px', background: 'rgba(255,255,255,0.02)', borderRadius: '6px', border: '1px solid var(--border-subtle)', fontSize: '0.72rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                <strong style={{ color: 'var(--text-primary)' }}>Evaluation Protocol &amp; Provenance:</strong> Models 1–3 were trained and evaluated using strict trial-disjoint splits and leak-free forward validation. Model 4 (Age) implements modular age embeddings, but empirical metrics are withheld (&ldquo;—&rdquo;) due to absent individual patient age linkage in public data. Confounder FPR measures the false positive rate on 432 sliding test windows under simulated vasoconstriction and thermal drift.
              </div>
            </div>
          )}

          {/* Tab 2: Confounder Rejection Deep Dive */}
          {activeAnalysisTab === 'confounder' && (
            <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '20px' }}>
              <div className="glass-card" style={{ padding: '20px' }}>
                <div className="text-caption" style={{ color: 'var(--accent-cyan)', marginBottom: '10px' }}>
                  The Thermal Confounder: Peripheral Vasoconstriction
                </div>
                <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: 1.5, marginBottom: '14px' }}>
                  Patient stress, cold ambient exposure, or hypovolemia triggers cutaneous arteriolar constriction. This cools skin by <strong>1.0°C to 2.5°C</strong>, mimicking the thermal signature of extravasating IV fluid.
                </p>
                <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: 1.5, marginBottom: '16px' }}>
                  Single-modality temperature threshold monitoring suffers from a <strong>23.6% False Positive Rate</strong> (102 false alarms out of 432 sliding test windows in benchmark challenge data). By fusing local tissue pressure (Model 1 Causal TCN), false alarms drop to <strong>1.15%</strong>. Fusing bioimpedance admittance tracking (Model 2) or systemic autonomic stress representation (Model 3) reduces false alarms to <strong>0.00%</strong>.
                </p>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px' }}>
                  <div style={{ background: 'rgba(239, 68, 68, 0.08)', border: '1px solid rgba(239, 68, 68, 0.25)', padding: '12px', borderRadius: '6px' }}>
                    <div style={{ fontSize: '0.65rem', color: '#f87171', fontWeight: 600 }}>TEMP THRESHOLD</div>
                    <div className="text-numeric" style={{ fontSize: '1.3rem', color: '#f87171', marginTop: '3px' }}>23.6%</div>
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-tertiary)' }}>102/432 False Alarms</div>
                  </div>
                  <div style={{ background: 'rgba(6, 182, 212, 0.08)', border: '1px solid rgba(6, 182, 212, 0.25)', padding: '12px', borderRadius: '6px' }}>
                    <div style={{ fontSize: '0.65rem', color: 'var(--accent-cyan)', fontWeight: 600 }}>MODEL 1 (T + P)</div>
                    <div className="text-numeric" style={{ fontSize: '1.3rem', color: 'var(--accent-cyan)', marginTop: '3px' }}>1.15%</div>
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-tertiary)' }}>Pressure Rejection</div>
                  </div>
                  <div style={{ background: 'rgba(74, 222, 128, 0.08)', border: '1px solid rgba(74, 222, 128, 0.25)', padding: '12px', borderRadius: '6px' }}>
                    <div style={{ fontSize: '0.65rem', color: '#4ade80', fontWeight: 600 }}>MODEL 2 &amp; 3</div>
                    <div className="text-numeric" style={{ fontSize: '1.3rem', color: '#4ade80', marginTop: '3px' }}>0.00%</div>
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-tertiary)' }}>Zero False Alarms</div>
                  </div>
                </div>
              </div>

              <div className="glass-card" style={{ padding: '20px' }}>
                <div className="text-caption" style={{ color: 'var(--text-secondary)', marginBottom: '10px' }}>
                  Physiological Signatures
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  <div style={{ padding: '10px 12px', background: 'rgba(255,255,255,0.02)', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
                    <div style={{ fontSize: '0.78rem', fontWeight: 600, color: '#f87171' }}>True Extravasation</div>
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
                      Cooling <strong>ΔT &lt; -1.5°C</strong>, compartment pressure spike <strong>ΔP &gt; +60 ADC</strong>, and tissue impedance drop <strong>ΔZ &lt; -25%</strong>.
                    </div>
                  </div>
                  <div style={{ padding: '10px 12px', background: 'rgba(255,255,255,0.02)', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
                    <div style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--accent-cyan)' }}>Vasoconstriction Cooling</div>
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
                      Cutaneous cooling <strong>ΔT &lt; -1.2°C</strong>, but compartment pressure remains flat <strong>ΔP ≈ 0</strong> and impedance <strong>ΔZ ≈ 0</strong>.
                    </div>
                  </div>
                  <div style={{ padding: '10px 12px', background: 'rgba(255,255,255,0.02)', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
                    <div style={{ fontSize: '0.78rem', fontWeight: 600, color: '#c084fc' }}>Motion Artifacts</div>
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
                      Transient mechanical spikes rejected by the 50s causal temporal window and sustained two-window alarm rule.
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Tab 3: Clinical Discrepancy Audit */}
          {activeAnalysisTab === 'discrepancy' && (
            <div className="glass-card" style={{ padding: '20px' }}>
              <div className="text-caption" style={{ color: '#facc15', marginBottom: '10px' }}>
                Clinical Bioimpedance Analysis: Jeong et al. (PLOS ONE 2019, N=18)
              </div>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: 1.5, marginBottom: '14px' }}>
                Retrospective evaluation using paired relative change <code>(After - Before) / |Before|</code> achieved <strong>0.9722 Balanced Accuracy</strong> across 18 patients. However, evaluating raw impedance features on held-out patients yielded <strong>0.7222 (Logistic Regression)</strong> and <strong>0.7500 (Random Forest)</strong> due to substantial inter-subject baseline variance (420–980 Ω).
              </p>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '14px' }}>
                <div style={{ background: 'rgba(255,255,255,0.02)', padding: '14px', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
                  <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '3px' }}>
                    1. Retrospective Paired Analysis (0.9722)
                  </div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                    Requires comparing post-infiltration measurements against each patient&apos;s pre-infusion baseline. Valid for retrospective verification, but uncalibrated incoming patients lack this reference.
                  </div>
                </div>

                <div style={{ background: 'rgba(255,255,255,0.02)', padding: '14px', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
                  <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '3px' }}>
                    2. Uncalibrated Single-Observation (0.7222)
                  </div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                    Raw multi-frequency impedance features evaluated under strict patient-quarantined leave-one-out cross-validation without baseline referencing.
                  </div>
                </div>

                <div style={{ background: 'rgba(74, 222, 128, 0.05)', padding: '14px', borderRadius: '6px', border: '1px solid rgba(74, 222, 128, 0.25)' }}>
                  <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#4ade80', marginBottom: '3px' }}>
                    3. Baseline Calibration Strategy (Under Consideration)
                  </div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                    An engineering calibration strategy under consideration: the edge firmware establishes a stable 60-second baseline window at infusion onset, enabling patient-specific relative impedance tracking during continuous monitoring.
                  </div>
                </div>
              </div>
            </div>
          )}

        </div>
      </section>

      {/* SECTION 05: HARDWARE & EDGE STREAM */}
      <section id="hardware" style={{ borderTop: '1px solid var(--border-subtle)', background: '#020305' }}>
        <div className="section-wrapper">
          <div style={{ display: 'grid', gridTemplateColumns: '340px 1fr', gap: '28px' }}>
            
            {/* Left: Device Controls & Pinout */}
            <div>
              <div className="title-primary" style={{ marginBottom: '4px' }}>HARDWARE INTERFACE</div>
              <div className="text-secondary" style={{ marginBottom: '18px', fontSize: '0.8rem' }}>
                ESP32 edge streaming and telemetry via Serial UART (115200 baud).
              </div>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '20px' }}>
                <button className={`btn ${serialConnected ? 'active' : ''}`} onClick={connectSerial}>
                  {serialConnected ? 'Connected: ESP32 (115200)' : 'Connect ESP32 (Web Serial)'}
                </button>
                <button className={`btn ${isSimulating ? 'active' : ''}`} onClick={() => setIsSimulating(!isSimulating)} disabled={serialConnected}>
                  {isSimulating ? 'Pause Replay Stream' : 'Run Offline Simulator Stream'}
                </button>
              </div>

              {/* Pinout Specification Table */}
              <div className="glass-card" style={{ padding: '14px' }}>
                <div className="text-caption" style={{ color: 'var(--accent-cyan)', marginBottom: '8px' }}>ESP32 Pinout Mapping</div>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', display: 'flex', flexDirection: 'column', gap: '5px' }}>
                  <div className="data-row">
                    <span>GPIO 4 (1-Wire):</span>
                    <span className="text-numeric" style={{ color: 'var(--text-primary)' }}>DS18B20 Temp Probe</span>
                  </div>
                  <div className="data-row">
                    <span>GPIO 34 (ADC1_CH6):</span>
                    <span className="text-numeric" style={{ color: 'var(--text-primary)' }}>FSR 402 Pressure</span>
                  </div>
                  <div className="data-row">
                    <span>GPIO 21 / 22 (I2C):</span>
                    <span className="text-numeric" style={{ color: 'var(--text-primary)' }}>AD5940 Bioimpedance</span>
                  </div>
                  <div className="data-row">
                    <span>UART0 (TX0/RX0):</span>
                    <span className="text-numeric" style={{ color: 'var(--text-primary)' }}>115200 Baud CSV Stream</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Right: Live Terminal Event Stream */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                <div className="text-caption">Edge Serial Event Log (1 Hz Telemetry)</div>
                <button 
                  onClick={() => setLogs([])}
                  style={{ background: 'transparent', border: 'none', color: 'var(--text-tertiary)', fontSize: '0.68rem', cursor: 'pointer' }}
                >
                  Clear Terminal
                </button>
              </div>

              <div className="glass-card" style={{ height: '280px', overflowY: 'auto', fontFamily: 'JetBrains Mono, monospace', fontSize: '0.72rem', background: '#000000', border: '1px solid rgba(255,255,255,0.08)' }}>
                {logs.map((l, i) => (
                  <div key={i} style={{ marginBottom: '6px', color: l.msg.includes('ALERT') ? 'var(--accent-red)' : l.msg.includes('Connected') ? '#4ade80' : 'var(--text-secondary)' }}>
                    <span style={{ opacity: 0.4, marginRight: '12px' }}>[{l.time}]</span>
                    {l.msg}
                  </div>
                ))}
                {logs.length === 0 && (
                  <div style={{ color: 'var(--text-tertiary)', padding: '14px' }}>
                    Waiting for data stream... Click &ldquo;Run Offline Simulator Stream&rdquo; or connect physical hardware.
                  </div>
                )}
              </div>
            </div>

          </div>
        </div>
      </section>

    </div>
  );
}

export default App;
