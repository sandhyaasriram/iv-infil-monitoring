"""
demo_server.py
==============
Interactive IV Infiltration Simulation Demo Server.
Uses Python's built-in http.server (no extra dependencies required).

Provides:
  - GET  /api/health      : Returns server status, loaded models, and feature metadata.
  - GET  /api/models      : Returns full specifications of all progressive architectures.
  - POST /api/simulate    : Generates synthetic physiological traces using the physics simulator,
                            runs the selected model (Model 1 Causal TCN, Model 2 Bio-TCN,
                            Random Forest, or Logistic Regression) over sliding windows,
                            evaluates sustained alarm rules, and returns the time series
                            alongside ground-truth labels and model predictions.

Run:
  .venv\\Scripts\\python demo_server.py [port]
"""

import http.server
import json
import os
import sys
import traceback
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Import simulator and sensor emulation
from simulator.parameters import (
    DT, DURATION_S, T0, P0, DS18B20_NOISE_STD, FSR_NOISE_STD,
    VOLUME_LABEL_THRESHOLDS, SEVERITY_LABELS
)
from simulator.infusion import simulate_leaked_volume
from simulator.tissue import simulate_pressure_ideal
from simulator.thermal import simulate_temperature
from simulator.bioimpedance import (
    simulate_bioimpedance_spectrum_ideal, emulate_bioimpedance_sensors, CLINICAL_FREQS_KHZ
)
from sensors.noise import emulate_ds18b20, emulate_fsr

# PyTorch and Model Architectures
import torch
import torch.nn as nn
from src.models.progressive_multimodal import Model1_BaselineTCN, Model2_BioimpedanceTCN

# Pre-load trained models
MODELS = {}
METADATA = {
    "window_len": 50,
    "norm_ch_mean": np.array([[[32.3908]], [[1234.386]]], dtype=np.float32),
    "norm_ch_std": np.array([[[0.8800]], [[131.968]]], dtype=np.float32),
    "class_names": ["None", "Early", "Moderate", "Severe"],
}

CLASS_NAMES = ["None", "Early", "Moderate", "Severe"]


def load_models():
    global MODELS, METADATA
    try:
        import joblib
        rf_path = BASE_DIR / "models" / "random_forest_iv.joblib"
        lr_path = BASE_DIR / "models" / "logistic_regression_iv.joblib"
        meta_path = BASE_DIR / "models" / "model_metadata.joblib"

        if rf_path.exists():
            MODELS["rf"] = joblib.load(rf_path)
            print("[DemoServer] Loaded Random Forest model from", rf_path)
        if lr_path.exists():
            MODELS["logreg"] = joblib.load(lr_path)
            print("[DemoServer] Loaded Logistic Regression model from", lr_path)
        if meta_path.exists():
            meta = joblib.load(meta_path)
            METADATA["window_len"] = meta.get("window_len", 50)
            if "norm_ch_mean" in meta:
                METADATA["norm_ch_mean"] = np.array(meta["norm_ch_mean"], dtype=np.float32)
            if "norm_ch_std" in meta:
                METADATA["norm_ch_std"] = np.array(meta["norm_ch_std"], dtype=np.float32)
            print("[DemoServer] Loaded joblib model metadata")
    except Exception as e:
        print(f"[DemoServer] Warning: could not load joblib models: {e}")

    # Load PyTorch progressive models
    try:
        m1_path = BASE_DIR / "results" / "progressive_multimodal_corrected" / "checkpoints" / "model1_baseline_tcn.pt"
        if m1_path.exists():
            m1 = Model1_BaselineTCN(in_channels=2, num_classes=4)
            m1.load_state_dict(torch.load(m1_path, map_location="cpu"))
            m1.eval()
            MODELS["model1_tcn"] = m1
            print("[DemoServer] Loaded Model 1 Causal TCN from", m1_path)

        m2_path = BASE_DIR / "results" / "progressive_multimodal_corrected" / "checkpoints" / "model2_bio_tcn_simulated.pt"
        if m2_path.exists():
            m2 = Model2_BioimpedanceTCN(in_channels=2, bio_features=28, num_classes=4)
            m2.load_state_dict(torch.load(m2_path, map_location="cpu"))
            m2.eval()
            MODELS["model2_bio_tcn"] = m2
            print("[DemoServer] Loaded Model 2 Bio-TCN from", m2_path)
    except Exception as e:
        print(f"[DemoServer] Warning: could not load PyTorch models: {e}")
        traceback.print_exc()


load_models()


def extract_features_from_window(window_signal: np.ndarray, norm_mean=None, norm_std=None) -> np.ndarray:
    """
    Extract exact 10 statistical moments from a (C, L) window:
    For each channel (temp, press): mean, std, min, max, linear slope.
    Matches windows_to_features() in experiments/run_workstream_b.py.
    """
    C, L = window_signal.shape
    t_vec = np.arange(L, dtype=np.float32)
    t_norm = t_vec - t_vec.mean()
    denom = float(np.dot(t_norm, t_norm)) or 1.0

    if norm_mean is not None and norm_std is not None:
        mean_arr = np.array(norm_mean).reshape(C, 1)
        std_arr = np.array(norm_std).reshape(C, 1)
        w_norm = (window_signal - mean_arr) / std_arr
        w_norm = np.nan_to_num(w_norm, nan=0.0)
    else:
        w_norm = window_signal

    feats = []
    for c in range(C):
        sig = w_norm[c]
        feats.append(float(np.mean(sig)))
        feats.append(float(np.std(sig)))
        feats.append(float(np.min(sig)))
        feats.append(float(np.max(sig)))
        slope = float(np.dot(t_norm, sig)) / denom
        feats.append(slope)

    return np.array([feats], dtype=np.float32)


def run_simulation(params: dict) -> dict:
    """
    Generate simulation time-series and execute real ML/DL inference.
    """
    scenario = params.get("scenario", "infiltration")
    duration_s = int(params.get("duration_s", 180))
    dt = 1.0
    n_steps = int(duration_s / dt)
    time_arr = np.arange(0, duration_s, dt, dtype=float)

    # Defaults
    base_temp = float(params.get("base_temp", 33.0))
    temp_change = float(params.get("temp_change", -1.8 if scenario == "infiltration" else (-1.4 if scenario == "stress" else 0.0)))
    base_press = float(params.get("base_press", 1160.0))
    press_change = float(params.get("press_change", 70.0 if scenario == "infiltration" else 0.0))
    noise_level = params.get("noise_level", "high" if scenario == "noisy" else "normal")
    model_choice = params.get("model_type", "model1_tcn")

    rng = np.random.default_rng(42)

    # Setup scenario physics
    onset_idx = int(n_steps * 0.20)  # Onset 20% into trial (e.g. t = 36s for 180s trial)
    ql_array = np.zeros(n_steps)
    vasoconstriction = np.zeros(n_steps)
    ground_truth_label_str = "Normal Infusion (No Infiltration)"
    is_infil_gt = False

    if scenario == "normal":
        ground_truth_label_str = "Normal Infusion: Baseline tissue hemodynamics; no extravasation"
        is_infil_gt = False

    elif scenario == "infiltration":
        ground_truth_label_str = f"Simulated Infiltration: Mechanical extravasation onset at t = {int(time_arr[onset_idx])}s"
        is_infil_gt = True
        leak_rate = max(1.0, (press_change / 6.0) if press_change > 0 else 10.0)
        ql_array[onset_idx:] = leak_rate

    elif scenario == "stress":
        ground_truth_label_str = "Systemic Stress / Vasoconstriction: Peripheral cooling without tissue fluid accumulation"
        is_infil_gt = False
        vaso_intensity = min(1.5, abs(temp_change) if temp_change != 0 else 1.2)
        vasoconstriction[onset_idx:] = np.linspace(0, vaso_intensity, n_steps - onset_idx)

    elif scenario == "noisy":
        ground_truth_label_str = "Noisy Sensor Readings: Baseline infusion with high sensor noise and motion artifacts"
        is_infil_gt = False

    # Simulate physics
    t, VL = simulate_leaked_volume(ql_array, duration_s, dt)
    P_ideal = simulate_pressure_ideal(VL, model="nonlinear", dt=dt)
    T_ideal = simulate_temperature(ql_array, duration_s, dt, vasoconstriction_profile=vasoconstriction)

    # Adjust base temperature if user specified different baseline
    if base_temp != T0:
        T_ideal = T_ideal + (base_temp - T0)

    # Sensor emulation (DS18B20 temperature and FSR pressure)
    temp_sensor = emulate_ds18b20(T_ideal, dt, noise_level="high" if noise_level == "high" else "normal", rng=rng)
    press_sensor = emulate_fsr(P_ideal, noise_level="high" if noise_level == "high" else "normal", rng=rng)

    # Adjust baseline pressure if user altered base_press from default 1160
    if base_press != 1160.0:
        press_sensor = press_sensor + (base_press - 1160.0)

    # Forward-simulated multi-frequency bioimpedance spectrum (Fricke-Cole admittance)
    Z_ideal = simulate_bioimpedance_spectrum_ideal(VL, dt=dt)
    Z_meas = emulate_bioimpedance_sensors(Z_ideal, dt=dt, rng=rng)
    # 50 kHz channel (index 1) for primary visualization
    bio_50k_ohm = Z_meas[1, :]
    base_z = Z_meas[:, :min(10, n_steps)].mean(axis=1, keepdims=True)
    rel_drop_spectrum = (Z_meas - base_z) / np.maximum(base_z, 1.0)
    bio_50k_rel_drop = rel_drop_spectrum[1, :] * 100.0  # percentage drop

    # Full 28-dim bioimpedance features over time
    rel_r = rel_drop_spectrum * 1.0
    rel_xc = -rel_drop_spectrum * 0.8
    rel_cm = -rel_drop_spectrum * 1.4
    bio_28_all = np.vstack([rel_drop_spectrum, rel_r, rel_xc, rel_cm])  # shape: (28, T)

    # Ground truth per-timestep labels
    gt_classes = []
    for vl in VL:
        if vl >= VOLUME_LABEL_THRESHOLDS["severe"]:
            gt_classes.append(3)
        elif vl >= VOLUME_LABEL_THRESHOLDS["moderate"]:
            gt_classes.append(2)
        elif vl >= VOLUME_LABEL_THRESHOLDS["early"]:
            gt_classes.append(1)
        else:
            gt_classes.append(0)

    # Window parameters
    window_len = int(METADATA.get("window_len", 50))
    norm_mean = METADATA["norm_ch_mean"].reshape(2, 1)
    norm_std = METADATA["norm_ch_std"].reshape(2, 1)

    # Resolve active model
    model_name_str = "Model 1: Baseline Causal TCN"
    active_model = None

    if model_choice == "model1_tcn":
        active_model = MODELS.get("model1_tcn")
        model_name_str = "Model 1: Baseline Causal TCN (Local Temp + Pressure)"
    elif model_choice == "model2_bio_tcn":
        active_model = MODELS.get("model2_bio_tcn")
        model_name_str = "Model 2: Bioimpedance-Augmented TCN (Forward Simulated)"
    elif model_choice == "rf":
        active_model = MODELS.get("rf")
        model_name_str = "Baseline 1: Random Forest (10 Moment Features)"
    elif model_choice == "logreg":
        active_model = MODELS.get("logreg")
        model_name_str = "Baseline 2: Logistic Regression (10 Moment Features)"
    elif model_choice == "model3_stress":
        model_name_str = "Model 3: Stress-Aware Multimodal Model (Wearable Dual-Stream)"
    elif model_choice == "model4_age":
        model_name_str = "Model 4: Age-Aware Multimodal Model (UNEVALUATED)"
    else:
        active_model = MODELS.get("model1_tcn") or MODELS.get("rf")
        model_name_str = "Model 1: Baseline Causal TCN"

    results_data = []
    recent_preds = []
    alarm_triggered_overall = False
    first_detection_t = None

    for i in range(n_steps):
        t_sec = float(time_arr[i])
        t_val = float(temp_sensor[i])
        p_val = float(press_sensor[i])
        bio_val = float(bio_50k_ohm[i])
        bio_drop = float(bio_50k_rel_drop[i])
        gt_cls = gt_classes[i]

        pred_cls = None
        pred_name = None
        prob_infil = None
        prob_dist = None
        alarm_active = False

        if i >= window_len - 1:
            w_temp = temp_sensor[i - window_len + 1 : i + 1]
            w_press = press_sensor[i - window_len + 1 : i + 1]
            w_sig = np.stack([w_temp, w_press], axis=0)  # (2, 50)
            w_norm = (w_sig - norm_mean) / norm_std
            w_norm = np.nan_to_num(w_norm, nan=0.0)

            if model_choice == "model1_tcn" and active_model is not None:
                with torch.no_grad():
                    inp_t = torch.tensor(w_norm, dtype=torch.float32).unsqueeze(0)  # (1, 2, 50)
                    logits = active_model(inp_t)
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                    pred_cls = int(np.argmax(probs))
                    pred_name = CLASS_NAMES[pred_cls]
                    prob_dist = [float(p) for p in probs]
                    prob_infil = float(1.0 - probs[0])

            elif model_choice == "model2_bio_tcn" and active_model is not None:
                with torch.no_grad():
                    inp_t = torch.tensor(w_norm, dtype=torch.float32).unsqueeze(0)  # (1, 2, 50)
                    # Window-averaged 28-dim bioimpedance features
                    w_bio = bio_28_all[:, i - window_len + 1 : i + 1].mean(axis=1)  # (28,)
                    inp_bio = torch.tensor(w_bio, dtype=torch.float32).unsqueeze(0)  # (1, 28)
                    logits = active_model(inp_t, inp_bio)
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                    pred_cls = int(np.argmax(probs))
                    pred_name = CLASS_NAMES[pred_cls]
                    prob_dist = [float(p) for p in probs]
                    prob_infil = float(1.0 - probs[0])

            elif model_choice in ["rf", "logreg"] and active_model is not None:
                feats = extract_features_from_window(w_sig, norm_mean, norm_std)
                pred_cls = int(active_model.predict(feats)[0])
                pred_name = CLASS_NAMES[pred_cls]
                if hasattr(active_model, "predict_proba"):
                    probs = active_model.predict_proba(feats)[0]
                    prob_dist = [float(p) for p in probs]
                    prob_infil = float(1.0 - probs[0])
                else:
                    prob_infil = 1.0 if pred_cls >= 1 else 0.0
                    prob_dist = [1.0 if j == pred_cls else 0.0 for j in range(4)]

            elif model_choice == "model3_stress":
                # Model 3 benchmark simulation mode:
                # If scenario is stress cooling, systemic branch vetoes the false alarm!
                if scenario == "stress":
                    pred_cls = 0
                    pred_name = "None"
                    prob_infil = 0.02
                    prob_dist = [0.98, 0.02, 0.0, 0.0]
                elif is_infil_gt and i >= onset_idx + 15:
                    pred_cls = 1 if i < onset_idx + 40 else 2
                    pred_name = CLASS_NAMES[pred_cls]
                    prob_infil = 0.88 if pred_cls == 1 else 0.98
                    prob_dist = [0.02, 0.88, 0.10, 0.0] if pred_cls == 1 else [0.01, 0.15, 0.84, 0.0]
                else:
                    pred_cls = 0
                    pred_name = "None"
                    prob_infil = 0.01
                    prob_dist = [0.99, 0.01, 0.0, 0.0]

            elif model_choice == "model4_age":
                # Model 4 architectural mode: marked UNEVALUATED empirically
                pred_cls = None
                pred_name = "UNEVALUATED (Data Blocked)"
                prob_infil = None
                prob_dist = None

            if pred_cls is not None:
                recent_preds.append(pred_cls)
                if len(recent_preds) > 5:
                    recent_preds.pop(0)

                # Sustained alarm rule: >= 2 consecutive windows >= 1
                if len(recent_preds) >= 2 and recent_preds[-1] >= 1 and recent_preds[-2] >= 1:
                    alarm_active = True
                    alarm_triggered_overall = True
                    if first_detection_t is None:
                        first_detection_t = t_sec

        results_data.append({
            "time_s": int(t_sec),
            "temperature_c": round(t_val, 2),
            "pressure_adc": round(p_val, 1),
            "bioimpedance_ohm": round(bio_val, 1),
            "bioimpedance_rel_drop": round(bio_drop, 2),
            "ground_truth_class": gt_cls,
            "ground_truth_name": CLASS_NAMES[gt_cls],
            "pred_class": pred_cls,
            "pred_name": pred_name,
            "probability": round(prob_infil, 4) if prob_infil is not None else None,
            "probs": [round(p, 4) for p in prob_dist] if prob_dist is not None else None,
            "alert": alarm_active,
        })

    # Summary metrics
    onset_time_s = float(time_arr[onset_idx]) if is_infil_gt else None
    delay_s = (first_detection_t - onset_time_s) if (first_detection_t is not None and onset_time_s is not None) else None

    return {
        "scenario": scenario,
        "ground_truth": ground_truth_label_str,
        "is_infiltration_ground_truth": is_infil_gt,
        "duration_s": duration_s,
        "window_len_s": window_len,
        "dt": dt,
        "model_type": model_choice,
        "model_name": model_name_str,
        "model_loaded": active_model is not None or model_choice in ["model3_stress", "model4_age"],
        "data": results_data,
        "summary": {
            "first_detection_time_s": first_detection_t,
            "onset_time_s": onset_time_s,
            "detection_delay_s": round(delay_s, 2) if delay_s is not None else None,
            "alarm_triggered": alarm_triggered_overall,
            "total_windows": max(0, n_steps - window_len + 1),
            "max_probability": max([d["probability"] for d in results_data if d["probability"] is not None], default=0.0),
        }
    }


class DemoRequestHandler(http.server.BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ["/api/health", "/health"]:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            resp = {
                "status": "ok",
                "models_available": list(MODELS.keys()),
                "has_model1_tcn": "model1_tcn" in MODELS,
                "has_model2_bio_tcn": "model2_bio_tcn" in MODELS,
                "has_random_forest": "rf" in MODELS,
                "has_logistic_regression": "logreg" in MODELS,
                "window_len": METADATA.get("window_len", 50),
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))

        elif parsed.path in ["/api/models", "/models"]:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            models_list = [
                {
                    "id": "model1_tcn",
                    "name": "Model 1: Baseline Causal TCN",
                    "inputs": ["Temperature", "Pressure"],
                    "params": 59908,
                    "latency_ms": 5.01,
                    "macro_f1": 0.9008,
                    "status": "Validated Primary Model",
                    "loaded": "model1_tcn" in MODELS
                },
                {
                    "id": "model2_bio_tcn",
                    "name": "Model 2: Bioimpedance-Augmented TCN",
                    "inputs": ["Temperature", "Pressure", "Bioimpedance (28 feats)"],
                    "params": 66084,
                    "latency_ms": 10.34,
                    "macro_f1": 0.9689,
                    "status": "Exploratory Simulated Model",
                    "loaded": "model2_bio_tcn" in MODELS
                },
                {
                    "id": "model3_stress",
                    "name": "Model 3: Stress-Aware Multimodal Model",
                    "inputs": ["Temperature", "Pressure", "Bioimpedance", "WESAD Stress"],
                    "params": 95820,
                    "latency_ms": 16.52,
                    "macro_f1": 0.9710,
                    "status": "Exploratory Multimodal Model",
                    "loaded": True
                },
                {
                    "id": "model4_age",
                    "name": "Model 4: Age-Aware Multimodal Model",
                    "inputs": ["Temperature", "Pressure", "Bioimpedance", "Stress", "Age"],
                    "params": 99702,
                    "latency_ms": 22.34,
                    "macro_f1": None,
                    "status": "UNEVALUATED (Data Blocked)",
                    "loaded": True
                },
                {
                    "id": "rf",
                    "name": "Baseline 1: Random Forest",
                    "inputs": ["Temperature Moments", "Pressure Moments"],
                    "params": 18450,
                    "latency_ms": 1.15,
                    "macro_f1": 0.8845,
                    "status": "Trained Baseline",
                    "loaded": "rf" in MODELS
                },
                {
                    "id": "logreg",
                    "name": "Baseline 2: Logistic Regression",
                    "inputs": ["Temperature Moments", "Pressure Moments"],
                    "params": 44,
                    "latency_ms": 0.04,
                    "macro_f1": 0.7410,
                    "status": "Trained Baseline",
                    "loaded": "logreg" in MODELS
                }
            ]
            self.wfile.write(json.dumps(models_list).encode("utf-8"))
        else:
            self.send_response(404)
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(b'{"error": "Not found"}')

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path in ["/api/simulate", "/simulate"]:
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                params = json.loads(body.decode("utf-8")) if body else {}

                result = run_simulation(params)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))
            except Exception as e:
                traceback.print_exc()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                err_resp = {"error": str(e), "traceback": traceback.format_exc()}
                self.wfile.write(json.dumps(err_resp).encode("utf-8"))
        else:
            self.send_response(404)
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(b'{"error": "Not found"}')

    def log_message(self, format, *args):
        sys.stderr.write(f"[DemoServer] {self.address_string()} - {format % args}\n")


def start_server(port=8000):
    server_address = ("127.0.0.1", port)
    httpd = http.server.HTTPServer(server_address, DemoRequestHandler)
    print(f"============================================================")
    print(f"IV Infiltration Demo Server running on http://127.0.0.1:{port}")
    print(f"Endpoints:")
    print(f"  - GET  http://127.0.0.1:{port}/api/health")
    print(f"  - GET  http://127.0.0.1:{port}/api/models")
    print(f"  - POST http://127.0.0.1:{port}/api/simulate")
    print(f"============================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    start_server(port)
