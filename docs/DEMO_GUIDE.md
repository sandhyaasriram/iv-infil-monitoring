# Demo Guide: IV Infiltration Monitoring System

This guide outlines how to present the system in your final-year demonstration. There are two primary modes of operation: **Live Hardware Mode** (using the ESP32) and **Trace Simulation Mode** (offline, software-only).

---

## 1. Trace Simulation Mode (Fallback/Offline)

If your ESP32 hardware is unavailable or you want to demonstrate specific edge-case scenarios precisely, use the simulation mode.

1. **Start the Frontend**:
   ```bash
   cd frontend
   npm run dev
   ```
2. Open `http://localhost:5173` in your browser (Chrome or Edge recommended).
3. **Start Simulation**: Click the **Start Sim** button in the top right.
4. **Demonstrate Robustness**: Use the **Fault Injection** panel to toggle conditions like `NOISY_TEMP` or `MISSING_PRESSURE`. The 3D Arm, waveforms, and ML probability will react to the canonical offline trace instantly.
5. **Robustness Matrix**: Point out the F1 scores in the matrix panel to show how the models handle the injected faults.

---

## 2. Live Hardware Mode (ESP32 via Web Serial)

This mode demonstrates the edge-inference capabilities of the ESP32 in real-time.

1. **Flash the ESP32**: Ensure the ESP32 is flashed with `firmware/sketch/sketch.ino`.
2. **Connect via USB**: Plug the ESP32 into your computer.
3. **Start the Frontend**: Run `npm run dev` in the `frontend` folder.
4. **Connect Serial**: Click the **Connect ESP32** button. The browser will prompt you to select a COM port. Select the one matching your ESP32 (e.g., `CP210x` or `CH340`).
5. **Live Demonstration (Hardware-in-the-loop)**: 
   - The React frontend will immediately start streaming the `replay_trace.csv` data to the ESP32 over the USB cable.
   - The ESP32 evaluates the Logistic Regression model onboard in real-time.
   - The ESP32 streams the inference results back to the UI.
   - The UI will instantly reflect the incoming serial data, updating the waveforms, the 3D arm status, and the event log.
   - If the ESP32 determines `severity > 1` (Moderate or Severe), the UI will flash red and indicate an alert. This proves the edge device is successfully running the ML model!

### Troubleshooting Web Serial
- Ensure no other applications (like the Arduino Serial Monitor) are using the COM port.
- Web Serial requires a Chromium-based browser (Chrome, Edge, Opera) and must be served over `localhost` or `https`.
