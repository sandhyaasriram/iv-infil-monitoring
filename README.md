# Intelligent IV Infiltration Monitoring System

Low-cost, edge-deployable monitoring for intravenous (IV) infiltration and extravasation, built around physics-inspired simulation, lightweight machine learning, and an ESP32-based sensing prototype.

> **Status:** V1 in progress — physics simulator complete, phantom-arm validation and edge deployment pending.
> **SDG alignment:** SDG 3 (Good Health & Well-Being, Target 3.8) · SDG 9 (Industry, Innovation & Infrastructure, Target 9.5)

---

## The problem

IV infiltration — where infused fluid leaks from the vein into surrounding tissue — is one of the most common complications of infusion therapy, and one of the hardest to catch early. Detection today relies almost entirely on periodic visual and tactile inspection by nursing staff, which means the event is often only caught after visible swelling, discoloration, or patient discomfort has already developed. Commercial automated alternatives (e.g. optical/bioimpedance systems) exist but are expensive and impractical for continuous use in resource-constrained clinical settings — exactly the settings where nurse-to-patient ratios make manual monitoring hardest to sustain.

## What this project does

This project investigates whether **inexpensive temperature and pressure sensing, combined with lightweight on-device machine learning, can enable earlier, continuous IV infiltration monitoring** without specialized or costly hardware.

The approach is deliberately staged rather than jumping straight to hardware:

```
Physics-inspired simulation  →  Synthetic sensor dataset  →  Lightweight ML
        →  Physical phantom-arm validation  →  Quantized model on ESP32
```

Simulating first, before any physical data collection, lets us systematically vary conditions that are hard to control experimentally (leak rate, ambient temperature, sensor noise, missing/failed sensors) and generate a much larger, more diverse labeled dataset than a phantom-arm setup alone could produce in the available time — while still validating everything against real hardware before drawing conclusions.

## Research question

*Can low-cost multimodal sensing (temperature + pressure) and lightweight machine learning models enable early IV infiltration monitoring in resource-constrained environments?*

## Why this, and why now

This isn't an unexplored problem — there's real prior work, most notably an [open-source wearable multimodal system](https://doi.org/10.1109/TIM.2020.3025394) combining pressure, temperature, and optical sensing with a Light-ConvLSTM model (IEEE TIM, 2021). This project positions itself as a **cost- and resource-reduced variant** of that lineage: dropping the optical sensor, targeting sub-$20 hardware, and — unlike prior work, which trains and infers server-side — running the final detection model directly on-device via TensorFlow Lite Micro. See [`docs/related-work.md`](docs/related-work.md) *(add your literature review here)* for the full positioning against existing architectures.

## Repository structure

```
.
├── simulator/              Physics-inspired sensor simulation (V1: linear swelling, 0D lumped thermal)
│   ├── parameters.py        All tunable physical/sensor constants in one place
│   ├── infusion.py          Leaked-volume accumulation model
│   ├── tissue.py             Swelling + ideal pressure-sensor response
│   ├── thermal.py            0D lumped thermal ODE
│   └── scenarios.py          Labeled trial generator (4 severity classes)
├── sensors/
│   └── noise.py              DS18B20 / FSR noise, drift, quantization, missing-sensor emulation
├── generate_dataset.py      End-to-end pipeline → labeled CSV dataset
├── firmware/                 ESP32 firmware + Wokwi simulation project
│   ├── sketch.ino
│   ├── diagram.json
│   └── libraries.txt
├── models/                   (planned) trained models, TFLite exports
├── notebooks/                 (planned) Colab/Jupyter exploration
└── docs/                      (planned) related-work, methodology writeup
```

## Quickstart

**Simulate the dataset (Google Colab or local Python):**
```bash
pip install numpy pandas matplotlib
python generate_dataset.py
```
Produces `synthetic_iv_infiltration_dataset.csv` — labeled temperature/pressure time-series across four severity classes (none / early / moderate / severe), each replicated under five sensor-robustness conditions (clean, noisy temperature, noisy pressure, missing temperature, missing pressure).

**Run the firmware (Wokwi, free, browser-based, no hardware needed):**
1. Create a new ESP32 project at [wokwi.com](https://wokwi.com)
2. Paste in `firmware/sketch.ino` and `firmware/diagram.json`
3. Add the `OneWire` and `DallasTemperature` libraries via the Library Manager
4. Run the simulation and watch CSV sensor output over Serial

Full step-by-step setup (Colab + Wokwi) is in [`SETUP.md`](SETUP.md).

## Hardware (planned)

| Component | Role |
|---|---|
| ESP32 | Edge processing + inference |
| DS18B20 | Skin-surface temperature sensing |
| FSR (force-sensitive resistor) | Swelling / mechanical deformation sensing |
| Phantom arm (DIY gelatin/agar, or IV-cannulation training arm) | Controlled infiltration validation |

## Current limitations (V1, stated explicitly)

- **Swelling model is linear.** Real tissue compliance is nonlinear (resists more as it stretches); this is a documented V1 simplification, not an oversight.
- **Thermal model is 0D lumped**, not spatially resolved — sufficient for feature-based ML, but can't yet study sensor-placement sensitivity. A 2D bioheat model is a possible V2 extension.
- **All physical constants are placeholder values** pending calibration against real phantom-arm data.
- **No clinical validation.** This system is evaluated on synthetic data and artificial tissue phantoms only, and is not intended for clinical use in its current form.

## Roadmap

- [x] Physics simulator (leakage, swelling, thermal) — V1
- [x] Sensor noise/drift/missing-data emulation layer
- [x] ESP32 firmware validated in Wokwi
- [ ] Lightweight ML baselines (logistic regression → random forest → 1D CNN)
- [ ] Sensor ablation + robustness experiments
- [ ] Physical phantom-arm construction and data collection
- [ ] Synthetic-to-physical transfer evaluation
- [ ] Model quantization + on-device ESP32 deployment
- [ ] Latency / memory / power benchmarking on real hardware

## Contributions

This work, as currently scoped, aims to contribute:
1. A physics-inspired, open-source sensor simulator for IV infiltration research (filling a gap where real clinical datasets are essentially unavailable for ethical reasons).
2. A resource-aware evaluation of multimodal detection under realistic sensor noise, drift, and failure conditions — not just clean-signal accuracy.
3. An empirical study of synthetic-to-physical transfer performance, and a fully edge-deployed (not server-dependent) reference implementation.

## Citation

If you use this simulator or dataset, please cite (update once published):
```bibtex
@misc{iv-infiltration-monitoring,
  title  = {Intelligent IV Infiltration Monitoring System},
  author = {<your names>},
  year   = {2026},
  note   = {Undergraduate research project},
  url    = {<repo URL>}
}
```

## License

*(choose one — MIT is a reasonable default for an open-source research tool; add `LICENSE` file)*

## Acknowledgements

Built on the architectural direction of Lee & Lin, *"An Open-Source Wearable Sensor System for Detecting Extravasation of Intravenous Infusion,"* IEEE Transactions on Instrumentation and Measurement, 2021, and the broader extravasation-sensing literature reviewed in this project's related-work documentation.
