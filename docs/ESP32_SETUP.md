# ESP32 Setup & Wiring Guide

This document explains how to set up the ESP32 hardware and flash the edge-inference firmware for the IV Infiltration Monitoring System.

## Hardware Requirements

**NO PHYSICAL SENSORS ARE REQUIRED FOR THIS DEMO!**

Since you do not have the physical temperature and pressure sensors, we are using the ESP32 in a **Hardware-in-the-Loop (HIL) Edge AI configuration**. 
This means:
1. The laptop (or frontend) acts as the "virtual sensor" and sends raw data to the ESP32 over the USB cable.
2. The ESP32 receives this data, computes the rolling window statistics, and runs the Logistic Regression model (Edge AI).
3. The ESP32 sends the calculated probability and severity back to the laptop over the same USB cable to be displayed on the 3D dashboard.

All you need is:
- 1x ESP32 Development Board
- 1x Micro-USB or USB-C cable (ensure it supports data transfer, not just charging)

## Flashing the Firmware (How to Upload)

1. Download and install the [Arduino IDE](https://www.arduino.cc/en/software).
2. Open the Arduino IDE, go to **File > Preferences**, and paste this URL into the "Additional Boards Manager URLs" field:
   `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
3. Go to **Tools > Board > Boards Manager**, search for `esp32` and install it.
4. Open the file `firmware/sketch/sketch.ino` in the Arduino IDE.
5. In the same folder, ensure `model_params.h` is present (this contains your LogReg math).
6. Go to **Tools > Board** and select your ESP32 model (usually `DOIT ESP32 DEVKIT V1` or similar).
7. Go to **Tools > Port** and select the COM port that appears when you plug in the ESP32.
8. Click the **Upload** button (the right arrow icon at the top left).

## How it works with your laptop

Once flashed, the ESP32 will constantly listen to the USB serial port. 
When you click **"Connect ESP32"** in the React frontend, the frontend will automatically start sending the offline trace data to the ESP32. The ESP32 will do the heavy ML math, and send the result back. This proves to your panelists that the AI is running on the edge microcontroller, not the laptop!

## Testing the Serial Output

Once flashed, open the Serial Monitor at **115200 baud**. You should see a CSV header followed by data rows:
```csv
time_ms,temperature_c,pressure_adc,condition,probability,severity_class,alert
3500,32.8,750,normal,0.0102,0,0
...
```

If the sensor is disconnected, it will output:
`sensor not ready, skipping this cycle`

When connected via USB to the web dashboard, the dashboard will automatically parse this serial stream.
