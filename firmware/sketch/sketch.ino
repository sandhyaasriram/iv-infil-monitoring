/*
  IV Infiltration Monitoring -- V3 Wokwi Firmware

  Reads DS18B20 temperature + an FSR-proxy potentiometer (Wokwi has no native
  FSR part; a potentiometer is electrically equivalent for firmware-testing
  purposes -- it's still just a variable resistance/voltage on an analog pin).
  Prints CSV over Serial for logging.

  Severity is estimated via NEAREST-CENTROID classification in (temp, pressure)
  space, using the actual class-mean centroids computed from the v3 synthetic
  dataset (train_df.groupby('label')[['temp_mean','press_mean']].mean()).
  This is a lightweight stand-in for the full trained model -- replace with a
  real quantized TFLite Micro / emlearn-ported model once phantom-arm data is
  available and the model is finalized (Step 11-12 of the project plan).

  CAUTION: PRESS_CENTROIDS gaps between adjacent classes are under 1-6 ADC
  counts in the synthetic data, small relative to real ESP32 ADC noise (often
  several counts of jitter). Treat pressure-based classification as provisional
  until recalibrated against real hardware readings -- don't trust it blindly.

  Wokwi wiring (see diagram.json):
    DS18B20 data pin                 -> GPIO 4  (needs 4.7k pullup to 3.3V, in diagram.json)
    Potentiometer (FSR proxy), wiper -> GPIO 34 (ADC1_CH6, input-only pin)
    Alert LED                        -> GPIO 2  (through 220ohm resistor)

  Required libraries: OneWire, DallasTemperature
  (Browser Wokwi: add via Library Manager tab. VS Code extension: arduino-cli
  lib install "OneWire" "DallasTemperature" -- see firmware setup notes.)
*/

#include <OneWire.h>
#include <DallasTemperature.h>

#define ONE_WIRE_PIN 4
#define FSR_PIN 34
#define LED_PIN 2

OneWire oneWire(ONE_WIRE_PIN);
DallasTemperature tempSensor(&oneWire);

// --- Class centroids from train_df.groupby('label')[['temp_mean','press_mean']].mean() ---
// index: 0=none, 1=early, 2=moderate, 3=severe
const float TEMP_CENTROIDS[4]  = {32.870564, 32.122957, 31.428571, 30.839139};
const float PRESS_CENTROIDS[4] = {50.060393, 50.992162, 52.966085, 56.063391};
const char* SEVERITY_NAMES[4]  = {"none", "early", "moderate", "severe"};

// Feature scaling: temp spans ~2.03 C total across classes, pressure spans
// ~6.00 counts total -- very different raw scales. Without normalizing each
// axis by its own range, nearest-centroid distance would be dominated by
// whichever channel happens to have larger raw numbers, not whichever is
// actually more informative (this mirrors why we standardize features before
// the Random Forest / Logistic Regression in the Colab pipeline).
const float TEMP_RANGE  = TEMP_CENTROIDS[0] - TEMP_CENTROIDS[3];   // ~2.03
const float PRESS_RANGE = PRESS_CENTROIDS[3] - PRESS_CENTROIDS[0]; // ~6.00

unsigned long lastSample = 0;
const unsigned long SAMPLE_INTERVAL_MS = 1000;  // ~1 Hz, matches simulator DT

int classifySeverity(float tempC, int pressureAdc) {
  int bestClass = 0;
  float bestDist = 1e9;
  for (int i = 0; i < 4; i++) {
    float dT = (tempC - TEMP_CENTROIDS[i]) / TEMP_RANGE;
    float dP = (pressureAdc - PRESS_CENTROIDS[i]) / PRESS_RANGE;
    float dist = dT * dT + dP * dP;   // squared Euclidean in normalized space
    if (dist < bestDist) {
      bestDist = dist;
      bestClass = i;
    }
  }
  return bestClass;
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  pinMode(FSR_PIN, INPUT);
  tempSensor.begin();
  Serial.println("time_ms,temperature_c,pressure_adc,severity_class,severity_name,alert");
}

void loop() {
  unsigned long now = millis();
  if (now - lastSample < SAMPLE_INTERVAL_MS) return;
  lastSample = now;

  tempSensor.requestTemperatures();
  float tempC = tempSensor.getTempCByIndex(0);
  int pressureAdc = analogRead(FSR_PIN);

  int severityClass = classifySeverity(tempC, pressureAdc);
  bool alert = severityClass > 0;   // anything above "none" triggers the LED

  digitalWrite(LED_PIN, alert ? HIGH : LOW);

  Serial.print(now);                            Serial.print(",");
  Serial.print(tempC, 3);                        Serial.print(",");
  Serial.print(pressureAdc);                     Serial.print(",");
  Serial.print(severityClass);                   Serial.print(",");
  Serial.print(SEVERITY_NAMES[severityClass]);   Serial.print(",");
  Serial.println(alert ? 1 : 0);
}