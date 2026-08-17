/*
  IV Infiltration Monitoring -- V1 Wokwi Firmware

  Reads DS18B20 temperature + an FSR-proxy potentiometer (Wokwi has no native
  FSR part; a potentiometer is electrically equivalent for firmware-testing
  purposes -- it's still just a variable resistance/voltage on an analog pin).
  Prints CSV over Serial for logging, and runs a PLACEHOLDER threshold alert.

  Replace the placeholder threshold logic with your quantized TFLite Micro
  model's inference call once it's trained (Step 11-12 of the project plan).
  Keep the sampling/printing structure the same -- only swap what happens
  between "read sensors" and "set alert".

  Wokwi wiring (see diagram.json):
    DS18B20 data pin      -> GPIO 4  (needs 4.7k pullup to 3.3V, included in diagram.json)
    Potentiometer (FSR proxy), wiper -> GPIO 34 (ADC1_CH6, input-only pin)
    Alert LED              -> GPIO 2 (through 220ohm resistor)

  Required Wokwi libraries (add via the Library Manager tab in the Wokwi editor,
  or list in firmware/libraries.txt):
    OneWire
    DallasTemperature
*/

#include <OneWire.h>
#include <DallasTemperature.h>

#define ONE_WIRE_PIN 4
#define FSR_PIN 34
#define LED_PIN 2

OneWire oneWire(ONE_WIRE_PIN);
DallasTemperature tempSensor(&oneWire);

// --- placeholder thresholds; tune against your simulator's noise-free deltas,
//     then replace entirely with the trained model's decision logic ---
const float TEMP_ALERT_DELTA = 0.4;      // deg C drop from baseline (see sanity-check plot)
const int   PRESSURE_ALERT_DELTA = 200;  // ADC counts rise from baseline

float baselineTemp = -1000;   // sentinel "not yet set"
int   baselinePressure = -1;

unsigned long lastSample = 0;
const unsigned long SAMPLE_INTERVAL_MS = 1000;  // ~1 Hz, matches simulator DT

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  pinMode(FSR_PIN, INPUT);
  tempSensor.begin();
  Serial.println("time_ms,temperature_c,pressure_adc,alert");
}

void loop() {
  unsigned long now = millis();
  if (now - lastSample < SAMPLE_INTERVAL_MS) return;
  lastSample = now;

  tempSensor.requestTemperatures();
  float tempC = tempSensor.getTempCByIndex(0);
  int pressureAdc = analogRead(FSR_PIN);

  // establish baseline on first valid reading
  if (baselineTemp < -999 && tempC != DEVICE_DISCONNECTED_C) {
    baselineTemp = tempC;
    baselinePressure = pressureAdc;
  }

  bool alert = false;
  if (baselineTemp > -999) {
    alert = (baselineTemp - tempC > TEMP_ALERT_DELTA) ||
            (pressureAdc - baselinePressure > PRESSURE_ALERT_DELTA);
  }

  digitalWrite(LED_PIN, alert ? HIGH : LOW);

  Serial.print(now);        Serial.print(",");
  Serial.print(tempC, 3);   Serial.print(",");
  Serial.print(pressureAdc);Serial.print(",");
  Serial.println(alert ? 1 : 0);
}
