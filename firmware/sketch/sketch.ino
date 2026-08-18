#include <OneWire.h>
#include <DallasTemperature.h>

#define ONE_WIRE_PIN 4
#define FSR_PIN 34
#define LED_PIN 2

#define DEMO_MODE true

OneWire oneWire(ONE_WIRE_PIN);
DallasTemperature tempSensor(&oneWire);

// Class centroids

const float TEMP_CENTROIDS[4] = {
  32.870564,
  32.122957,
  31.428571,
  30.839139
};

const float PRESS_CENTROIDS[4] = {
  50.060393,
  50.992162,
  52.966085,
  56.063391
};

const char* SEVERITY_NAMES[4] = {
  "none",
  "early",
  "moderate",
  "severe"
};

// Feature scaling

const float TEMP_RANGE =
  TEMP_CENTROIDS[0] - TEMP_CENTROIDS[3];

const float PRESS_RANGE =
  PRESS_CENTROIDS[3] - PRESS_CENTROIDS[0];

unsigned long lastSample = 0;
const unsigned long SAMPLE_INTERVAL_MS = 2500;

// Demo scenarios

struct DemoScenario {
  float temperature;
  int pressure;
};

const DemoScenario DEMO_SCENARIOS[] = {
  {32.90, 50},
  {32.75, 50},
  {32.30, 51},
  {32.15, 51},
  {31.70, 52},
  {31.55, 53},
  {31.20, 54},
  {31.00, 55},
  {30.85, 56},
  {30.70, 57},
  {32.45, 51},
  {31.80, 52},
  {32.85, 50}
};

const int NUM_DEMO_SCENARIOS =
  sizeof(DEMO_SCENARIOS) / sizeof(DEMO_SCENARIOS[0]);

int lastDemoIndex = -1;

// Classification

int classifySeverity(float tempC, int pressureAdc) {

  int bestClass = 0;
  float bestDist = 1e9;

  for (int i = 0; i < 4; i++) {

    float dT =
      (tempC - TEMP_CENTROIDS[i]) / TEMP_RANGE;

    float dP =
      (pressureAdc - PRESS_CENTROIDS[i]) / PRESS_RANGE;

    float dist =
      dT * dT + dP * dP;

    if (dist < bestDist) {
      bestDist = dist;
      bestClass = i;
    }
  }

  return bestClass;
}

// Setup

void setup() {

  Serial.begin(115200);

  pinMode(LED_PIN, OUTPUT);
  pinMode(FSR_PIN, INPUT);

  tempSensor.begin();

  randomSeed(analogRead(0));

  Serial.println(
    "time_ms,temperature_c,pressure_adc,severity_class,severity_name,alert"
  );
}

// Loop

void loop() {

  unsigned long now = millis();

  if (now - lastSample < SAMPLE_INTERVAL_MS)
    return;

  lastSample = now;

  float tempC;
  int pressureAdc;

  if (!DEMO_MODE) {

    tempSensor.requestTemperatures();

    tempC = tempSensor.getTempCByIndex(0);
    pressureAdc = analogRead(FSR_PIN);

    if (tempC == DEVICE_DISCONNECTED_C) {

      Serial.println(
        "sensor not ready, skipping this cycle"
      );

      return;
    }

  } else {

    int demoIndex;

    do {
      demoIndex = random(NUM_DEMO_SCENARIOS);
    } while (demoIndex == lastDemoIndex);

    lastDemoIndex = demoIndex;

    tempC =
      DEMO_SCENARIOS[demoIndex].temperature;

    pressureAdc =
      DEMO_SCENARIOS[demoIndex].pressure;
  }

  int severityClass =
    classifySeverity(tempC, pressureAdc);

  bool alert =
    severityClass > 0;

  digitalWrite(
    LED_PIN,
    alert ? HIGH : LOW
  );

  Serial.print(now);
  Serial.print(",");

  Serial.print(tempC, 3);
  Serial.print(",");

  Serial.print(pressureAdc);
  Serial.print(",");

  Serial.print(severityClass);
  Serial.print(",");

  Serial.print(SEVERITY_NAMES[severityClass]);
  Serial.print(",");

  Serial.println(alert ? 1 : 0);
}