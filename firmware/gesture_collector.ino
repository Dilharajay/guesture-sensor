/*
 * =====================================================
 *  Gesture Recognition Wearable — Phase 01
 *  Data Collection Firmware
 *  Board  : ESP8266 (NodeMCU / Wemos D1)
 *  Sensor : MPU6050 via I2C (SDA=D2, SCL=D1)
 *  Baud   : 115200
 * =====================================================
 *
 *  HOW TO USE:
 *    1. Upload this sketch to your ESP8266.
 *    2. Open Serial Monitor at 115200 baud (or run the Python logger).
 *    3. Send a single character over Serial to start/stop recording:
 *         's' = start recording a gesture window
 *         'x' = stop / discard current window
 *    4. The Python logger will handle labeling and saving to CSV.
 *
 *  OUTPUT FORMAT (CSV line per sample):
 *    timestamp_ms,ax,ay,az,gx,gy,gz
 *    1023,0.123,-0.456,9.801,0.012,-0.034,0.005
 */

#include <Wire.h>

// ── MPU6050 I2C address ─────────────────────────────
#define MPU_ADDR     0x68
#define PWR_MGMT_1   0x6B
#define ACCEL_CONFIG 0x1C
#define GYRO_CONFIG  0x1B
#define ACCEL_XOUT_H 0x3B

// ── Sampling config ─────────────────────────────────
#define SAMPLE_RATE_HZ  50        // samples per second
#define WINDOW_SIZE     50        // 50 samples = 1 second window
#define SAMPLE_DELAY_MS (1000 / SAMPLE_RATE_HZ)

// ── Scale factors ───────────────────────────────────
// Accel: ±2g range  → 16384 LSB/g
// Gyro : ±250°/s    → 131.0 LSB/°/s
#define ACCEL_SCALE  16384.0f
#define GYRO_SCALE   131.0f

// ── State ───────────────────────────────────────────
bool     recording    = false;
uint16_t sampleCount  = 0;
uint32_t lastSampleMs = 0;

// ── Raw read buffer ─────────────────────────────────
int16_t rawData[7]; // ax, ay, az, temp, gx, gy, gz

// ────────────────────────────────────────────────────
void setupMPU6050() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(PWR_MGMT_1);
  Wire.write(0x00); // wake up
  Wire.endTransmission(true);
  delay(100);

  // Accel: ±2g
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(ACCEL_CONFIG);
  Wire.write(0x00);
  Wire.endTransmission(true);

  // Gyro: ±250°/s
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(GYRO_CONFIG);
  Wire.write(0x00);
  Wire.endTransmission(true);
}

// ────────────────────────────────────────────────────
void readMPU6050(float &ax, float &ay, float &az,
                 float &gx, float &gy, float &gz) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(ACCEL_XOUT_H);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 14, true);

  for (int i = 0; i < 7; i++) {
    rawData[i]  = Wire.read() << 8;
    rawData[i] |= Wire.read();
  }

  ax = rawData[0] / ACCEL_SCALE;
  ay = rawData[1] / ACCEL_SCALE;
  az = rawData[2] / ACCEL_SCALE;
  // rawData[3] = temperature (skip)
  gx = rawData[4] / GYRO_SCALE;
  gy = rawData[5] / GYRO_SCALE;
  gz = rawData[6] / GYRO_SCALE;
}

// ────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  Wire.begin(4, 5); // SDA=D2, SCL=D1 on NodeMCU
  setupMPU6050();

  Serial.println("# GESTURE COLLECTOR READY");
  Serial.println("# Send 's' to START a window, 'x' to DISCARD");
  Serial.println("# Format: timestamp_ms,ax,ay,az,gx,gy,gz");
}

// ────────────────────────────────────────────────────
void loop() {
  // Handle serial commands
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == 's' && !recording) {
      recording   = true;
      sampleCount = 0;
      Serial.println("# WINDOW_START");
    } else if (cmd == 'x' && recording) {
      recording = false;
      Serial.println("# WINDOW_DISCARD");
    }
  }

  // Timed sampling
  if (recording) {
    uint32_t now = millis();
    if (now - lastSampleMs >= SAMPLE_DELAY_MS) {
      lastSampleMs = now;

      float ax, ay, az, gx, gy, gz;
      readMPU6050(ax, ay, az, gx, gy, gz);

      // CSV line
      Serial.print(now);        Serial.print(',');
      Serial.print(ax, 4);      Serial.print(',');
      Serial.print(ay, 4);      Serial.print(',');
      Serial.print(az, 4);      Serial.print(',');
      Serial.print(gx, 4);      Serial.print(',');
      Serial.print(gy, 4);      Serial.print(',');
      Serial.println(gz, 4);

      sampleCount++;

      // Auto-stop after full window
      if (sampleCount >= WINDOW_SIZE) {
        recording = false;
        Serial.println("# WINDOW_END");
      }
    }
  }
}
