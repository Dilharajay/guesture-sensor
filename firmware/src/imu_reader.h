/*
 * imu_reader.h
 * MPU6050 read helpers — Phase 03
 * Reuses the same register map from Phase 01.
 */

#ifndef IMU_READER_H
#define IMU_READER_H

#include <Wire.h>

#define MPU_ADDR      0x68
#define PWR_MGMT_1    0x6B
#define ACCEL_CONFIG  0x1C
#define GYRO_CONFIG   0x1B
#define ACCEL_XOUT_H  0x3B

#define ACCEL_SCALE   16384.0f   // ±2g
#define GYRO_SCALE    131.0f     // ±250°/s

static int16_t _raw[7];

inline void initIMU() {
  // Wake up
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(PWR_MGMT_1); Wire.write(0x00);
  Wire.endTransmission(true);
  delay(100);

  // Accel ±2g
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(ACCEL_CONFIG); Wire.write(0x00);
  Wire.endTransmission(true);

  // Gyro ±250°/s
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(GYRO_CONFIG); Wire.write(0x00);
  Wire.endTransmission(true);

  Serial.println("[IMU] MPU6050 initialized.");
}

inline void readIMU(float &ax, float &ay, float &az,
                    float &gx, float &gy, float &gz) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(ACCEL_XOUT_H);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 14, true);

  for (int i = 0; i < 7; i++) {
    _raw[i]  = Wire.read() << 8;
    _raw[i] |= Wire.read();
  }

  ax = _raw[0] / ACCEL_SCALE;
  ay = _raw[1] / ACCEL_SCALE;
  az = _raw[2] / ACCEL_SCALE;
  gx = _raw[4] / GYRO_SCALE;
  gy = _raw[5] / GYRO_SCALE;
  gz = _raw[6] / GYRO_SCALE;
}

#endif // IMU_READER_H
