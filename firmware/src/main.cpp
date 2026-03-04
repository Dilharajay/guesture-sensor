/*
 * =====================================================
 *  Gesture Recognition Wearable — Revised Phase 03
 *  ESP8266 UDP IMU Streamer
 *
 *  Board  : ESP8266 NodeMCU / Wemos D1 Mini
 *  Sensor : MPU6050  SDA=D2  SCL=D1
 *
 *  What this firmware does:
 *    1. Connects to WiFi
 *    2. Samples MPU6050 at 50Hz
 *    3. Fills a 50-sample sliding window
 *    4. Sends the complete window as a compact binary
 *       UDP packet to the inference server on your PC
 *    5. Repeats — no ML on device at all
 *
 *  UDP packet format (1208 bytes):
 *    [4 bytes]  uint32  sequence number
 *    [4 bytes]  uint32  timestamp_ms
 *    [1200 bytes] float32[50][6]  window (ax,ay,az,gx,gy,gz)
 *
 *  Libraries needed (platformio.ini already updated):
 *    ESP8266WiFi (built-in)
 *    Wire       (built-in)
 * =====================================================
 */

#include <Arduino.h>
#include <Wire.h>
#include <ESP8266WiFi.h>
#include <WiFiUdp.h>

#include "imu_reader.h"
#include "window_buffer.h"

// -- WiFi config ---------------------------------------
#define WIFI_SSID       "Dilhara’s iPhone"
#define WIFI_PASSWORD   "98765432100"

// -- UDP config ----------------------------------------
// Set this to your PC local IP (from: hostname -I)
#define UDP_SERVER_IP   "172.20.10.3"
#define UDP_SERVER_PORT  5005
#define UDP_LOCAL_PORT   5006

// -- Sampling config -----------------------------------
#define WINDOW_SIZE        50
#define N_FEATURES         6
#define SAMPLE_INTERVAL_MS 20     // 50Hz

// -- UDP packet size -----------------------------------
// 4 (seq) + 4 (ts) + 50*6*4 (floats) = 1208 bytes
#define PACKET_SIZE (4 + 4 + WINDOW_SIZE * N_FEATURES * sizeof(float))

// -- Globals -------------------------------------------
WiFiUDP      udp;
WindowBuffer window;
IPAddress    serverIP;

uint32_t lastSampleMs = 0;
uint32_t seqNumber    = 0;

// Static buffer avoids heap fragmentation on ESP8266
uint8_t packetBuf[PACKET_SIZE];

// -----------------------------------------------------
void connectWiFi() {
  Serial.printf("[WiFi] Connecting to %s", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(400);
    Serial.print(".");
  }
  Serial.printf("\n[WiFi] Connected. IP: %s\n",
                WiFi.localIP().toString().c_str());
}

// -----------------------------------------------------
void sendWindow() {
  uint32_t ts = millis();

  memcpy(packetBuf,     &seqNumber, 4);
  memcpy(packetBuf + 4, &ts,        4);
  window.copyTo((float*)(packetBuf + 8));

  udp.beginPacket(serverIP, UDP_SERVER_PORT);
  udp.write(packetBuf, PACKET_SIZE);
  udp.endPacket();

  seqNumber++;

  if (seqNumber % 100 == 0) {
    Serial.printf("[UDP] Sent %lu windows\n", seqNumber);
  }
}

// -----------------------------------------------------
void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n== Gesture Wearable - UDP IMU Streamer ==");

  Wire.begin(D2, D1);
  initIMU();
  connectWiFi();

  serverIP.fromString(UDP_SERVER_IP);
  udp.begin(UDP_LOCAL_PORT);

  Serial.printf("[UDP] Streaming to %s:%d\n",
                UDP_SERVER_IP, UDP_SERVER_PORT);
  Serial.println("[System] Ready.");
}

// -----------------------------------------------------
void loop() {
  uint32_t now = millis();

  if (now - lastSampleMs >= SAMPLE_INTERVAL_MS) {
    lastSampleMs = now;

    float ax, ay, az, gx, gy, gz;
    readIMU(ax, ay, az, gx, gy, gz);

    float sample[N_FEATURES] = { ax, ay, az, gx, gy, gz };
    window.push(sample);

    if (window.isFull()) {
      sendWindow();
    }
  }
}
