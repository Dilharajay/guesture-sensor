# GestureCMD — Wearable Gesture Recognition System

A complete end-to-end gesture recognition wearable system that recognizes hand gestures using an ESP8266 + MPU6050 IMU sensor and translates them into keyboard shortcuts on your PC. Control your desktop, presentations, or smart home with hand movements.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python)
![PlatformIO](https://img.shields.io/badge/PlatformIO-ESP8266-orange?logo=platformio)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20+-red?logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## Features

- **Real-time gesture recognition** — 50Hz IMU sampling with sub-100ms inference latency
- **7 gesture classes** — wave left/right, flick up/down, fist hold, wrist rotate, idle
- **PC-based inference** — lightweight TFLite INT8 model runs on your computer
- **Global keyboard shortcuts** — gestures trigger customizable hotkeys (Ctrl+Tab, Alt+Tab, etc.)
- **Live dashboard** — browser-based real-time monitoring with MQTT WebSocket
- **Modular architecture** — separate data collection, training, and inference pipelines

---

## Project Structure

```
gesture-sensor/
├── firmware/                    # ESP8266 PlatformIO project
│   ├── src/
│   │   ├── main.cpp             # UDP IMU streaming firmware
│   │   ├── data_collection.cpp  # Serial data collection firmware
│   │   ├── imu_reader.h         # MPU6050 I2C driver
│   │   └── window_buffer.h      # Circular buffer for sliding window
│   ├── include/
│   │   └── cmsis_gcc.h          # CMSIS compatibility shim
│   └── platformio.ini           # Build configuration
│
├── src/                         # Python source code
│   ├── training/                # ML pipeline
│   │   ├── gesture_logger.py    # Collect labeled gesture data
│   │   ├── preprocess.py        # Normalize and split dataset
│   │   ├── train_model.py       # Train 1D-CNN model
│   │   └── quantize_export.py   # Convert to INT8 TFLite
│   ├── server/
│   │   └── inference_server.py  # UDP→TFLite→MQTT inference
│   └── hci/
│       └── hci_controller.py    # MQTT→keyboard shortcuts
│
├── dashboard/
│   └── gesture_dashboard.html   # Live gesture monitoring UI
│
├── data/                        # Dataset and preprocessed arrays
│   ├── gesture_dataset.csv      # Raw collected samples
│   ├── X_train.npy, y_train.npy # Training data
│   ├── label_map.json           # Gesture → index mapping
│   └── scaler.pkl               # StandardScaler for inference
│
├── models/                      # Trained models
│   ├── gesture_model.keras      # Full Keras model
│   ├── gesture_model_int8.tflite # Quantized TFLite model
│   └── gesture_model.h          # C header for embedded use
│
├── config/
│   └── mosquitto.conf           # MQTT broker configuration
│
├── docs/                        # Documentation and diagrams
├── pyproject.toml               # Python dependencies (uv)
└── README.md                    # This file
```

---

## Hardware Requirements

| Component | Specification | Notes |
|-----------|--------------|-------|
| **Microcontroller** | ESP8266 (NodeMCU v2 or Wemos D1 Mini) | Built-in WiFi |
| **IMU Sensor** | MPU6050 6-axis (Accel + Gyro) | I2C interface |
| **Power** | 5V USB or 3.7V LiPo | ~80mA typical draw |
| **Mounting** | Wrist strap or glove | Secure attachment important |

### Wiring Diagram

```
                    ESP8266 NodeMCU
                   ┌──────────────────┐
                   │                  │
        ┌──────────┤ D2 (GPIO4) SDA   │
        │          │                  │
        │  ┌───────┤ D1 (GPIO5) SCL   │
        │  │       │                  │
        │  │   ┌───┤ 3V3              │
        │  │   │   │                  │
        │  │   │  ─┤ GND              │
        │  │   │   │                  │
        │  │   │   └──────────────────┘
        │  │   │
        │  │   │        MPU6050
        │  │   │   ┌──────────────────┐
        │  │   └───┤ VCC              │
        │  │       │                  │
        │  └───────┤ SCL              │
        │          │                  │
        └──────────┤ SDA              │
                   │                  │
               ────┤ GND              │
                   │                  │
                   │ AD0 ─── GND      │ (I2C addr 0x68)
                   │ INT ─── NC       │ (not used)
                   └──────────────────┘
```

### Pin Connections

| MPU6050 Pin | ESP8266 Pin | Description |
|-------------|-------------|-------------|
| VCC | 3V3 | 3.3V power supply |
| GND | GND | Ground |
| SCL | D1 (GPIO5) | I2C clock |
| SDA | D2 (GPIO4) | I2C data |
| AD0 | GND | I2C address select (0x68) |
| INT | NC | Not connected |

---

## Quick Start

### Prerequisites

- **Python 3.12+** with [uv](https://docs.astral.sh/uv/) package manager
- **PlatformIO** (VS Code extension or CLI)
- **Mosquitto MQTT broker** (`apt install mosquitto mosquitto-clients`)
- **ESP8266 board** with USB cable

### 1. Clone and Install Dependencies

```bash
git clone https://github.com/dilharajay/gesture-sensor.git
cd gesture-sensor

# Install Python dependencies
uv sync
```

### 2. Configure MQTT Broker

```bash
# Copy config to Mosquitto
sudo cp config/mosquitto.conf /etc/mosquitto/conf.d/wearable.conf
sudo systemctl restart mosquitto

# Verify broker is running
mosquitto_sub -t "wearable/gesture" -v &
```

### 3. Flash Firmware

#### For Data Collection (Phase 1):
```bash
cd firmware
pio run -e data -t upload
```

#### For Inference Streaming (Phase 3):
```bash
# Edit src/main.cpp with your WiFi credentials and PC IP address:
#   WIFI_SSID, WIFI_PASSWORD, UDP_SERVER_IP

cd firmware
pio run -e esp8266 -t upload
```

### 4. Run the System

```bash
# Terminal 1: Start inference server
uv run python src/server/inference_server.py

# Terminal 2: Start HCI controller (converts gestures to shortcuts)
# On Wayland, run with sudo for uinput backend:
sudo -E uv run python src/hci/hci_controller.py --backend uinput

# Or on X11:
uv run python src/hci/hci_controller.py
```

### 5. Open Dashboard (Optional)

Open `dashboard/gesture_dashboard.html` in a browser and connect to `localhost:9001`.

---

## Training Your Own Model

### Phase 1: Data Collection

Collect ~200 samples per gesture class:

```bash
# Flash data collection firmware
cd firmware && pio run -e data -t upload

# Run the logger (repeat for each gesture)
uv run python src/training/gesture_logger.py \
    --port /dev/ttyUSB0 \
    --gesture wave_right \
    --goal 200

uv run python src/training/gesture_logger.py \
    --port /dev/ttyUSB0 \
    --gesture wave_left \
    --goal 200

# Continue for: flick_up, flick_down, fist_hold, wrist_rotate, idle
```

**Tips for good data:**
- Perform each gesture naturally, with variation in speed and intensity
- Include the "idle" class for resting hand positions
- Aim for 200+ samples per class for robust accuracy

### Phase 2: Preprocessing

```bash
uv run python src/training/preprocess.py
```

This creates:
- `data/X_train.npy`, `X_val.npy`, `X_test.npy` — normalized input arrays
- `data/y_train.npy`, `y_val.npy`, `y_test.npy` — label arrays
- `data/label_map.json` — gesture name to index mapping
- `data/scaler.pkl` — StandardScaler for inference normalization

### Phase 3: Model Training

```bash
uv run python src/training/train_model.py
```

**Architecture:** Lightweight 1D-CNN (~15K parameters)
- 3x Conv1D blocks with BatchNorm and Dropout
- GlobalAveragePooling + Dense classification head
- Target: >92% validation accuracy

**Outputs:**
- `models/gesture_model.keras` — full Keras model
- `models/training_history.png` — loss/accuracy curves

### Phase 4: Quantization

```bash
uv run python src/training/quantize_export.py
```

**Outputs:**
- `models/gesture_model_int8.tflite` — INT8 quantized model (~4x smaller)
- `models/gesture_model.h` — C header for embedded deployment
- `models/quantization_report.txt` — accuracy comparison

---

## Configuration

### Gesture Bindings (`src/hci/hci_controller.py`)

Default gesture-to-shortcut mappings:

| Gesture | Shortcut | Description |
|---------|----------|-------------|
| `wave_right` | Ctrl+Right | Next desktop / tab |
| `wave_left` | Ctrl+Left | Previous desktop / tab |
| `flick_up` | Super | Show all windows |
| `flick_down` | Ctrl+Alt+D | Show desktop |
| `fist_hold` | Ctrl+L | Focus address bar / lock |
| `wrist_rotate` | Alt+Tab | Switch application |
| `idle` | (none) | Suppressed |

**Customize:** Create a `bindings.json` file:

```json
{
  "wave_right": {
    "keys": ["ctrl", "pagedown"],
    "description": "Next browser tab"
  },
  "flick_up": {
    "keys": ["f11"],
    "description": "Toggle fullscreen"
  }
}
```

Run with: `uv run python src/hci/hci_controller.py --config bindings.json`

### Inference Server Options

```bash
uv run python src/server/inference_server.py \
    --udp-port 5005 \
    --mqtt-host localhost \
    --mqtt-port 1883 \
    --model models/gesture_model_int8.tflite \
    --threshold 0.85
```

### Firmware Configuration

Edit `firmware/src/main.cpp`:

```cpp
#define WIFI_SSID       "YourNetwork"
#define WIFI_PASSWORD   "YourPassword"
#define UDP_SERVER_IP   "192.168.1.100"  // Your PC's IP
#define UDP_SERVER_PORT 5005
```

---

### Data Flow

1. **Sensor** → MPU6050 samples at 50Hz (ax, ay, az, gx, gy, gz)
2. **Windowing** → ESP8266 collects 50 samples (1 second window)
3. **Transmission** → UDP packet (1208 bytes) sent to PC
4. **Inference** → TFLite INT8 model predicts gesture class
5. **Publication** → Result published to MQTT topic
6. **Action** → HCI controller fires keyboard shortcut

---

## Troubleshooting

### Keyboard shortcuts not working on Wayland

Wayland blocks global keyboard injection from user-space. Solutions:

1. **Run as root with uinput backend:**
   ```bash
   sudo -E uv run python src/hci/hci_controller.py --backend uinput
   ```

2. **Use an X11 session** (log out, select "GNOME on Xorg" at login)

3. **Check the startup message** — the controller reports backend status

### No UDP packets received

1. Verify WiFi credentials in `firmware/src/main.cpp`
2. Check firewall: `sudo ufw allow 5005/udp`
3. Verify ESP8266 serial output: `pio device monitor -b 115200`
4. Ensure PC and ESP8266 are on the same network

### Low accuracy

- Collect more training samples (400+ per class)
- Ensure sensor is securely mounted
- Check for sensor drift; recalibrate if needed
- Verify preprocessing scaler matches training

### MQTT connection fails

```bash
# Check broker status
systemctl status mosquitto

# Test with CLI
mosquitto_pub -t "test" -m "hello"
mosquitto_sub -t "test"
```

---

## Performance

| Metric | Value |
|--------|-------|
| Sampling rate | 50 Hz |
| Window size | 50 samples (1 sec) |
| Model size (INT8) | ~14 KB |
| Inference latency | <10 ms |
| End-to-end latency | ~60 ms |
| Accuracy (7 classes) | >94% |

---

## 🔮 Future Enhancements

- [ ] On-device inference (TFLite Micro on ESP32)
- [ ] Bluetooth Low Energy support
- [ ] Additional gesture classes
- [ ] Gesture sequences / combos
- [ ] Mobile app for configuration
- [ ] Battery monitoring

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
