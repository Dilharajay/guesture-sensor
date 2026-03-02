# Gesture Sensor

A gesture recognition wearable system — **Phase 01: Data Collection**. An ESP8266 reads accelerometer and gyroscope data from an MPU6050 sensor and streams it over serial, while a Python logger captures labeled gesture windows into a structured CSV dataset for downstream ML training.

## Hardware

| Component | Details |
|-----------|---------|
| Board | ESP8266 (NodeMCU / Wemos D1) |
| Sensor | MPU6050 6-axis IMU (I2C: SDA → D2, SCL → D1) |
| Baud rate | 115200 |

### Sensor Configuration

- **Accelerometer**: ±2 g range (16384 LSB/g)
- **Gyroscope**: ±250 °/s (131 LSB/°/s)
- **Sample rate**: 50 Hz
- **Window size**: 50 samples (1 second per gesture window)

## Project Structure

```
├── firmware/
│   └── gesture_collector.ino   # ESP8266 Arduino firmware
├── data/
│   └── gesture_dataset.csv     # Collected gesture samples
├── gesture_logger.py           # Python serial data logger
├── pyproject.toml
└── uv.lock
```

## Getting Started

### 1. Flash the Firmware

1. Open `firmware/gesture_collector.ino` in the Arduino IDE.
2. Install the ESP8266 board package if you haven't already.
3. Select your board (NodeMCU 1.0 or Wemos D1) and serial port.
4. Upload the sketch.

### 2. Install Python Dependencies

Requires Python ≥ 3.12. This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

### 3. Collect Gesture Data

```bash
uv run gesture_logger.py --port /dev/ttyUSB0 --gesture wave_right
```

**Arguments:**

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--port` | Yes | — | Serial port (e.g. `COM3`, `/dev/ttyUSB0`) |
| `--gesture` | Yes | — | Gesture label (e.g. `wave_right`, `fist_hold`) |
| `--goal` | No | 200 | Number of windows to collect |
| `--baud` | No | 115200 | Serial baud rate |

The logger will:

1. Connect to the ESP8266 over serial.
2. Prompt you to press **Enter** before each gesture.
3. Send a trigger to the ESP to record a 50-sample window.
4. Label and append the window to `data/gesture_dataset.csv`.
5. Display a progress bar and repeat until the goal is reached.

Collection resumes automatically if `gesture_dataset.csv` already contains data for the given label.

## Data Format

The dataset is saved as `data/gesture_dataset.csv` with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `label` | string | Gesture name (e.g. `wave_right`) |
| `sample_id` | int | Window index for this label |
| `timestamp_ms` | int | Millisecond timestamp from the ESP8266 |
| `ax` | float | Accelerometer X (g) |
| `ay` | float | Accelerometer Y (g) |
| `az` | float | Accelerometer Z (g) |
| `gx` | float | Gyroscope X (°/s) |
| `gy` | float | Gyroscope Y (°/s) |
| `gz` | float | Gyroscope Z (°/s) |

Each gesture window consists of 50 consecutive rows sharing the same `label` and `sample_id`.

## Serial Protocol

The firmware accepts single-character commands over serial:

- **`s`** — Start recording a gesture window
- **`x`** — Discard the current window

Responses are prefixed with `#`:

```
# WINDOW_START       → Recording began
<csv data lines>     → 50 lines of timestamp,ax,ay,az,gx,gy,gz
# WINDOW_END         → Window completed successfully
# WINDOW_DISCARD     → Window was cancelled
```
