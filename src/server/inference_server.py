"""
=======================================================
  Gesture Recognition Wearable — Revised Phase 03
  PC Inference Server

  Receives raw IMU windows from the ESP8266 over UDP,
  runs the INT8 TFLite model locally, and publishes
  the recognized gesture to Mosquitto via MQTT.

  Architecture:
    ESP8266 (UDP) --> this server --> Mosquitto (MQTT)
                                          |
                              Home Assistant / Presenter

  Requirements:
    pip install numpy tflite-runtime paho-mqtt

  Note: use tflite-runtime (lightweight) instead of
  full tensorflow. Install with:
    pip install tflite-runtime
  Or if that fails on your platform:
    pip install tensorflow   (full package, also works)

  Usage:
    python inference_server.py

  Optional flags:
    --udp-port  5005          (default)
    --mqtt-host 192.168.1.105 (default: localhost)
    --mqtt-port 1883          (default)
    --model     models/gesture_model_int8.tflite
    --threshold 0.85
=======================================================
"""

import argparse
import json
import pickle
import socket
import struct
import threading
import time
import numpy as np
import os

# ── Try tflite-runtime first, fall back to full TF ───
try:
    import tflite_runtime.interpreter as tflite
    print("[ML] Using tflite-runtime (lightweight)")
except ImportError:
    import tensorflow.lite as tflite
    print("[ML] Using tensorflow.lite (full package)")

import paho.mqtt.client as mqtt

# ── Config defaults ───────────────────────────────────
UDP_HOST    = "0.0.0.0"
UDP_PORT    = 5005
MQTT_HOST   = "localhost"
MQTT_PORT   = 1883
MQTT_TOPIC  = "wearable/gesture"
ROOT_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DATA_PATH = os.path.join(ROOT_DIR, "data")
MODEL_PATH  = os.path.join(MODELS_DIR,"gesture_model_int8.tflite")
SCALER_PATH = os.path.join(DATA_PATH, "scaler.pkl")
LABEL_MAP   = os.path.join(DATA_PATH, "label_map.json")
WINDOW_SIZE = 50
N_FEATURES  = 6

# ── UDP packet format ─────────────────────────────────
# 4 (seq) + 4 (ts) + 50*6*4 (floats) = 1208 bytes
PACKET_SIZE  = 4 + 4 + WINDOW_SIZE * N_FEATURES * 4
HEADER_FMT   = "<II"   # little-endian uint32 seq, uint32 ts
HEADER_BYTES = 8


class InferenceServer:
    def __init__(self, args):
        self.args        = args
        self.running     = False
        self.seq_last    = -1
        self.dropped     = 0
        self.total       = 0
        self.cooldown_ms = 800
        self.last_fire   = 0

        self._load_model()
        self._load_scaler()
        self._load_labels()
        self._setup_mqtt()
        self._setup_udp()

    # ── Model ─────────────────────────────────────────
    def _load_model(self):
        print(f"[ML] Loading model: {self.args.model}")
        self.interpreter = tflite.Interpreter(model_path=self.args.model)
        self.interpreter.allocate_tensors()

        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        in_shape = self.input_details[0]['shape']
        print(f"[ML] Input  shape : {in_shape}")
        print(f"[ML] Input  dtype : {self.input_details[0]['dtype']}")
        print(f"[ML] Output shape : {self.output_details[0]['shape']}")

        self.is_int8 = (self.input_details[0]['dtype'] == np.int8)
        if self.is_int8:
            self.in_scale, self.in_zp   = self.input_details[0]['quantization']
            self.out_scale, self.out_zp = self.output_details[0]['quantization']
            print(f"[ML] INT8 model  in_scale={self.in_scale:.6f}  out_scale={self.out_scale:.6f}")
        else:
            print("[ML] Float32 model")

    # ── Scaler ────────────────────────────────────────
    def _load_scaler(self):
        with open(self.args.scaler, "rb") as f:
            self.scaler = pickle.load(f)
        print(f"[ML] Scaler loaded from {self.args.scaler}")

    # ── Labels ────────────────────────────────────────
    def _load_labels(self):
        with open(self.args.labels) as f:
            label_map = json.load(f)
        # Invert {name: idx} -> {idx: name}
        self.labels = {int(v): k for k, v in label_map.items()}
        print(f"[ML] Labels: {self.labels}")

    # ── MQTT ──────────────────────────────────────────
    def _setup_mqtt(self):
        self.mqtt_client = mqtt.Client(client_id="gesture_inference_server")
        self.mqtt_client.on_connect    = self._on_mqtt_connect
        self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
        try:
            self.mqtt_client.connect(self.args.mqtt_host, self.args.mqtt_port)
            self.mqtt_client.loop_start()
        except Exception as e:
            print(f"[MQTT] Connection failed: {e}. Will retry...")

    def _on_mqtt_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"[MQTT] Connected to {self.args.mqtt_host}:{self.args.mqtt_port}")
        else:
            print(f"[MQTT] Connect failed rc={rc}")

    def _on_mqtt_disconnect(self, client, userdata, rc):
        print(f"[MQTT] Disconnected (rc={rc}). Auto-reconnecting...")

    def _publish(self, gesture: str, confidence: float):
        payload = json.dumps({"gesture": gesture, "confidence": round(float(confidence), 3)})
        self.mqtt_client.publish(MQTT_TOPIC, payload, qos=0)
        print(f"[MQTT] Published: {payload}")

    # ── UDP ───────────────────────────────────────────
    def _setup_udp(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((UDP_HOST, self.args.udp_port))
        self.sock.settimeout(1.0)
        print(f"[UDP] Listening on {UDP_HOST}:{self.args.udp_port}")

    # ── Inference ─────────────────────────────────────
    def infer(self, window: np.ndarray) -> tuple[str, float]:
        """
        window shape: (50, 6) float32, raw IMU values
        Returns (gesture_label, confidence)
        """
        # Normalize using the same scaler fitted in Phase 02
        flat = window.reshape(-1, N_FEATURES)
        flat = self.scaler.transform(flat).astype(np.float32)
        window_norm = flat.reshape(1, WINDOW_SIZE, N_FEATURES)

        if self.is_int8:
            # Quantize input
            inp = (window_norm / self.in_scale + self.in_zp).astype(np.int8)
        else:
            inp = window_norm

        self.interpreter.set_tensor(self.input_details[0]['index'], inp)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])

        if self.is_int8:
            scores = (output.astype(np.float32) - self.out_zp) * self.out_scale
        else:
            scores = output.astype(np.float32)

        scores = scores.flatten()
        best_idx   = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        return self.labels[best_idx], best_score, scores

    # ── Packet parser ─────────────────────────────────
    def parse_packet(self, data: bytes):
        if len(data) != PACKET_SIZE:
            print(f"[UDP] Bad packet size: {len(data)} (expected {PACKET_SIZE})")
            return None, None, None

        seq, ts = struct.unpack_from(HEADER_FMT, data, 0)
        floats  = np.frombuffer(data, dtype=np.float32, offset=HEADER_BYTES)
        window  = floats.reshape(WINDOW_SIZE, N_FEATURES)
        return seq, ts, window

    # ── Main loop ─────────────────────────────────────
    def run(self):
        self.running = True
        print(f"\n[Server] Inference server running. Threshold={self.args.threshold}")
        print("[Server] Waiting for ESP8266 UDP packets...\n")

        while self.running:
            try:
                data, addr = self.sock.recvfrom(2048)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[UDP] Recv error: {e}")
                continue

            seq, ts, window = self.parse_packet(data)
            if window is None:
                continue

            self.total += 1

            # Detect dropped packets
            if self.seq_last >= 0 and seq != self.seq_last + 1:
                gap = seq - self.seq_last - 1
                self.dropped += gap

            self.seq_last = seq

            # Run inference
            t0 = time.perf_counter()
            label, confidence, scores = self.infer(window)
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000

            # Debug line
            score_str = "  ".join(
                f"{self.labels[i]}:{scores[i]:.2f}"
                for i in range(len(scores))
            )
            print(f"[#{seq:06d}] {score_str}  | {latency_ms:.1f}ms")

            # Fire if above threshold and cooldown elapsed.
            # Never publish idle -- it is a catch-all rest class, not an action.
            now_ms = int(time.time() * 1000)
            if label == "idle":
                print(f"  --- idle  ({confidence*100:.0f}%)  suppressed")
            elif (confidence >= self.args.threshold and
                    (now_ms - self.last_fire) >= self.cooldown_ms):
                self.last_fire = now_ms
                print(f"  >>> GESTURE: {label}  ({confidence*100:.0f}%)")
                self._publish(label, confidence)

        self.sock.close()
        self.mqtt_client.loop_stop()
        print(f"\n[Server] Stopped. Total={self.total}  Dropped={self.dropped}")


# ── Entry point ───────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Gesture Inference Server")
    parser.add_argument("--udp-port",  type=int,   default=UDP_PORT)
    parser.add_argument("--mqtt-host", type=str,   default=MQTT_HOST)
    parser.add_argument("--mqtt-port", type=int,   default=MQTT_PORT)
    parser.add_argument("--model",     type=str,   default=MODEL_PATH)
    parser.add_argument("--scaler",    type=str,   default=SCALER_PATH)
    parser.add_argument("--labels",    type=str,   default=LABEL_MAP)
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()

    server = InferenceServer(args)
    try:
        server.run()
    except KeyboardInterrupt:
        print("\n[Server] Interrupted by user.")
        server.running = False


if __name__ == "__main__":
    main()
