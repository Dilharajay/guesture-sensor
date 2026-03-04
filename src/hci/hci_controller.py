#!/usr/bin/env python3
"""
=======================================================
  Gesture Recognition Wearable — HCI Layer
  Gesture to Keyboard Shortcut Controller

  Subscribes to Mosquitto MQTT and fires keyboard
  shortcuts whenever a gesture is detected.

  Requirements:
    pip install paho-mqtt pynput

  Usage:
    python hci_controller.py
    python hci_controller.py --config my_bindings.json
    python hci_controller.py --mqtt-host 192.168.1.105

  Key syntax:
    Single key   : "space", "enter", "f5", "esc"
    Modifier+key : ["ctrl", "c"]  or  ["alt", "tab"]
    Multi-key    : ["ctrl", "shift", "t"]

  All available key names:
    https://pynput.readthedocs.io/en/latest/keyboard.html
=======================================================
"""

import argparse
import json
import os
import sys
import time
import threading
import paho.mqtt.client as mqtt

# ── Keyboard backend bootstrap ──────────────────────────
def _get_cli_opt(flag: str, default: str) -> str:
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


_SESSION_TYPE = os.environ.get("XDG_SESSION_TYPE", "").lower()
_REQUESTED_BACKEND = os.environ.get(
    "HCI_KEYBOARD_BACKEND",
    _get_cli_opt("--backend", "auto")
).lower()
if _REQUESTED_BACKEND == "auto":
    _REQUESTED_BACKEND = "uinput" if _SESSION_TYPE == "wayland" else "xorg"

os.environ["PYNPUT_BACKEND"] = _REQUESTED_BACKEND
_BACKEND_WARNING = None

try:
    from pynput.keyboard import Key, Controller as KeyboardController
except Exception as e:
    if _REQUESTED_BACKEND != "uinput":
        raise
    _BACKEND_WARNING = (
        f"uinput backend unavailable ({str(e).splitlines()[0]}). "
        "Falling back to xorg."
    )
    os.environ["PYNPUT_BACKEND"] = "xorg"
    sys.modules.pop("pynput.keyboard", None)
    sys.modules.pop("pynput", None)
    from pynput.keyboard import Key, Controller as KeyboardController
    _REQUESTED_BACKEND = "xorg"

_ACTIVE_BACKEND = _REQUESTED_BACKEND

_keyboard = KeyboardController()

# Map string key names to pynput Key objects where needed
_SPECIAL_KEYS = {
    "ctrl":      Key.ctrl,   "shift":  Key.shift,
    "alt":       Key.alt,    "super":  Key.cmd,
    "enter":     Key.enter,  "esc":    Key.esc,
    "space":     Key.space,  "tab":    Key.tab,
    "backspace": Key.backspace,
    "up":        Key.up,     "down":   Key.down,
    "left":      Key.left,   "right":  Key.right,
    "f1":  Key.f1,  "f2":  Key.f2,  "f3":  Key.f3,  "f4":  Key.f4,
    "f5":  Key.f5,  "f6":  Key.f6,  "f7":  Key.f7,  "f8":  Key.f8,
    "f9":  Key.f9,  "f10": Key.f10, "f11": Key.f11, "f12": Key.f12,
    "audioraisevolume": Key.media_volume_up,
    "audiolowervolume": Key.media_volume_down,
    "audiovolumemute":  Key.media_volume_mute,
}

_KEY_HOLD_SEC = 0.03
_CHORD_SETTLE_SEC = 0.02


def _resolve_key(k: str):
    return _SPECIAL_KEYS.get(k.lower(), k)


def _tap_key(key):
    _keyboard.press(key)
    time.sleep(_KEY_HOLD_SEC)
    _keyboard.release(key)

# ── Default MQTT config ───────────────────────────────
MQTT_HOST  = "localhost"
MQTT_PORT  = 1883
MQTT_TOPIC = "wearable/gesture"

# ── Default gesture to shortcut bindings ─────────────
# Edit this dict or pass a --config JSON file to override.
#
# Each entry:
#   "gesture_label": {
#       "keys":        str or list   — key(s) to press
#       "description": str           — shown in terminal log
#       "min_conf":    float         — per-gesture confidence override (optional)
#   }
#
# Set "keys" to null to disable a gesture without removing it.

DEFAULT_BINDINGS = {
    "wave_right": {
        "keys":        ["ctrl", "right"],
        "description": "Next desktop / Next tab"
    },
    "wave_left": {
        "keys":        ["ctrl", "left"],
        "description": "Prev desktop / Prev tab"
    },
    "flick_up": {
        "keys":        ["super"],
        "description": "Show all windows (Super key)"
    },
    "flick_down": {
        "keys":        ["ctrl", "alt", "d"],
        "description": "Show desktop"
    },
    "fist_hold": {
        "keys":        ["ctrl", "l"],
        "description": "Focus address bar / lock screen"
    },
    "wrist_rotate": {
        "keys":        ["alt", "tab"],
        "description": "Switch application"
    },
    "idle": {
        "keys":        None,
        "description": "Rest state — no action"
    }
}

# ─────────────────────────────────────────────────────
def load_bindings(path: str) -> dict:
    if not os.path.exists(path):
        print(f"[Config] {path} not found. Using default bindings.")
        return DEFAULT_BINDINGS
    with open(path) as f:
        custom = json.load(f)
    merged = {**DEFAULT_BINDINGS, **custom}
    print(f"[Config] Loaded bindings from {path}")
    return merged


def save_default_config(path: str):
    with open(path, "w") as f:
        json.dump(DEFAULT_BINDINGS, f, indent=2)
    print(f"[Config] Default bindings saved to {path}")


def fire_shortcut(keys):
    """
    Press a key or key combination using pynput.
    keys can be a str ("f5") or a list (["ctrl", "shift", "t"]).
    """
    if isinstance(keys, str):
        _tap_key(_resolve_key(keys))
    elif isinstance(keys, list) and keys:
        resolved = [_resolve_key(k) for k in keys]
        if len(resolved) == 1:
            _tap_key(resolved[0])
            return

        modifiers = resolved[:-1]
        trigger = resolved[-1]

        for k in modifiers:
            _keyboard.press(k)
        time.sleep(_CHORD_SETTLE_SEC)

        _tap_key(trigger)
        time.sleep(_CHORD_SETTLE_SEC)

        for k in reversed(modifiers):
            _keyboard.release(k)


# ─────────────────────────────────────────────────────
class HCIController:
    def __init__(self, args, bindings: dict):
        self.args     = args
        self.bindings = bindings
        self.cooldown = args.cooldown / 1000.0   # ms -> seconds
        self.last_fire: dict[str, float] = {}
        self.total_fired = 0
        self._setup_mqtt()

    def _setup_mqtt(self):
        self.client = mqtt.Client(client_id="gesture_hci_controller")
        self.client.on_connect    = self._on_connect
        self.client.on_message    = self._on_message
        self.client.on_disconnect = self._on_disconnect

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            client.subscribe(MQTT_TOPIC)
            print(f"[MQTT] Connected and subscribed to {MQTT_TOPIC}")
        else:
            print(f"[MQTT] Connect failed rc={rc}")

    def _on_disconnect(self, client, userdata, rc):
        print(f"[MQTT] Disconnected (rc={rc}). Reconnecting...")

    def _on_message(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode())
        except Exception:
            return

        gesture    = data.get("gesture", "")
        confidence = float(data.get("confidence", 0))

        binding = self.bindings.get(gesture)
        if not binding:
            print(f"[HCI] Unknown gesture: {gesture} — add it to bindings to use it")
            return

        keys        = binding.get("keys")
        description = binding.get("description", "")
        min_conf    = float(binding.get("min_conf", self.args.threshold))

        # Suppressed gesture
        if keys is None:
            print(f"[HCI] {gesture} ({confidence*100:.0f}%) — suppressed")
            return

        # Confidence gate
        if confidence < min_conf:
            print(f"[HCI] {gesture} ({confidence*100:.0f}%) — below threshold {min_conf*100:.0f}%")
            return

        # Per-gesture cooldown
        now = time.monotonic()
        last = self.last_fire.get(gesture, 0)
        if (now - last) < self.cooldown:
            remaining = self.cooldown - (now - last)
            print(f"[HCI] {gesture} — cooldown {remaining:.2f}s remaining")
            return

        self.last_fire[gesture] = now
        self.total_fired += 1

        # Fire the shortcut
        try:
            fire_shortcut(keys)
            key_str = "+".join(keys) if isinstance(keys, list) else keys
            print(f"[HCI] {gesture} ({confidence*100:.0f}%)  →  {key_str}  [{description}]  total={self.total_fired}")
        except Exception as e:
            print(f"[HCI] Failed to fire shortcut for {gesture}: {e}")

    def run(self):
        print("\n" + "="*54)
        print("  Gesture HCI Controller")
        print(f"  Broker : {self.args.mqtt_host}:{self.args.mqtt_port}")
        print(f"  Topic  : {MQTT_TOPIC}")
        print(f"  Input  : {_ACTIVE_BACKEND} backend   Session: {_SESSION_TYPE or 'unknown'}")
        print(f"  Cooldown: {self.args.cooldown}ms   Threshold: {self.args.threshold*100:.0f}%")
        print("="*54)
        if _BACKEND_WARNING:
            print(f"[Input] {_BACKEND_WARNING}")
        if _SESSION_TYPE == "wayland" and _ACTIVE_BACKEND == "xorg":
            print("[Input] Wayland blocks many global shortcuts via xorg backend.")
            print("[Input] Run with root uinput backend for desktop-wide key injection:")
            print("        sudo -E uv run python scripts/HCI/hci_controller.py --backend uinput")
        print("\nActive bindings:")
        for label, b in self.bindings.items():
            keys = b.get("keys")
            if keys is None:
                key_str = "(suppressed)"
            elif isinstance(keys, list):
                key_str = "+".join(keys)
            else:
                key_str = keys
            print(f"  {label:16s}  →  {key_str:25s}  {b.get('description','')}")
        print("\n[System] Running. Move your hand to trigger shortcuts.")
        print("[System] Press Ctrl+C in this terminal to stop.\n")

        try:
            self.client.connect(self.args.mqtt_host, self.args.mqtt_port)
            self.client.loop_forever()
        except KeyboardInterrupt:
            print(f"\n[System] Stopped. Total shortcuts fired: {self.total_fired}")
        except Exception as e:
            print(f"[System] Fatal error: {e}")
            sys.exit(1)


# ── Entry point ───────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Gesture HCI Controller")
    parser.add_argument("--mqtt-host",  default=MQTT_HOST)
    parser.add_argument("--mqtt-port",  type=int, default=MQTT_PORT)
    parser.add_argument("--backend",    choices=["auto", "xorg", "uinput"], default="auto",
                        help="Keyboard injection backend (auto uses uinput on Wayland, xorg otherwise)")
    parser.add_argument("--config",     default="bindings.json",
                        help="Path to custom gesture bindings JSON file")
    parser.add_argument("--threshold",  type=float, default=0.85,
                        help="Global minimum confidence to fire a shortcut")
    parser.add_argument("--cooldown",   type=int, default=800,
                        help="Milliseconds between repeated gesture fires")
    parser.add_argument("--save-config", action="store_true",
                        help="Save default bindings.json and exit")
    args = parser.parse_args()

    if args.save_config:
        save_default_config(args.config)
        return

    bindings   = load_bindings(args.config)
    controller = HCIController(args, bindings)
    controller.run()


if __name__ == "__main__":
    main()
