#!/usr/bin/env python3
"""
=======================================================
  Gesture Recognition Wearable — Phase 01
  Python Data Logger
  Reads gesture windows from ESP8266 Serial and saves
  labeled samples to a structured CSV dataset.

  Requirements:
    pip install pyserial pandas

  Usage:
    python gesture_logger.py --port COM3 --gesture wave_right
    python gesture_logger.py --port /dev/ttyUSB0 --gesture fist_hold

  The script:
    1. Opens the Serial port to the ESP8266
    2. Prompts you to perform a gesture on keypress
    3. Sends 's' to the ESP to trigger a 50-sample window
    4. Collects and labels the window
    5. Appends it to data/gesture_dataset.csv
    6. Repeats until you have enough samples (default goal: 200)
=======================================================
"""

import argparse
import os
import time
import serial
import pandas as pd
from datetime import datetime

# ── Config ──────────────────────────────────────────
BAUD_RATE    = 115200
WINDOW_SIZE  = 50
SAMPLE_GOAL  = 200   # samples to collect per gesture
ROOT_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR   = os.path.join(ROOT_DIR, "data")
OUTPUT_CSV   = os.path.join(DATA_DIR, "gesture_dataset.csv")

COLUMNS = ["label", "sample_id", "timestamp_ms",
           "ax", "ay", "az", "gx", "gy", "gz"]

# ── Helpers ──────────────────────────────────────────
def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

def load_existing(gesture_label: str) -> int:
    """Returns how many samples already exist for this gesture."""
    if not os.path.exists(OUTPUT_CSV):
        return 0
    df = pd.read_csv(OUTPUT_CSV)
    mask = df["label"] == gesture_label
    # Each window = WINDOW_SIZE rows, count unique sample_ids
    unique = df[mask]["sample_id"].nunique()
    return unique

def collect_window(ser: serial.Serial, label: str, sample_id: int) -> list[dict]:
    """
    Sends 's' to the ESP, reads one WINDOW_SIZE block,
    and returns a list of row dicts ready for the DataFrame.
    """
    rows = []

    # Flush any stale bytes
    ser.reset_input_buffer()

    # Trigger the ESP window
    ser.write(b's')

    in_window = False
    timeout_start = time.time()

    while True:
        if time.time() - timeout_start > 10:
            print("  ⚠  Timeout waiting for window. Skipping.")
            return []

        raw = ser.readline().decode("utf-8", errors="ignore").strip()
        if not raw:
            continue

        if raw == "# WINDOW_START":
            in_window = True
            continue

        if raw in ("# WINDOW_END", "# WINDOW_DISCARD"):
            break

        if in_window and not raw.startswith("#"):
            parts = raw.split(",")
            if len(parts) != 7:
                continue
            try:
                rows.append({
                    "label":        label,
                    "sample_id":    sample_id,
                    "timestamp_ms": int(parts[0]),
                    "ax":           float(parts[1]),
                    "ay":           float(parts[2]),
                    "az":           float(parts[3]),
                    "gx":           float(parts[4]),
                    "gy":           float(parts[5]),
                    "gz":           float(parts[6]),
                })
            except ValueError:
                continue

    return rows

def append_to_csv(rows: list[dict]):
    df_new = pd.DataFrame(rows, columns=COLUMNS)
    if os.path.exists(OUTPUT_CSV):
        df_new.to_csv(OUTPUT_CSV, mode="a", header=False, index=False)
    else:
        df_new.to_csv(OUTPUT_CSV, index=False)

def print_progress(label: str, done: int, goal: int):
    bar_len = 30
    filled  = int(bar_len * done / goal)
    bar     = "█" * filled + "░" * (bar_len - filled)
    pct     = int(100 * done / goal)
    print(f"  [{bar}] {done}/{goal} ({pct}%)  label={label}")

# ── Main ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Gesture Data Logger")
    parser.add_argument("--port",    required=True,  help="Serial port (e.g. COM3 or /dev/ttyUSB0)")
    parser.add_argument("--gesture", required=True,  help="Gesture label (e.g. wave_right, fist_hold)")
    parser.add_argument("--goal",    type=int, default=SAMPLE_GOAL, help="Number of windows to collect")
    parser.add_argument("--baud",    type=int, default=BAUD_RATE)
    args = parser.parse_args()

    label = args.gesture.lower().replace(" ", "_")
    goal  = args.goal

    ensure_data_dir()

    print(f"\n{'='*52}")
    print(f"  Gesture Logger  |  label: {label}")
    print(f"  Port: {args.port}  |  Goal: {goal} windows")
    print(f"{'='*52}\n")

    already_done = load_existing(label)
    if already_done >= goal:
        print(f"  ✓ Already have {already_done} windows for '{label}'. Done!")
        return

    print(f"  Resuming from window #{already_done + 1}\n")

    try:
        ser = serial.Serial(args.port, args.baud, timeout=5)
    except serial.SerialException as e:
        print(f"  ✗ Could not open port: {e}")
        return

    time.sleep(2)  # Wait for ESP8266 to boot / reset
    ser.reset_input_buffer()
    print("  Connected to ESP8266.\n")

    sample_id = already_done
    failed    = 0

    try:
        while sample_id < goal:
            remaining = goal - sample_id
            print(f"  Ready to collect window {sample_id + 1}/{goal}")
            print(f"  Press ENTER to perform gesture  '{label}'  (or type 'q' + ENTER to quit)")

            user = input("  > ").strip().lower()
            if user == "q":
                print("\n  Stopped by user.")
                break

            print("  Collecting...")
            rows = collect_window(ser, label, sample_id)

            if len(rows) == WINDOW_SIZE:
                append_to_csv(rows)
                sample_id += 1
                failed = 0
                print(f"  ✓ Window {sample_id} saved.\n")
                print_progress(label, sample_id, goal)
                print()
            else:
                failed += 1
                print(f"  ✗ Incomplete window ({len(rows)} samples). Try again. (fail #{failed})\n")
                if failed >= 5:
                    print("  ⚠  5 consecutive failures. Check hardware and retry.")
                    break

    except KeyboardInterrupt:
        print("\n\n  Interrupted.")

    finally:
        ser.close()
        total = load_existing(label)
        print(f"\n  Session complete.  Total windows for '{label}': {total}/{goal}")
        if os.path.exists(OUTPUT_CSV):
            df = pd.read_csv(OUTPUT_CSV)
            summary = df.groupby("label")["sample_id"].nunique()
            print("\n  Full dataset summary:")
            print(summary.to_string())
        print(f"\n  Saved to: {OUTPUT_CSV}\n")

if __name__ == "__main__":
    main()
