"""
=======================================================
  Gesture Recognition Wearable
  Step 1: Data Preprocessing
  
  Reads data/gesture_dataset.csv collected
  applies normalization and windowing, then saves
  ready-to-train numpy arrays.

  Requirements:
    pip install pandas numpy scikit-learn matplotlib

  Usage:
    python preprocess.py

  Outputs (saved to data/):
    X_train.npy, X_val.npy, X_test.npy
    y_train.npy, y_val.npy, y_test.npy
    label_map.json   (gesture name -> integer index)
    scaler.pkl       (StandardScaler for reuse in firmware prep)
=======================================================
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ── Config ──────────────────────────────────────────
ROOT_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR   = os.path.join(ROOT_DIR, "data")
INPUT_CSV   = os.path.join(DATA_DIR, "gesture_dataset.csv")
WINDOW_SIZE = 50    # must match firmware
FEATURES    = ["ax", "ay", "az", "gx", "gy", "gz"]
N_FEATURES  = len(FEATURES)  # 6

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# ────────────────────────────────────────────────────
def load_dataset(path: str) -> pd.DataFrame:
    print(f"Loading dataset from {path} ...")
    df = pd.read_csv(path)
    print(f"  Total rows     : {len(df)}")
    print(f"  Unique labels  : {df['label'].nunique()} -> {sorted(df['label'].unique())}")
    windows = df.groupby(["label", "sample_id"]).size()
    print(f"  Total windows  : {len(windows)}")
    print(f"  Windows/label  :")
    for label, cnt in df.groupby("label")["sample_id"].nunique().items():
        print(f"    {label:20s} : {cnt}")
    return df

def build_windows(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Groups by (label, sample_id) and stacks into shape
    (N_windows, WINDOW_SIZE, N_FEATURES).
    Returns X array and list of string labels.
    """
    X_list, y_list = [], []

    for (label, _sid), group in df.groupby(["label", "sample_id"]):
        window = group[FEATURES].values  # shape (WINDOW_SIZE, 6)
        if window.shape[0] != WINDOW_SIZE:
            continue  # skip incomplete windows
        X_list.append(window)
        y_list.append(label)

    X = np.array(X_list, dtype=np.float32)   # (N, 50, 6)
    return X, y_list

def normalize(X_train, X_val, X_test):
    """
    Fit StandardScaler on training data only.
    Reshape to (N*50, 6), fit, reshape back.
    """
    N_tr, W, F = X_train.shape

    scaler = StandardScaler()
    X_tr_flat = X_train.reshape(-1, F)
    scaler.fit(X_tr_flat)

    def transform(X):
        n = X.shape[0]
        return scaler.transform(X.reshape(-1, F)).reshape(n, W, F)

    return transform(X_train), transform(X_val), transform(X_test), scaler

def plot_sample(X: np.ndarray, y_labels: list, label: str, idx: int = 0):
    """Plot one window for visual sanity check."""
    mask = [i for i, l in enumerate(y_labels) if l == label]
    if not mask:
        return
    sample = X[mask[idx]]  # (50, 6)
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    fig.suptitle(f"Sample window — label: {label}", fontsize=12)

    for i, name in enumerate(["ax", "ay", "az"]):
        axes[0].plot(sample[:, i], label=name)
    axes[0].set_ylabel("Acceleration (g)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for i, name in enumerate(["gx", "gy", "gz"]):
        axes[1].plot(sample[:, i + 3], label=name)
    axes[1].set_ylabel("Gyroscope (°/s)")
    axes[1].set_xlabel("Sample index")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(DATA_DIR, f"sample_{label}.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Saved sample plot -> {out}")

# ── Main ─────────────────────────────────────────────
def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    df = load_dataset(INPUT_CSV)

    print("\nBuilding windows ...")
    X, y_strings = build_windows(df)
    print(f"  X shape : {X.shape}  (windows, timesteps, features)")
    print(f"  y count : {len(y_strings)}")

    # Encode labels to integers
    le = LabelEncoder()
    y_int = le.fit_transform(y_strings).astype(np.int32)
    label_map = {name: int(idx) for idx, name in enumerate(le.classes_)}
    print(f"\nLabel map: {label_map}")

    # Train / val / test split (stratified)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_int, test_size=TEST_RATIO, stratify=y_int, random_state=42
    )
    val_rel = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_rel, stratify=y_temp, random_state=42
    )

    print(f"\nSplit sizes:")
    print(f"  Train : {len(X_train)}")
    print(f"  Val   : {len(X_val)}")
    print(f"  Test  : {len(X_test)}")

    # Normalize
    print("\nNormalizing ...")
    X_train, X_val, X_test, scaler = normalize(X_train, X_val, X_test)
    print(f"  Mean (train flat): {X_train.reshape(-1,6).mean(axis=0).round(4)}")
    print(f"  Std  (train flat): {X_train.reshape(-1,6).std(axis=0).round(4)}")

    # Plot sanity check for first gesture
    first_label = list(label_map.keys())[0]
    plot_sample(X_train, [list(label_map.keys())[v] for v in y_train], first_label)

    # Save arrays
    print("\nSaving arrays ...")
    np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(DATA_DIR, "X_val.npy"),   X_val)
    np.save(os.path.join(DATA_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(DATA_DIR, "y_val.npy"),   y_val)
    np.save(os.path.join(DATA_DIR, "y_test.npy"),  y_test)

    with open(os.path.join(DATA_DIR, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    with open(os.path.join(DATA_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print("\n  Saved:")
    print("    data/X_train.npy, X_val.npy, X_test.npy")
    print("    data/y_train.npy, y_val.npy, y_test.npy")
    print("    data/label_map.json")
    print("    data/scaler.pkl")
    print("\n  Preprocessing complete. Run train_model.py next.\n")

if __name__ == "__main__":
    main()