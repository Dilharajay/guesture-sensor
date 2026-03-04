#!/usr/bin/env python3
"""
=======================================================
  Gesture Recognition Wearable 
  Step 2: Model Training (1D-CNN)

  Trains a lightweight 1D Convolutional Neural Network
  on the preprocessed gesture windows. Targets >92%
  val accuracy. Model is kept small enough for TFLite
  Micro on the ESP8266 (< 30KB weights).

  Usage:
    python train_model.py

  Outputs:
    models/gesture_model.keras   (full Keras model)
    models/training_history.png  (loss + accuracy curves)
=======================================================
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ── Config ──────────────────────────────────────────
ROOT_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR   = os.path.join(ROOT_DIR, "data")
INPUT_CSV   = os.path.join(DATA_DIR, "gesture_dataset.csv")
MODELS_DIR   = os.path.join(ROOT_DIR,"models")
WINDOW_SIZE  = 50
N_FEATURES   = 6

EPOCHS       = 80
BATCH_SIZE   = 32
LEARNING_RATE = 1e-3
DROPOUT_RATE  = 0.3

# ────────────────────────────────────────────────────
def load_data():
    print("Loading preprocessed arrays ...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_val   = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_val   = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    with open(os.path.join(DATA_DIR, "label_map.json")) as f:
        label_map = json.load(f)

    n_classes = len(label_map)
    print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    print(f"  Classes: {n_classes}  ->  {list(label_map.keys())}")
    return X_train, X_val, X_test, y_train, y_val, y_test, n_classes, label_map

def build_model(n_classes: int) -> keras.Model:
    """
    Lightweight 1D-CNN architecture.
    Input  : (50, 6)  - 50 timesteps, 6 IMU axes
    Output : (n_classes,) softmax

    Architecture breakdown:
      Conv1D x2 (feature extraction)  ~16KB weights
      GlobalAvgPool (removes temporal dimension)
      Dense x2 (classification head)
    Total params target: < 15,000
    """
    inputs = keras.Input(shape=(WINDOW_SIZE, N_FEATURES), name="imu_input")

    # Block 1
    x = layers.Conv1D(32, kernel_size=5, padding="same", activation="relu",
                      name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)
    x = layers.Dropout(DROPOUT_RATE, name="drop1")(x)

    # Block 2
    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu",
                      name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool2")(x)
    x = layers.Dropout(DROPOUT_RATE, name="drop2")(x)

    # Block 3 (optional depth, remove if overfitting on small dataset)
    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu",
                      name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Dropout(DROPOUT_RATE, name="drop3")(x)

    # Temporal pooling
    x = layers.GlobalAveragePooling1D(name="gap")(x)

    # Classification head
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(DROPOUT_RATE, name="drop4")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs, outputs, name="gesture_cnn")
    return model

def plot_history(history, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training History", fontsize=13)

    axes[0].plot(history.history["loss"],     label="Train loss")
    axes[0].plot(history.history["val_loss"], label="Val loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["accuracy"],     label="Train acc")
    axes[1].plot(history.history["val_accuracy"], label="Val acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Saved training plot -> {out_path}")

def evaluate(model, X_test, y_test, label_map):
    print("\nEvaluating on test set ...")
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    from sklearn.metrics import classification_report, confusion_matrix
    names = list(label_map.keys())
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=names))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    acc = np.mean(y_pred == y_test)
    print(f"\nTest accuracy: {acc*100:.2f}%")
    if acc < 0.92:
        print("  ⚠  Below 92% target. Consider collecting more samples or tuning hyperparameters.")
    else:
        print("  ✓  Accuracy target met.")
    return acc

# ── Main ─────────────────────────────────────────────
def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test, n_classes, label_map = load_data()

    print(f"\nBuilding model ...")
    model = build_model(n_classes)
    model.summary()

    total_params = model.count_params()
    size_kb = total_params * 4 / 1024
    print(f"\n  Estimated float32 weight size: {size_kb:.1f} KB")

    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-5,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODELS_DIR, "best_checkpoint.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=0
        )
    ]

    print(f"\nTraining for up to {EPOCHS} epochs ...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    plot_history(history, os.path.join(MODELS_DIR, "training_history.png"))
    evaluate(model, X_test, y_test, label_map)

    # Save final model
    save_path = os.path.join(MODELS_DIR, "gesture_model.keras")
    model.save(save_path)
    print(f"\n  Model saved -> {save_path}")
    print("  Run quantize_export.py next to convert to TFLite.\n")

if __name__ == "__main__":
    main()