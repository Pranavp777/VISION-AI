"""
TensorFlow/Keras Model Training & Export Script (`ml/train.py`).

Supports three modes:
1. `--mode bootstrap` (default):
   Builds and exports the production-ready `ml/model.keras` and ensures
   `ml/class_names.json` is synchronized. Fast and requires no external dataset download.
2. `--mode cifar10`:
   Downloads the CIFAR-10 dataset via `tf.keras.datasets.cifar10`, trains a
   Convolutional Neural Network with data augmentation, and saves `ml/model.keras`
   and `ml/class_names.json`.
3. `--mode custom --data-dir /path/to/dataset`:
   Trains a transfer-learning MobileNetV2 classifier on a directory of images
   organized as `data_dir/<class_name>/*.jpg`, saving `ml/model.keras` and
   updating `ml/class_names.json`.

Usage:
    python ml/train.py --mode bootstrap
    python ml/train.py --mode cifar10 --epochs 10
    python ml/train.py --mode custom --data-dir ./dataset --epochs 15
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ML_DIR = Path(__file__).resolve().parent
MODEL_PATH = ML_DIR / "model.keras"
CLASS_NAMES_PATH = ML_DIR / "class_names.json"

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def save_class_names(class_list: list[str]) -> None:
    """Save class index mapping to `ml/class_names.json`."""
    mapping = {str(i): name for i, name in enumerate(class_list)}
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as fp:
        json.dump(mapping, fp, indent=2)
    print(f"[+] Saved {len(class_list)} classes to {CLASS_NAMES_PATH}")


def build_cnn_model(input_shape: tuple[int, int, int], num_classes: int) -> keras.Model:
    """Build a modern CNN architecture with BatchNormalization and GlobalAveragePooling."""
    inputs = keras.Input(shape=input_shape, name="image_input")

    x = layers.Conv2D(32, (3, 3), strides=2, padding="same", activation="relu", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    x = layers.Dropout(0.25, name="dropout")(x)
    x = layers.Dense(64, activation="relu", name="fc1")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="VisionClassify_CNN")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_bootstrap() -> None:
    """Initialize and export `ml/model.keras` matching `ml/class_names.json`."""
    sys.path.insert(0, str(ML_DIR.parent))
    from ml.predictor import _ModelSingleton

    class_names = _ModelSingleton.load_class_names()
    model = _ModelSingleton._build_and_save_default_model(tf, len(class_names))
    print(f"[+] Model summary ({len(class_names)} classes):")
    model.summary()
    print(f"[+] Successfully exported production model to {MODEL_PATH}")


def train_cifar10(epochs: int = 10, batch_size: int = 64) -> None:
    """Train a CNN classifier on CIFAR-10 and save `ml/model.keras`."""
    print("[*] Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    save_class_names(CIFAR10_CLASSES)
    model = build_cnn_model(input_shape=(32, 32, 3), num_classes=len(CIFAR10_CLASSES))

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
        ),
    ]

    print(f"[*] Training on {len(x_train)} images for {epochs} epochs...")
    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"[+] Test accuracy: {acc * 100:.2f}% | Test loss: {loss:.4f}")
    model.save(str(MODEL_PATH))
    print(f"[+] Saved trained model to {MODEL_PATH}")


def train_custom_directory(data_dir: str, epochs: int = 15, batch_size: int = 32) -> None:
    """Train a MobileNetV2 transfer-learning model on a custom folder dataset."""
    dataset_path = Path(data_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    img_size = (224, 224)
    train_ds = keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
    )

    class_names = list(train_ds.class_names)
    save_class_names(class_names)

    normalization = layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (normalization(x), y)).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (normalization(x), y)).prefetch(tf.data.AUTOTUNE)

    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(224, 224, 3), name="image_input")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(len(class_names), activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="VisionClassify_MobileNetV2")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    model.save(str(MODEL_PATH))
    print(f"[+] Saved custom trained model to {MODEL_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or export VisionClassify (PREDICTOR) Keras model.")
    parser.add_argument(
        "--mode",
        choices=["bootstrap", "cifar10", "custom"],
        default="bootstrap",
        help="Training mode: 'bootstrap' (instant production model), 'cifar10', or 'custom'.",
    )
    parser.add_argument("--data-dir", type=str, default="", help="Path to custom dataset directory.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    args = parser.parse_args()

    if args.mode == "bootstrap":
        train_bootstrap()
    elif args.mode == "cifar10":
        train_cifar10(epochs=args.epochs, batch_size=args.batch_size)
    elif args.mode == "custom":
        if not args.data_dir:
            parser.error("--data-dir is required when --mode=custom")
        train_custom_directory(args.data_dir, epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
