"""
Train the deepfake detection model.

Dataset layout:
    dataset/real/<video_id>/frame_0000.jpg ...
    dataset/fake/<video_id>/frame_0000.jpg ...

Usage:
    python train.py --dataset_root dataset/ --epochs 30
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

from config import CFG
from model import build_model
from preprocessing import normalize


class FaceDataset(Sequence):
    def __init__(self, samples, batch_size=16, augment=False, shuffle=True):
        self.samples    = samples
        self.batch_size = batch_size
        self.augment    = augment
        self.shuffle    = shuffle
        self.indices    = list(range(len(samples)))
        if shuffle:
            random.shuffle(self.indices)

    def __len__(self):
        return max(1, len(self.samples) // self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X = np.zeros((len(batch_idx), CFG.face_size, CFG.face_size, 3), dtype="float32")
        y = np.zeros(len(batch_idx), dtype="float32")
        for i, si in enumerate(batch_idx):
            path, label = self.samples[si]
            img = cv2.imread(str(path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (CFG.face_size, CFG.face_size))
            if self.augment and random.random() < 0.5:
                img = img[:, ::-1, :]   # horizontal flip
            X[i] = normalize(img)
            y[i] = float(label)
        return X, y.reshape(-1, 1)


def build_splits(dataset_root):
    samples = []
    for cls, label in [("real", 0), ("fake", 1)]:
        cls_dir = Path(dataset_root) / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"Missing directory: {cls_dir}")
        for p in sorted(cls_dir.rglob("*.jpg")):
            samples.append((p, label))
    random.seed(CFG.seed if hasattr(CFG, 'seed') else 42)
    random.shuffle(samples)
    n      = len(samples)
    n_val  = int(n * 0.15)
    n_test = int(n * 0.10)
    return samples[n_val + n_test:], samples[:n_val], samples[n_val:n_val + n_test]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--epochs",      type=int, default=30)
    parser.add_argument("--batch_size",  type=int, default=16)
    args = parser.parse_args()

    train_s, val_s, _ = build_splits(args.dataset_root)
    print(f"Train: {len(train_s)}  Val: {len(val_s)}")

    train_gen = FaceDataset(train_s, batch_size=args.batch_size, augment=True)
    val_gen   = FaceDataset(val_s,   batch_size=args.batch_size, augment=False)

    model = build_model()
    model.summary()

    Path(CFG.weights_dir if hasattr(CFG, 'weights_dir') else "weights").mkdir(exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            CFG.model_path, monitor="val_auc", mode="max",
            save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=6,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1,
        ),
    ]

    # Phase 1: warmup — frozen backbone
    print("\nPhase 1: Warmup (backbone frozen)")
    model.fit(train_gen, validation_data=val_gen,
              epochs=5, callbacks=callbacks, verbose=1)

    # Phase 2: fine-tune top layers
    print("\nPhase 2: Fine-tuning top layers")
    for layer in model.layers:
        if hasattr(layer, "layers"):
            for sub in layer.layers[-20:]:
                sub.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    model.fit(train_gen, validation_data=val_gen,
              epochs=args.epochs, initial_epoch=5,
              callbacks=callbacks, verbose=1)

    print(f"\nModel saved to {CFG.model_path}")


if __name__ == "__main__":
    main()