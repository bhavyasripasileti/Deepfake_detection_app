"""
Train the deepfake detection model — CNN + LSTM two-stage pipeline.

Dataset layout (image frames extracted from videos):
    dataset/real/<video_id>/frame_0000.jpg ...
    dataset/fake/<video_id>/frame_0000.jpg ...

Two training stages:
  Stage 1 — CNN image model fine-tuning  (same as before)
  Stage 2 — BiLSTM video model training  (new — uses pre-extracted CNN features)

Usage:
    # Full pipeline
    python train.py --dataset_root dataset/ --epochs 30

    # Skip CNN stage (already trained), only train LSTM
    python train.py --dataset_root dataset/ --epochs 30 --skip_cnn

    # Separate video folder
    python train.py --dataset_root dataset/ --video_root videos/ --skip_cnn
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
from preprocessing import normalize, extract_frame_features_for_lstm


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 helpers — image dataset (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

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
        all_files = sorted(cls_dir.rglob("*.jpg"))[:1000]
        for p in all_files:
            samples.append((p, label))
    random.seed(42)
    random.shuffle(samples)
    n      = len(samples)
    n_val  = int(n * 0.15)
    n_test = int(n * 0.10)
    return samples[n_val + n_test:], samples[:n_val], samples[n_val:n_val + n_test]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 helpers — video feature extraction + LSTM dataset
# ─────────────────────────────────────────────────────────────────────────────

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def collect_video_paths(video_root):
    """
    Collect video paths and labels from:
        video_root/real/*.mp4
        video_root/fake/*.mp4
    Returns list of (path, label) tuples.
    """
    samples = []
    for cls, label in [("real", 0), ("fake", 1)]:
        folder = Path(video_root) / cls
        if not folder.exists():
            print(f"[WARN] {folder} not found — skipping.")
            continue
        for f in folder.iterdir():
            if f.suffix.lower() in VIDEO_EXTS:
                samples.append((str(f), label))
    random.seed(42)
    random.shuffle(samples)
    return samples


def preextract_video_features(video_samples, cnn_extractor, num_frames):
    """
    Pre-extract CNN frame features for all videos.
    Returns X: (N, num_frames, feature_dim), y: (N,)
    Skips videos where no faces are detected.
    """
    X, y = [], []
    total = len(video_samples)
    for i, (path, label) in enumerate(video_samples):
        print(f"  Extracting features [{i+1}/{total}]: {Path(path).name}", end="\r")
        seq = extract_frame_features_for_lstm(path, cnn_extractor, num_frames)
        if seq is not None:
            X.append(seq[0])    # (num_frames, feature_dim)
            y.append(float(label))
        else:
            print(f"\n  [WARN] No faces found, skipping: {path}")
    print()
    if not X:
        return np.empty((0,)), np.empty((0,))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_lstm_model(num_frames, feature_dim):
    """Build the BiLSTM video model matching create_model.py architecture."""
    from tensorflow.keras.layers import (
        Input, LayerNormalization, Bidirectional, LSTM,
        Dropout, GlobalMaxPooling1D, Dense,
    )
    from tensorflow.keras.models import Model

    inp = Input(shape=(num_frames, feature_dim), name="sequence_input")
    x   = LayerNormalization(name="layer_norm")(inp)
    x   = Bidirectional(
              LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
              name="bilstm_1")(x)
    x   = Dropout(0.4)(x)
    x   = Bidirectional(
              LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
              name="bilstm_2")(x)
    x   = Dropout(0.4)(x)
    x   = GlobalMaxPooling1D(name="temporal_pool")(x)
    x   = Dense(128, activation="relu", name="dense_128")(x)
    x   = Dropout(0.4)(x)
    out = Dense(1, activation="sigmoid", name="lstm_output")(x)

    model = Model(inputs=inp, outputs=out, name="CNN_LSTM_Video_Classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True,
                        help="Root with real/ fake/ subdirs of frame images")
    parser.add_argument("--video_root",   default=None,
                        help="Root with real/ fake/ subdirs of video files "
                             "(defaults to dataset_root if not set)")
    parser.add_argument("--epochs",       type=int, default=30)
    parser.add_argument("--epochs_lstm",  type=int, default=20,
                        help="Epochs for LSTM stage (default 20)")
    parser.add_argument("--batch_size",   type=int, default=16)
    parser.add_argument("--num_frames",   type=int, default=CFG.num_frames)
    parser.add_argument("--skip_cnn",     action="store_true",
                        help="Skip Stage 1 CNN training")
    args = parser.parse_args()

    video_root = args.video_root or args.dataset_root
    Path("weights").mkdir(exist_ok=True)
    lstm_path      = "weights/lstm_model.h5"
    extractor_path = "weights/cnn_extractor.h5"

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 1: CNN image model fine-tuning (original logic preserved exactly)
    # ══════════════════════════════════════════════════════════════════════════
    if not args.skip_cnn:
        print(f"\n{'='*60}")
        print("STAGE 1: CNN Image Model Training")
        print(f"{'='*60}")

        train_s, val_s, _ = build_splits(args.dataset_root)
        print(f"Train: {len(train_s)}  Val: {len(val_s)}")

        train_gen = FaceDataset(train_s, batch_size=args.batch_size, augment=True)
        val_gen   = FaceDataset(val_s,   batch_size=args.batch_size, augment=False)

        model = build_model()
        model.summary()

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

        print(f"\n[Stage 1] CNN model saved → {CFG.model_path}")

        # Derive and save feature extractor from the trained CNN
        from tensorflow.keras.models import Model as KModel
        try:
            gap_out = model.get_layer("gap").output
        except ValueError:
            gap_out = model.layers[-3].output
        extractor = KModel(inputs=model.input, outputs=gap_out,
                           name="CNN_Feature_Extractor")
        extractor.save(extractor_path)
        print(f"[Stage 1] CNN extractor saved → {extractor_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2: BiLSTM video model training
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("STAGE 2: CNN + BiLSTM Video Model Training")
    print(f"{'='*60}")

    import os
    from tensorflow.keras.models import load_model as keras_load
    from tensorflow.keras.models import Model as KModel

    if os.path.exists(extractor_path):
        cnn_extractor = keras_load(extractor_path, compile=False)
        print(f"Loaded extractor from {extractor_path}")
    else:
        cnn_model = keras_load(CFG.model_path, compile=False)
        try:
            gap_out = cnn_model.get_layer("gap").output
        except ValueError:
            gap_out = cnn_model.layers[-3].output
        cnn_extractor = KModel(inputs=cnn_model.input, outputs=gap_out,
                               name="CNN_Feature_Extractor")
        cnn_extractor.save(extractor_path)
        print(f"Derived and saved extractor → {extractor_path}")

    # Collect videos
    video_samples = collect_video_paths(video_root)
    if not video_samples:
        print("[Stage 2] No videos found — skipping LSTM training.")
        print(f"\nTraining complete. CNN model: {CFG.model_path}")
        return

    print(f"Found {len(video_samples)} videos. Pre-extracting CNN features…")
    X, y = preextract_video_features(video_samples, cnn_extractor, args.num_frames)

    if len(X) == 0:
        print("[Stage 2] No valid video features extracted — skipping.")
        return

    print(f"Feature tensor: {X.shape}  Labels: {y.shape}")

    # Train/val split
    from sklearn.model_selection import train_test_split
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.15, stratify=y.astype(int), random_state=42
    )
    print(f"LSTM Train: {len(X_tr)}  Val: {len(X_va)}")

    feature_dim = X.shape[2]
    lstm_model  = build_lstm_model(args.num_frames, feature_dim)
    lstm_model.summary()

    lstm_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            lstm_path, monitor="val_auc", mode="max",
            save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=7,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1,
        ),
    ]

    lstm_model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=args.epochs_lstm,
        batch_size=args.batch_size,
        callbacks=lstm_callbacks,
        verbose=1,
    )

    print(f"\n[Stage 2] BiLSTM model saved → {lstm_path}")
    print("\nTraining complete ✓")
    print(f"  CNN image model   : {CFG.model_path}")
    print(f"  CNN extractor     : {extractor_path}")
    print(f"  BiLSTM video model: {lstm_path}")
    print("\nLaunch the app:  streamlit run app.py")


if __name__ == "__main__":
    main()
