"""
model.py — Model Architecture Definitions

Two models are defined:
  1. build_cnn_image_model()       → EfficientNetB0 + head for single-image classification
  2. build_cnn_lstm_video_model()  → EfficientNetB0 (frozen feature extractor) + BiLSTM for video

The CNN encoder weights are shared conceptually; in practice both load from the same
checkpoint when available (see weights/ directory).
"""

import numpy as np

# ─── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE      = 224          # EfficientNet-B0 canonical input
FEATURE_DIM   = 1280         # EfficientNet-B0 top-layer feature size (after GAP)
SEQ_LEN       = 20           # default frames per video clip
LSTM_UNITS    = 256
DROPOUT_RATE  = 0.4


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CNN Image Model
# ─────────────────────────────────────────────────────────────────────────────
def build_cnn_image_model(weights_path: str | None = None):
    """
    EfficientNetB0 backbone → GlobalAveragePooling → Dense head.
    Output: sigmoid scalar (probability of being FAKE).

    Args:
        weights_path: path to a .h5 / .weights.h5 checkpoint.
                      If None, ImageNet weights are used (fine-tuning ready).
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import EfficientNetB0

    # Backbone (frozen by default; unfreeze top layers for fine-tuning)
    backbone = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling=None,
    )
    backbone.trainable = False   # freeze all; caller can unfreeze selectively

    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image_input")
    x   = backbone(inp, training=False)
    x   = layers.GlobalAveragePooling2D(name="gap")(x)          # (batch, 1280)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(DROPOUT_RATE)(x)
    x   = layers.Dense(512, activation="relu", name="dense_512")(x)
    x   = layers.Dropout(DROPOUT_RATE)(x)
    out = layers.Dense(1, activation="sigmoid", name="cnn_output")(x)

    model = Model(inputs=inp, outputs=out, name="CNN_Image_Classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    if weights_path and _file_exists(weights_path):
        model.load_weights(weights_path)
        print(f"[CNN] Loaded weights from {weights_path}")
    else:
        print("[CNN] Using ImageNet initialised weights (no checkpoint found).")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CNN Feature Extractor  (shared encoder, output = feature vector)
# ─────────────────────────────────────────────────────────────────────────────
def build_cnn_feature_extractor(weights_path: str | None = None):
    """
    Same EfficientNetB0 backbone but output is the 1280-d feature vector,
    not a binary prediction.  Used by the video pipeline to encode each frame.

    Args:
        weights_path: optional path to fine-tuned CNN weights (.h5).
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import EfficientNetB0

    backbone = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling=None,
    )
    backbone.trainable = False

    inp  = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="frame_input")
    x    = backbone(inp, training=False)
    feat = layers.GlobalAveragePooling2D(name="frame_features")(x)   # (batch, 1280)

    extractor = Model(inputs=inp, outputs=feat, name="CNN_Feature_Extractor")

    if weights_path and _file_exists(weights_path):
        # Load only layers that match (backbone layers)
        extractor.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print(f"[Extractor] Loaded matching backbone weights from {weights_path}")

    return extractor


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CNN + BiLSTM Video Model
# ─────────────────────────────────────────────────────────────────────────────
def build_cnn_lstm_video_model(
    seq_len: int = SEQ_LEN,
    feature_dim: int = FEATURE_DIM,
    lstm_units: int = LSTM_UNITS,
    weights_path: str | None = None,
):
    """
    Sequence model that accepts pre-extracted CNN feature vectors.

    Input shape:  (batch, seq_len, feature_dim)   — frame feature sequences
    Output shape: (batch, 1)                        — fake probability

    Architecture:
        Input → LayerNorm → BiLSTM(256) → Dropout → BiLSTM(128) → Dropout
              → GlobalMaxPool1D → Dense(128) → Dropout → Dense(1, sigmoid)

    The CNN feature extractor runs *outside* this model (in the video pipeline)
    so that:
      • The same frozen CNN can be reused across both image & video paths.
      • Only the LSTM weights need updating during video fine-tuning.

    Args:
        seq_len:      number of frames in the sequence (None = variable length).
        feature_dim:  CNN output dimensionality (1280 for EfficientNet-B0).
        lstm_units:   base LSTM hidden size.
        weights_path: optional .h5 checkpoint path.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    inp = tf.keras.Input(
        shape=(seq_len, feature_dim),
        name="sequence_input",
    )

    # Normalise feature scale across frames
    x = layers.LayerNormalization(name="layer_norm")(inp)

    # First BiLSTM — return sequences for stacking
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
        name="bilstm_1",
    )(x)
    x = layers.Dropout(DROPOUT_RATE)(x)

    # Second BiLSTM — return sequences for pooling
    x = layers.Bidirectional(
        layers.LSTM(lstm_units // 2, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
        name="bilstm_2",
    )(x)
    x = layers.Dropout(DROPOUT_RATE)(x)

    # Aggregate across time
    x = layers.GlobalMaxPooling1D(name="temporal_pool")(x)

    x   = layers.Dense(128, activation="relu", name="dense_128")(x)
    x   = layers.Dropout(DROPOUT_RATE)(x)
    out = layers.Dense(1, activation="sigmoid", name="lstm_output")(x)

    model = Model(inputs=inp, outputs=out, name="CNN_LSTM_Video_Classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    if weights_path and _file_exists(weights_path):
        model.load_weights(weights_path)
        print(f"[LSTM] Loaded weights from {weights_path}")
    else:
        print("[LSTM] No checkpoint found — using random weights (demo/inference mode).")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# 4.  End-to-End trainable model  (optional, for training from scratch)
# ─────────────────────────────────────────────────────────────────────────────
def build_end_to_end_video_model(
    seq_len: int = SEQ_LEN,
    lstm_units: int = LSTM_UNITS,
):
    """
    Full end-to-end model using TimeDistributed(EfficientNetB0) + BiLSTM.
    Suitable for fine-tuning on GPU with tf.data pipelines.

    Input shape: (batch, seq_len, 224, 224, 3)
    Output:      (batch, 1)

    NOTE: This model is memory-intensive. For inference use the two-stage
          approach (build_cnn_feature_extractor + build_cnn_lstm_video_model).
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import EfficientNetB0

    backbone = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling="avg",
    )
    backbone.trainable = False

    frame_inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    feat_out  = backbone(frame_inp, training=False)
    frame_encoder = Model(frame_inp, feat_out, name="frame_encoder")

    vid_inp = tf.keras.Input(shape=(seq_len, IMG_SIZE, IMG_SIZE, 3), name="video_input")
    x = layers.TimeDistributed(frame_encoder, name="time_distributed_cnn")(vid_inp)
    x = layers.LayerNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True, dropout=0.2))(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units // 2, return_sequences=True, dropout=0.2))(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=vid_inp, outputs=out, name="E2E_CNN_LSTM")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────
def _file_exists(path: str) -> bool:
    import os
    return os.path.isfile(path)


def model_summary():
    """Print summaries for both models (quick sanity check)."""
    print("=" * 60)
    print("CNN IMAGE MODEL")
    print("=" * 60)
    m1 = build_cnn_image_model()
    m1.summary()

    print("\n" + "=" * 60)
    print("CNN + BiLSTM VIDEO MODEL  (receives pre-extracted features)")
    print("=" * 60)
    m2 = build_cnn_lstm_video_model()
    m2.summary()


if __name__ == "__main__":
    model_summary()
