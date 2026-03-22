"""
model.py — Model Architecture Definitions

Models:
  1. build_model()                 → EfficientNetB4 + head  (matches original project)
  2. load_model()                  → loads saved .h5 from CFG.model_path
  3. build_cnn_feature_extractor() → EfficientNetB4 GAP output (for LSTM pipeline)
  4. build_cnn_lstm_video_model()  → BiLSTM head for video sequences
"""

import os

FEATURE_DIM  = 1792   # EfficientNetB4 GAP output size
SEQ_LEN      = 20
LSTM_UNITS   = 256
DROPOUT_RATE = 0.4


def build_model():
    """EfficientNetB4 backbone + Dense head. Used by train.py Stage 1."""
    import tensorflow as tf
    from tensorflow.keras.applications import EfficientNetB4
    from tensorflow.keras import layers, Model
    from config import CFG

    backbone = EfficientNetB4(
        include_top=False, weights="imagenet",
        input_shape=(CFG.face_size, CFG.face_size, 3), pooling=None,
    )
    backbone.trainable = False

    inp = tf.keras.Input(shape=(CFG.face_size, CFG.face_size, 3))
    x   = backbone(inp, training=False)
    x   = layers.GlobalAveragePooling2D(name="gap")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(CFG.dropout_rate)(x)
    x   = layers.Dense(CFG.dense_units, activation="relu", name="dense")(x)
    x   = layers.Dropout(CFG.dropout_rate)(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=inp, outputs=out, name="EfficientNetB4_Classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def load_model():
    """Load the trained CNN model from CFG.model_path. Called by inference.py."""
    from tensorflow.keras.models import load_model as keras_load_model
    from config import CFG

    if not os.path.exists(CFG.model_path):
        raise FileNotFoundError(
            f"Model not found at '{CFG.model_path}'.\n"
            "Run create_model.py to create a dummy model, or train.py to train one."
        )
    return keras_load_model(CFG.model_path, compile=False)


def build_cnn_feature_extractor(weights_path=None):
    """EfficientNetB4 → 1792-d feature vector per frame. Used by video pipeline."""
    import tensorflow as tf
    from tensorflow.keras.applications import EfficientNetB4
    from tensorflow.keras import layers, Model
    from config import CFG

    backbone = EfficientNetB4(
        include_top=False, weights="imagenet",
        input_shape=(CFG.face_size, CFG.face_size, 3), pooling=None,
    )
    backbone.trainable = False

    inp  = tf.keras.Input(shape=(CFG.face_size, CFG.face_size, 3), name="frame_input")
    x    = backbone(inp, training=False)
    feat = layers.GlobalAveragePooling2D(name="frame_features")(x)

    extractor = Model(inputs=inp, outputs=feat, name="CNN_Feature_Extractor")

    if weights_path and os.path.isfile(weights_path):
        extractor.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print(f"[Extractor] Loaded backbone weights from {weights_path}")

    return extractor


def build_cnn_lstm_video_model(
    seq_len=SEQ_LEN, feature_dim=FEATURE_DIM,
    lstm_units=LSTM_UNITS, weights_path=None,
):
    """BiLSTM head for video sequences. Input: (batch, seq_len, feature_dim)."""
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    inp = tf.keras.Input(shape=(seq_len, feature_dim), name="sequence_input")
    x   = layers.LayerNormalization(name="layer_norm")(inp)
    x   = layers.Bidirectional(
              layers.LSTM(lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
              name="bilstm_1")(x)
    x   = layers.Dropout(DROPOUT_RATE)(x)
    x   = layers.Bidirectional(
              layers.LSTM(lstm_units // 2, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
              name="bilstm_2")(x)
    x   = layers.Dropout(DROPOUT_RATE)(x)
    x   = layers.GlobalMaxPooling1D(name="temporal_pool")(x)
    x   = layers.Dense(128, activation="relu", name="dense_128")(x)
    x   = layers.Dropout(DROPOUT_RATE)(x)
    out = layers.Dense(1, activation="sigmoid", name="lstm_output")(x)

    model = Model(inputs=inp, outputs=out, name="CNN_LSTM_Video_Classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    if weights_path and os.path.isfile(weights_path):
        model.load_weights(weights_path)
        print(f"[LSTM] Loaded weights from {weights_path}")
    else:
        print("[LSTM] No checkpoint — using random weights (demo mode).")

    return model
