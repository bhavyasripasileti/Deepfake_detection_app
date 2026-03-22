"""
create_model.py — Creates and saves dummy model weights for both pipelines.

Run this once to initialise the weights/ folder before launching the app
if you do not yet have trained checkpoints.

Usage:
    python create_model.py
"""

import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization,
    Bidirectional, LSTM, LayerNormalization, GlobalMaxPooling1D, Input,
)
from tensorflow.keras.models import Model

from config import CFG

os.makedirs("weights", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CNN Image Model  (EfficientNetB4 + classification head)
#    Saved to CFG.model_path  (used by image prediction & as feature extractor)
# ─────────────────────────────────────────────────────────────────────────────
print("Building CNN image model (EfficientNetB4)…")

base = EfficientNetB4(
    weights=None,
    include_top=False,
    input_shape=(CFG.face_size, CFG.face_size, 3),
)
base.trainable = False

x      = GlobalAveragePooling2D(name="gap")(base.output)
x      = BatchNormalization()(x)
x      = Dropout(0.4)(x)
x      = Dense(256, activation="relu", name="dense_256")(x)
x      = Dropout(0.4)(x)
output = Dense(1, activation="sigmoid", name="cnn_output")(x)

cnn_model = Model(inputs=base.input, outputs=output, name="CNN_Image_Classifier")
cnn_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
)

cnn_model.save(CFG.model_path)
print(f"✅ CNN model saved → {CFG.model_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. CNN Feature Extractor  (same backbone, outputs 1280-d feature vector)
#    Saved to weights/cnn_extractor.h5
#    Used by the video pipeline to encode each frame before the LSTM.
# ─────────────────────────────────────────────────────────────────────────────
print("Building CNN feature extractor…")

feat_out   = GlobalAveragePooling2D(name="frame_features")(base.output)
extractor  = Model(inputs=base.input, outputs=feat_out, name="CNN_Feature_Extractor")

extractor_path = os.path.join("weights", "cnn_extractor.h5")
extractor.save(extractor_path)
print(f"✅ CNN feature extractor saved → {extractor_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. BiLSTM Video Model  (accepts pre-extracted CNN feature sequences)
#    Input:  (batch, num_frames, feature_dim)
#    Output: (batch, 1)
#    Saved to weights/lstm_model.h5
# ─────────────────────────────────────────────────────────────────────────────
print("Building BiLSTM video model…")

# Infer feature_dim from the extractor output shape
feature_dim = extractor.output_shape[-1]   # e.g. 1792 for EfficientNetB4

seq_input = Input(shape=(CFG.num_frames, feature_dim), name="sequence_input")

x = LayerNormalization(name="layer_norm")(seq_input)

x = Bidirectional(
    LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
    name="bilstm_1",
)(x)
x = Dropout(0.4)(x)

x = Bidirectional(
    LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
    name="bilstm_2",
)(x)
x = Dropout(0.4)(x)

x      = GlobalMaxPooling1D(name="temporal_pool")(x)
x      = Dense(128, activation="relu", name="dense_128")(x)
x      = Dropout(0.4)(x)
lstm_out = Dense(1, activation="sigmoid", name="lstm_output")(x)

lstm_model = Model(inputs=seq_input, outputs=lstm_out, name="CNN_LSTM_Video_Classifier")
lstm_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
)

lstm_path = os.path.join("weights", "lstm_model.h5")
lstm_model.save(lstm_path)
print(f"✅ BiLSTM video model saved → {lstm_path}")

# ─────────────────────────────────────────────────────────────────────────────
print("\nAll dummy models created successfully!")
print(f"  CNN image model   : {CFG.model_path}")
print(f"  CNN extractor     : {extractor_path}")
print(f"  BiLSTM video model: {lstm_path}")
print("\nNext steps:")
print("  • Train on real data:  python train.py --dataset_root dataset/")
print("  • Launch the app:      streamlit run app.py")
