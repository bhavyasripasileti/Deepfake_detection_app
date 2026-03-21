import tensorflow as tf
from tensorflow.keras import layers, Model, Input, regularizers
from tensorflow.keras.applications import EfficientNetB4
from config import CFG

def build_model():
    """EfficientNetB4 + Dense head for binary deepfake classification."""
    backbone = EfficientNetB4(
        include_top=False,
        weights="imagenet",
        pooling="avg",
        input_shape=(CFG.face_size, CFG.face_size, 3),
    )
    backbone.trainable = False

    inp = Input(shape=(CFG.face_size, CFG.face_size, 3))
    x = backbone(inp, training=False)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(CFG.dense_units, activation="swish",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(CFG.dropout_rate)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inp, out, name="deepfake_efficientnetb4")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model

def load_model():
    """Load saved model weights for inference."""
    return tf.keras.models.load_model(CFG.model_path, compile=False)