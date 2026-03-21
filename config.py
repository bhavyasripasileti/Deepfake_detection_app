from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    # Model
    model_path:      str   = "weights/deepfake_model.h5"
    backbone:        str   = "efficientnetb4"
    face_size:       int   = 224
    dense_units:     int   = 256
    dropout_rate:    float = 0.5

    # Inference
    fake_threshold:  float = 0.5
    fake_label:      int   = 1
    real_label:      int   = 0

    # Video
    num_frames:      int   = 10
    max_upload_mb:   int   = 200

    # Grad-CAM
    gradcam_layer:   str   = "top_conv"
    gradcam_alpha:   float = 0.45

    # UI
    gallery_cols:    int   = 5

CFG = Config()