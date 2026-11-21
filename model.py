"""Hybrid Deep Learning Architecture for Fruit Quality Grading.

Combines:
  1. Deep Feature Extractor: MobileNetV2 (ImageNet Transfer Learning)
     - Global Average Pooling produces a 1280-D semantic embedding.
  2. Handcrafted Feature Branch:
     - 69-D OpenCV physical descriptors (color, texture, defects, shape).
     - Dense layer (64 units, ReLU) projects features into a latent embedding.
  3. Feature Fusion Layer:
     - Concatenates the 1280-D deep embedding and 64-D physical feature embedding.
  4. Classification Head:
     - Dense(128, ReLU) -> Dropout(0.3) -> Dense(num_classes, Softmax).
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from config import IMG_SIZE


def build_model(num_classes, feature_dim, use_features=True, weights="imagenet"):
    """Build the hybrid neural network architecture.

    Args:
        num_classes (int): Number of target quality grades (e.g. 3).
        feature_dim (int): Dimension of OpenCV features (e.g. 69).
        use_features (bool): Whether to include the handcrafted feature branch.
        weights (str): Weights for MobileNetV2 ("imagenet" or None).

    Returns:
        tf.keras.Model: Compiled or uncompiled Keras Model instance.
    """
    # 1. Image Input Branch
    img_in = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")

    # Data Augmentation (active during training only)
    x = layers.RandomFlip("horizontal_and_vertical")(img_in)
    x = layers.RandomRotation(0.12)(x)
    x = layers.RandomZoom(0.08)(x)
    x = layers.RandomContrast(0.08)(x)
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(x)  # Scale [0, 255] to [-1, 1]

    # MobileNetV2 Backbone
    try:
        base = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights=weights,
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            pooling="avg",
            name="backbone"
        )
    except Exception as e:
        print(f"[Warning] Could not load ImageNet weights ({e}); training from scratch.")
        base = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights=None,
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            pooling="avg",
            name="backbone"
        )

    base.trainable = False  # Freeze backbone in Phase 1
    x = base(x, training=False)
    x = layers.Dropout(0.30, name="backbone_dropout")(x)

    inputs = [img_in]

    # 2. Handcrafted OpenCV Feature Branch (Feature Fusion)
    if use_features:
        feat_in = layers.Input(shape=(feature_dim,), name="features")
        f = layers.Dense(64, activation="relu", name="feat_dense1")(feat_in)
        f = layers.Dropout(0.20, name="feat_dropout")(f)
        x = layers.Concatenate(name="feature_fusion")([x, f])
        inputs.append(feat_in)

    # 3. Dense Classifier Head
    x = layers.Dense(128, activation="relu", name="classifier_dense")(x)
    x = layers.Dropout(0.30, name="classifier_dropout")(x)
    out = layers.Dense(num_classes, activation="softmax", name="grade")(x)

    model = models.Model(inputs=inputs, outputs=out, name="fruit_quality_grader")
    return model


def unfreeze_top(model, n_layers=30):
    """Unfreeze top n layers of the MobileNetV2 backbone for fine-tuning.

    Keeps BatchNormalization layers frozen to maintain stable batch statistics.
    """
    base = model.get_layer("backbone")
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
