"""Model definitions for KidMood."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models


def build_mobilenet_model(num_classes: int = 4, input_shape: tuple[int, int, int] = (224, 224, 3)) -> tf.keras.Model:
    """Build MobileNetV2 transfer learning model."""

    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)
    # image_dataset_from_directory returns RGB float tensors in the 0-255 range.
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="KidMood_MobileNetV2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_baseline_cnn(num_classes: int = 4, input_shape: tuple[int, int, int] = (224, 224, 3)) -> tf.keras.Model:
    """Build a simple baseline CNN for comparison."""

    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Rescaling(1.0 / 255.0),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation="relu"),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="KidMood_BaselineCNN",
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
