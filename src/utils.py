"""Utility functions for KidMood."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


CLASS_NAMES = ["Angry", "Fear", "Happy", "Sad"]


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def grayscale_to_mobilenet(images: np.ndarray) -> np.ndarray:
    """Resize grayscale image arrays to 224x224 RGB for MobileNetV2."""

    images = tf.convert_to_tensor(images, dtype=tf.float32)
    images = tf.image.resize(images, (224, 224))

    if images.shape[-1] == 1:
        images = tf.image.grayscale_to_rgb(images)

    return images.numpy()


def load_single_image_for_inference(image_path: str | Path) -> np.ndarray:
    """Load image from disk using the same RGB 224x224 convention as training."""

    img = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    return arr[np.newaxis, ...]


def timed_prediction(model, batch: np.ndarray) -> tuple[np.ndarray, float]:
    """Run prediction and return probabilities + elapsed seconds."""

    start = time.perf_counter()
    probs = model.predict(batch)
    elapsed = time.perf_counter() - start
    return probs, elapsed
