"""Run inference on a single image.

Usage:
    python src/inference.py --image data/sample/happy_demo.jpg --model models/trained/kidmood_mobilenetv2.keras
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

from utils import CLASS_NAMES, load_single_image_for_inference, timed_prediction


def annotate_image(image_path: Path, label: str, confidence: float, output_path: Path) -> None:
    """Save image with prediction text overlay."""

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    text = f"{label} ({confidence:.1%})"

    draw.rectangle((0, 0, img.width, 32), fill=(0, 0, 0))
    draw.text((8, 8), text, fill=(255, 255, 255))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="models/trained/kidmood_mobilenetv2.keras")
    parser.add_argument("--output", default="results/images")
    args = parser.parse_args()

    image_path = Path(args.image)
    model_path = Path(args.model)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Train the model first with src/train.py."
        )

    model = tf.keras.models.load_model(model_path)
    batch = load_single_image_for_inference(image_path)

    probs, elapsed = timed_prediction(model, batch)
    pred_idx = int(probs[0].argmax())
    confidence = float(probs[0][pred_idx])
    label = CLASS_NAMES[pred_idx]

    output_dir = Path(args.output)
    output_path = output_dir / f"{image_path.stem}_prediction.jpg"
    annotate_image(image_path, label, confidence, output_path)

    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Inference Time: {elapsed:.4f} seconds")
    print(f"Output image saved to: {output_path}")


if __name__ == "__main__":
    main()
