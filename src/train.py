"""Train KidMood model.

Usage:
    python src/train.py --raw-dir data/raw --epochs 5 --model-out models/trained/kidmood_mobilenetv2.keras
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from data_processing import build_image_datasets, optimize_dataset
from model import build_baseline_cnn, build_mobilenet_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-type", choices=["mobilenet", "baseline"], default="mobilenet")
    parser.add_argument("--model-out", default="models/trained/kidmood_mobilenetv2.keras")
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--validation-steps", type=int, default=None)
    args = parser.parse_args()

    train_ds, val_ds, _ = build_image_datasets(
        raw_dir=args.raw_dir,
        batch_size=args.batch_size,
    )
    train_ds = optimize_dataset(train_ds)
    val_ds = optimize_dataset(val_ds)

    if args.model_type == "mobilenet":
        model = build_mobilenet_model(num_classes=4)
    else:
        model = build_baseline_cnn(num_classes=4)

    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.validation_steps,
        callbacks=callbacks,
    )

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_out)

    print(f"Model saved to {model_out}")


if __name__ == "__main__":
    main()
