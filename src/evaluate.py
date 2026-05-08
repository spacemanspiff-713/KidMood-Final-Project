"""Evaluate KidMood model.

Usage:
    python src/evaluate.py --raw-dir data/raw --model models/trained/kidmood_mobilenetv2.keras
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay

from data_processing import build_image_datasets, optimize_dataset
from utils import CLASS_NAMES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--model", default="models/trained/kidmood_mobilenetv2.keras")
    parser.add_argument("--results", default="results")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Train the model first.")

    _, _, test_ds = build_image_datasets(raw_dir=args.raw_dir, batch_size=args.batch_size)
    if args.max_batches is not None:
        test_ds = test_ds.take(args.max_batches)
    y_test = np.concatenate([labels.numpy() for _, labels in test_ds], axis=0)
    test_ds = optimize_dataset(test_ds)

    model = tf.keras.models.load_model(model_path, compile=False)
    probs = model.predict(test_ds)
    y_pred = np.argmax(probs, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(CLASS_NAMES))))

    results_dir = Path(args.results)
    viz_dir = results_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    metrics_text = (
        f"KidMood Evaluation Metrics\n"
        f"==========================\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Macro F1-score: {macro_f1:.4f}\n\n"
        f"Classification Report:\n{report}\n"
    )

    (results_dir / "metrics.txt").write_text(metrics_text, encoding="utf-8")
    print(metrics_text)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("KidMood Confusion Matrix")
    plt.tight_layout()
    plt.savefig(viz_dir / "confusion_matrix.png", dpi=160)
    print(f"Saved confusion matrix to {viz_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
