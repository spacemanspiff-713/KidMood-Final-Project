"""Data processing utilities for the folder-based FER-2013 dataset.

Usage:
    python src/data_processing.py --raw-dir data/raw --output data/processed

Expected dataset layout:
    data/raw/train/{angry,fear,happy,sad}
    data/raw/test/{angry,fear,happy,sad}
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


CLASS_DIRS = ["angry", "fear", "happy", "sad"]
CLASS_NAMES = ["Angry", "Fear", "Happy", "Sad"]
INDEX_TO_EMOTION = dict(enumerate(CLASS_NAMES))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def validate_directory_structure(raw_dir: str | Path) -> Path:
    """Confirm that required train/test class folders exist."""

    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_dir}")

    missing: list[Path] = []
    for split in ["train", "test"]:
        for class_dir in CLASS_DIRS:
            expected = raw_dir / split / class_dir
            if not expected.exists():
                missing.append(expected)

    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing required dataset folders:\n{missing_text}")

    return raw_dir


def count_images_by_class(split_dir: Path) -> dict[str, int]:
    """Count image files in each target class directory."""

    counts: dict[str, int] = {}
    for class_dir in CLASS_DIRS:
        folder = split_dir / class_dir
        counts[class_dir] = sum(
            1
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return counts


def write_class_map(output_dir: str | Path) -> None:
    """Save the class index mapping used by Keras."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_map = "\n".join(f"{idx},{name}" for idx, name in INDEX_TO_EMOTION.items())
    (output_dir / "class_map.csv").write_text(f"index,label\n{class_map}\n", encoding="utf-8")


def write_dataset_summary(raw_dir: str | Path, output_dir: str | Path) -> None:
    """Save a small text summary of the folder-based dataset."""

    raw_dir = validate_directory_structure(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_counts = count_images_by_class(raw_dir / "train")
    test_counts = count_images_by_class(raw_dir / "test")

    lines = [
        "KidMood Folder-Based FER-2013 Dataset Summary",
        "==============================================",
        "",
        f"Raw dataset directory: {raw_dir}",
        "",
        "Class mapping:",
    ]
    lines.extend(f"  {idx}: {name}" for idx, name in INDEX_TO_EMOTION.items())
    lines.extend(["", "Train image counts:"])
    lines.extend(f"  {CLASS_NAMES[i]}: {train_counts[CLASS_DIRS[i]]}" for i in range(len(CLASS_DIRS)))
    lines.extend(["", "Test image counts:"])
    lines.extend(f"  {CLASS_NAMES[i]}: {test_counts[CLASS_DIRS[i]]}" for i in range(len(CLASS_DIRS)))
    lines.extend(
        [
            "",
            f"Total train images: {sum(train_counts.values())}",
            f"Total test images: {sum(test_counts.values())}",
            "",
            "Images are loaded with tf.keras.utils.image_dataset_from_directory.",
            "The training split is further divided into train/validation during model training.",
        ]
    )

    (output_dir / "dataset_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_class_map(output_dir)


def build_image_datasets(
    raw_dir: str | Path = "data/raw",
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    validation_split: float = 0.15,
    seed: int = 42,
) -> tuple[Any, Any, Any]:
    """Create train, validation, and test datasets from the FER-2013 folders."""

    import tensorflow as tf

    raw_dir = validate_directory_structure(raw_dir)
    train_dir = raw_dir / "train"
    test_dir = raw_dir / "test"

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_DIRS,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=seed,
        validation_split=validation_split,
        subset="training",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_DIRS,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=seed,
        validation_split=validation_split,
        subset="validation",
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_DIRS,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
    )

    return train_ds, val_ds, test_ds


def optimize_dataset(dataset: Any) -> Any:
    """Prefetch a dataset for smoother training/evaluation."""

    import tensorflow as tf

    return dataset.prefetch(tf.data.AUTOTUNE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output", default="data/processed")
    args = parser.parse_args()

    write_dataset_summary(args.raw_dir, args.output)

    print("Processed dataset metadata saved to:", args.output)
    print("Class mapping:")
    for idx, name in INDEX_TO_EMOTION.items():
        print(f"  {idx}: {name}")


if __name__ == "__main__":
    main()
