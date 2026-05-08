# Dataset Documentation

## Dataset Used

This project uses the Kaggle FER-2013 image-folder dataset:

```txt
https://www.kaggle.com/datasets/msambare/fer2013
```

## Expected Files

Place or extract the dataset under `data/raw` with this structure:

```txt
data/raw/
├── train/
│   ├── angry/
│   ├── fear/
│   ├── happy/
│   └── sad/
└── test/
    ├── angry/
    ├── fear/
    ├── happy/
    └── sad/
```

The Kaggle dataset may also include `disgust`, `neutral`, and `surprise`. Those folders can remain in place; the project scripts explicitly load only `angry`, `fear`, `happy`, and `sad`.

## Labels Used in This Project

| Index | Emotion | Folder |
|---|---|---|
| 0 | Angry | `angry` |
| 1 | Fear | `fear` |
| 2 | Happy | `happy` |
| 3 | Sad | `sad` |

## Preprocessing Workflow

The project uses `tf.keras.utils.image_dataset_from_directory` to:

1. load images from train/test folders
2. infer labels from class folder names
3. resize images to 224x224
4. convert images to RGB
5. create a validation split from the training folder

Run:

```bash
python src/data_processing.py --raw-dir data/raw --output data/processed
```

This writes lightweight metadata files such as `class_map.csv` and `dataset_summary.txt`.
