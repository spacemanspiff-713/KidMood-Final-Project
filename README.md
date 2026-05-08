# KidMood: Classroom Emotion Recognition

![Demo Prediction](results/images/happy_demo_prediction.jpg)

A computer vision system that classifies facial expressions into **Happy**, **Sad**, **Angry**, or **Fear** using the Kaggle FER-2013 image-folder dataset and transfer learning.

## Team Members

- **Jason Trimble** - Solo developer, dataset preparation, model training, evaluation, demo, documentation

## Project Tier

**Tier 2** - This project uses transfer learning with MobileNetV2 instead of training a CNN completely from scratch. It includes a complete training pipeline, evaluation workflow, inference script, demo notebook, and final documentation.

---

## Problem & Solution

### The Problem

Teachers and childcare workers cannot continuously monitor every student's emotional state during a busy class or group activity. Early signs of sadness, fear, or anger can be missed, which may delay adult support or intervention.

### Our Solution

KidMood uses computer vision to classify a facial image into one of four emotions: Happy, Sad, Angry, or Fear. The system is designed as an assistive signal for awareness, not a medical or psychological diagnosis tool.

### Impact

Teachers, parents, counselors, and childcare workers could benefit from a quick visual summary of emotional signals. The value is not replacing human judgment, but helping adults notice possible emotional changes faster.

---

## Technical Details

### Approach

- **Task:** Image Classification / Facial Expression Recognition
- **Model:** MobileNetV2 transfer learning
- **Backup Model:** Simple CNN baseline
- **Framework:** TensorFlow / Keras
- **Key Libraries:** tensorflow, numpy, matplotlib, scikit-learn, opencv-python, pillow

### System Architecture

```txt
[Input Face Image]
        ↓
[Preprocessing: folder loading, resize, RGB conversion]
        ↓
[MobileNetV2 Transfer Learning Model]
        ↓
[Softmax Emotion Classifier]
        ↓
[Output: emotion label + confidence score]
```

---

## Dataset

- **Source:** Kaggle FER-2013 image-folder dataset
- **Expected Local Folders:**

```txt
data/raw/train/{angry,fear,happy,sad}
data/raw/test/{angry,fear,happy,sad}
```

- **Image Format:** FER-2013 face images stored as image files
- **Original Kaggle Classes:** angry, disgust, fear, happy, neutral, sad, surprise
- **Project Classes:** Angry, Fear, Happy, Sad
- **Preprocessing:** Keras directory loading, target-class filtering, resizing to 224x224, RGB conversion
- **Split:** Train 17,117 images, validation 3,020 images, test 5,003 images

Only the four target class folders are used. Extra Kaggle folders can remain in `data/raw`; the scripts pass an explicit class list to Keras.

---

## How to Run

### Installation

```bash
git clone spacemanspiff-713/KidMood-Final-Project
cd KidMood-Final-Project

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

### Quick Start

Option 1: Run the demo notebook:

```bash
jupyter notebook notebooks/04_demo.ipynb
```

Option 2: Run inference on one image:

```bash
python src/inference.py --image data/sample/happy_demo.jpg --model models/trained/kidmood_mobilenetv2.keras
```

### Detailed Usage

Step 1: Validate data and write dataset metadata:

```bash
python src/data_processing.py --raw-dir data/raw --output data/processed
```

Step 2: Train model:

```bash
python src/train.py --raw-dir data/raw --epochs 5 --model-out models/trained/kidmood_mobilenetv2.keras
```

Step 3: Evaluate model:

```bash
python src/evaluate.py --raw-dir data/raw --model models/trained/kidmood_mobilenetv2.keras
```

Step 4: Run inference:

```bash
python src/inference.py --image data/sample/happy_demo.jpg --model models/trained/kidmood_mobilenetv2.keras --output results/images
```

---

## Results

Final metrics are saved to:

```txt
results/metrics.txt
results/visualizations/confusion_matrix.png
```

### Performance Metrics

| Metric | Value |
|---|---|
| Accuracy | 0.5515 |
| Macro F1-score | 0.4887 |
| Weighted F1-score | 0.53 |
| Single-image inference time | About 0.78s on CPU in Codex sandbox |

Class-level results:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Angry | 0.46 | 0.37 | 0.41 | 958 |
| Fear | 0.60 | 0.23 | 0.33 | 1024 |
| Happy | 0.63 | 0.85 | 0.73 | 1774 |
| Sad | 0.46 | 0.53 | 0.49 | 1247 |

### Visualizations

- Confusion matrix: `results/visualizations/confusion_matrix.png`
- Prediction outputs: `results/images/`
- Sample prediction log: `results/sample_predictions.md`

### Success / Failure Cases

Successful examples:
- `data/sample/happy_demo.jpg` predicted **Happy** with 89.2% confidence.
- `data/sample/fear_demo.jpg` predicted **Fear** with 33.7% confidence.
- `data/sample/sad_demo.jpg` predicted **Sad** with 62.3% confidence.

Difficult or failed examples:
- `data/sample/angry_demo.jpg` was predicted as **Sad** with 40.1% confidence.
- `data/sample/sad-boy.jpg` was predicted as **Angry** with 34.5% confidence.

These mistakes are reasonable for FER-2013-style emotion recognition because angry, sad, and fearful expressions can share similar facial cues, and many source images are low resolution.

---

## Demo Video

Demo video will be recorded as a 3-5 minute screen recording.

Planned video flow:
1. Introduce KidMood
2. Show dataset/preprocessing
3. Run inference on 3-5 test images
4. Show metrics and confusion matrix
5. Explain success and failure cases

Video link placeholder:

```txt
Add YouTube or Google Drive link here after recording.
```

---

## Key Learnings

### What Worked Well

- Transfer learning made the model plan realistic with limited time and compute.
- FER-2013 is small enough to train quickly.
- Confusion matrix analysis helps explain model mistakes.

### Challenges Faced

- Emotion classes like Fear and Sad can look visually similar.
- FER-2013 images are only 48x48, so detail is limited.
- Facial expression labels may include noise or subjective interpretation.

### What I Would Do Differently

- Use a larger, cleaner facial expression dataset.
- Add face detection before classification.
- Compare multiple models like MobileNetV2 and ResNet50.

---

## AI Usage Documentation

See detailed log: `docs/AI_usage_log.md`

Summary:
- Used AI for project planning, repo scaffolding, code organization, and documentation drafting.
- All code should be reviewed, tested, and understood before final submission.

---

## Future Improvements

1. Add webcam-based real-time emotion detection.
2. Add face detection and cropping before classification.
3. Compare MobileNetV2 against ResNet50 and a custom CNN.
4. Add a simple dashboard showing emotion trends over time.
5. Improve fairness and bias testing across lighting, age, and facial variation.

---

## References

1. FER-2013 facial expression dataset
2. TensorFlow / Keras MobileNetV2 documentation
3. Scikit-learn classification metrics documentation
4. OpenCV image processing documentation

## License

Academic Use Only

## Acknowledgments

Thanks to Professor Hardik Gohel for an amazing course experience.  
Pretrained models from TensorFlow / Keras applications.  
AI assistance from ChatGPT for planning and scaffolding.
