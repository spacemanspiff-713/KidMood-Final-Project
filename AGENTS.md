# AGENTS.md — KidMood Final Project

## Project Identity

Project Name: **KidMood: Classroom Emotion Recognition**  
Author: **Jason Trimble**  
Course: **ITAI 1378 — Computer Vision and AI**  
Project Type: **Final Project**  
Tier: **Tier 2 — Transfer Learning**

This repository contains a complete academic computer vision project using the FER-2013 dataset and transfer learning for facial emotion recognition.

The goal is NOT to build a production application.  
The goal IS to build a clean, reproducible, well-documented academic ML workflow.

---

## Important Project Context

This project already contains:

- repository structure
- notebooks
- preprocessing pipeline
- training scripts
- evaluation scripts
- inference scripts
- README
- presentation materials
- AI usage documentation

Do **not** unnecessarily rewrite the project structure.

Your job is to:

1. help run/debug the project
2. improve reliability
3. fill in missing runtime outputs
4. help complete final deliverables

---

## Primary Project Goals

The final project must demonstrate:

- image classification
- transfer learning
- preprocessing workflow
- model training
- model evaluation
- inference/demo pipeline
- reproducibility
- academic documentation

The final deliverable should feel like a complete ML workflow.

---

## Current Project Structure

```txt
KidMood-Final-Project/
├── README.md
├── requirements.txt
├── environment.yml
├── demo_video_link.txt
├── data/
│   ├── raw/
│   ├── processed/
│   ├── sample/
│   └── README.md
├── models/
│   ├── pretrained/
│   ├── trained/
│   └── README.md
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_evaluation.ipynb
│   └── 04_demo.ipynb
├── src/
│   ├── data_processing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── utils.py
├── results/
│   ├── images/
│   ├── metrics.txt
│   └── visualizations/
└── docs/
    ├── proposal.pdf
    ├── presentation.pdf
    ├── presentation.pptx
    ├── demo_video_script.md
    └── AI_usage_log.md
```

Preserve this structure.

---

## Project Scope

The project classifies facial expressions into:

- Angry
- Fear
- Happy
- Sad

The intended application is:

- classroom engagement monitoring
- child behavior awareness
- assistive emotional signal detection

This project is **not**:

- a medical tool
- a psychological diagnosis system
- a surveillance platform
- a production app

Avoid overstating capabilities.

---

## Model Requirements

Primary model:

- MobileNetV2 transfer learning

Backup model:

- simple CNN baseline

Framework:

- TensorFlow / Keras

Expected architecture:

```txt
Input image
→ Resize to 224x224
→ Convert grayscale to RGB
→ MobileNetV2 backbone
→ GlobalAveragePooling2D
→ Dense(128, relu)
→ Dropout(0.5)
→ Dense(4, softmax)
```

Freeze the pretrained backbone initially.

---

## Dataset Requirements

Dataset:

- Kaggle FER-2013 image-folder dataset

Expected location:

```txt
data/raw/train/{angry,fear,happy,sad}
data/raw/test/{angry,fear,happy,sad}
```

Expected preprocessing:

1. load images with `tf.keras.utils.image_dataset_from_directory`
2. restrict labels to Angry, Fear, Happy, and Sad
3. resize images to 224x224
4. convert/load images as RGB
5. create a validation split from the training folder
6. save lightweight dataset metadata and class mapping

Do **not** commit massive datasets unless explicitly requested.

---

## Success Metrics

Primary metric:

- Accuracy

Secondary metrics:

- Macro F1-score
- Confusion matrix
- Precision / Recall

Generate:

```txt
results/metrics.txt
results/visualizations/confusion_matrix.png
```

After evaluation, update the README with actual metrics.

---

## Primary Agent Tasks

### Task 1 — Verify Environment

Help verify:

- dependencies
- TensorFlow install
- notebook execution
- FER-2013 folder paths

Expected command:

```bash
pip install -r requirements.txt
```

---

### Task 2 — Run Preprocessing

Expected command:

```bash
python src/data_processing.py
```

Confirm:

- train/test class folders are present
- image counts are reasonable
- labels are mapped correctly

Expected generated files:

```txt
data/processed/class_map.csv
data/processed/dataset_summary.txt
```

---

### Task 3 — Train Model

Expected command:

```bash
python src/train.py
```

Goals:

- model trains successfully
- model saves correctly
- no GPU/runtime issues
- no broken imports

Expected output:

```txt
models/trained/kidmood_mobilenetv2.keras
```

---

### Task 4 — Evaluate Model

Expected command:

```bash
python src/evaluate.py
```

Generate:

```txt
results/metrics.txt
results/visualizations/confusion_matrix.png
```

Then update README with real metrics.

---

### Task 5 — Demo Preparation

Help generate:

- 3–5 sample prediction images
- 1 failure/difficult example
- clean inference outputs

Expected command:

```bash
python src/inference.py \
  --image data/sample/happy_demo.jpg \
  --model models/trained/kidmood_mobilenetv2.keras
```

---

### Task 6 — Final Polish

Help:

- clean README formatting
- ensure notebooks run
- ensure presentation aligns with final repo
- ensure demo workflow is smooth
- update `docs/AI_usage_log.md`
- update `demo_video_link.txt` after recording

---

## Important Constraints

Do **not**:

- build a giant web app
- add unnecessary APIs
- add cloud deployment
- rewrite project architecture
- introduce Docker/Kubernetes
- convert this into a SaaS platform
- add authentication systems
- create a database layer
- add a webcam requirement unless explicitly requested

This is an academic final project, not a startup.

---

## Notebook Expectations

The notebooks should:

- execute cleanly
- be easy for a professor to follow
- contain markdown explanations
- contain visual outputs
- avoid overengineering

The demo notebook should:

- load a trained model
- run inference on images
- show prediction labels/confidence

---

## README Expectations

README must contain:

- project overview
- problem statement
- technical approach
- dataset plan
- setup instructions
- metrics/results
- screenshots/visualizations
- demo instructions
- future improvements
- AI usage disclosure

After training, update README with actual metrics/results.

---

## Final Presentation Goal

The final presentation should communicate:

1. the problem
2. the ML pipeline
3. preprocessing
4. transfer learning
5. results
6. strengths/limitations
7. future improvements

The project should feel:

- organized
- reproducible
- technically competent
- realistic

---

## Demo Video Goal

The demo video should:

- show the repository
- show inference working
- show metrics/confusion matrix
- explain one success case
- explain one failure case

A screen-recorded demo is acceptable.

Real-time webcam inference is optional and not required.

---

## Code Style

Prefer:

- readable code
- beginner-friendly structure
- comments where useful
- simple functions
- reproducibility

Avoid:

- unnecessary abstraction
- advanced metaprogramming
- overcomplicated patterns

---

## When Modifying Files

Before changing anything:

1. preserve structure
2. preserve assignment alignment
3. avoid breaking notebook imports
4. keep scripts runnable from repo root
5. keep paths relative whenever possible

---

## Final Priority Order

Priority:

1. project runs successfully
2. training works
3. evaluation works
4. demo works
5. repo looks professional
6. polish/documentation

Not priority:

- advanced production engineering
- perfect accuracy
- real-time deployment
- cloud hosting
- GUI dashboard

---

## Final Deliverable Checklist

Before considering the project complete, verify:

- [ ] FER-2013 loads successfully
- [ ] preprocessing runs
- [ ] model trains
- [ ] evaluation runs
- [ ] inference works
- [ ] `metrics.txt` generated
- [ ] `confusion_matrix.png` generated
- [ ] demo notebook works
- [ ] README updated with actual results
- [ ] sample images exist
- [ ] demo video recorded
- [ ] presentation finalized
- [ ] AI usage log updated
- [ ] repository is clean and understandable
