# Demo Video Script (3-5 Minutes)

## 0:00 - 0:30 Introduction

Hi, I'm Jason Trimble, and this is KidMood: Classroom Emotion Recognition. This project uses computer vision to classify facial expressions into Happy, Sad, Angry, or Fear.

## 0:30 - 1:15 Project Setup

Show the GitHub repository structure. Point out:
- README.md
- requirements.txt
- notebooks
- src scripts
- results folder
- docs/AI_usage_log.md

## 1:15 - 2:15 System Working

Run the demo notebook or inference script on sample images.

Example command:

```bash
python src/inference.py --image data/sample/happy_demo.jpg --model models/trained/kidmood_mobilenetv2.keras
```

Explain the prediction label and confidence score.

## 2:15 - 3:15 Results

Show:
- metrics.txt
- confusion matrix
- prediction examples

## 3:15 - 4:15 Success and Failure Case

Show one correct prediction and one difficult case. Explain why the model may struggle, such as low resolution or similar expressions.

## 4:15 - 5:00 Conclusion

Summarize what worked, what was challenging, and future improvements.
