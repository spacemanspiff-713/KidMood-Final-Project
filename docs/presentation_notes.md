# Final Presentation Notes

## Slide 1 - Title
Introduce KidMood, Jason Trimble, and Tier 2 transfer learning.

## Slide 2 - Problem & Motivation
Explain that teachers cannot monitor every student's emotional state at all times.

## Slide 3 - Solution Overview
Explain that KidMood classifies face images into Happy, Sad, Angry, or Fear.

## Slide 4 - Technical Approach
Discuss image classification, MobileNetV2, TensorFlow/Keras, and transfer learning.

## Slide 5 - Model Architecture
Walk through preprocessing, MobileNetV2 backbone, dense layer, dropout, and softmax output.

## Slide 6 - Dataset & Preprocessing
Explain the Kaggle FER-2013 folder dataset, selected classes, Keras directory loading, 224x224 resizing, and RGB conversion.

Final evaluation results:
- Accuracy: 0.5515
- Macro F1-score: 0.4887
- Best class: Happy, with 0.73 F1-score
- Harder classes: Fear and Angry

## Slide 7 - Live Demo
Run the demo notebook or inference script on 2-3 sample images.

## Slide 8 - Results
Show accuracy, macro F1-score, confusion matrix, and inference time after training.

## Slide 9 - Success & Failure Cases
Show examples and explain why the model succeeded or failed.

## Slide 10 - Key Learnings
Discuss transfer learning, class confusion, and dataset limitations.

## Slide 11 - Future Work
Mention webcam demo, face detection, better datasets, and trend dashboard.

## Slide 12 - Acknowledgments
Thank instructor, dataset source, TensorFlow/Keras, and AI tools.
