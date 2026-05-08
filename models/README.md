# Model Documentation

## Model Folders

```txt
models/
├── pretrained/   # notes or links for pretrained backbones
└── trained/      # trained KidMood models
```

## Primary Model

- MobileNetV2
- Pretrained on ImageNet
- Frozen backbone during first training stage
- Custom 4-class softmax head for Angry, Fear, Happy, Sad

## Saved Model Path

After training, the model should be saved as:

```txt
models/trained/kidmood_mobilenetv2.keras
```

## Notes

Large trained model files can be committed if allowed by GitHub size limits, or uploaded separately and linked in this file.
