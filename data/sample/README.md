# Sample Data

This folder is for small test images used during the live demo.

Recommended final demo setup:
1. Copy 3-5 images from `data/raw/test/{angry,fear,happy,sad}` into this folder.
2. Include at least:
   - one likely success case
   - one difficult/failure case
3. Run inference using:

```bash
python src/inference.py --image data/sample/happy_demo.jpg --model models/trained/kidmood_mobilenetv2.keras
```
