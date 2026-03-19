# Skin Cancer Detection

A web app that detects skin cancer from images using deep learning.

> ⚠️ For educational purposes only. Not for real medical diagnosis.

## What it does
- Upload a skin lesion image or use camera
- Predicts Benign or Malignant
- Shows confidence score and Grad-CAM heatmap

## Model
- Dataset: HAM10000 (10,015 images)
- Model: ResNet50 transfer learning
- AUC: 0.90 | Recall: 85.6% | Accuracy: 75.4%

## How to run

1. Install dependencies
```bash
pip install tensorflow flask opencv-python pillow numpy
```

2. Train the model by running the notebook

3. Run the app
```bash
python app.py
```

4. Open browser at `http://127.0.0.1:5000`
