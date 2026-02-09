# AI-CliniScan — Chest X-ray Abnormality Detection

## 📌 Overview


AI-CliniScan is a medical AI project designed to detect 14 chest X-ray abnormalities using YOLOv8 object detection.
The project focuses on building an end-to-end medical imaging pipeline, from DICOM preprocessing to model training, evaluation, and analysis.

🧠 Key Features

DICOM to image preprocessing

YOLO-format annotation generation

Object detection on chest X-rays

Class-wise performance evaluation

Test Time Augmentation (TTA)

🧬 Detected Abnormalities (14)

Aortic enlargement, Atelectasis, Calcification, Cardiomegaly, Consolidation,
ILD, Infiltration, Lung Opacity, Nodule, Other lesion,
Pleural effusion, Pleural thickening, Pneumothorax, Fibrosis

## 🧪 Model Details

| Component | Value |
|---------|------|
| Architecture | YOLOv8-Nano |
| Parameters | 3.01M |
| Input Size | 640 × 640 |
| Epochs | 30 |
| Training Time | ~5.3 hours (CPU) |

📊 Performance Summary

Best mAP50: 0.246

Final mAP50: 0.235

Precision: 0.283

Recall: 0.272

Strong performance observed for Cardiomegaly and Aortic Enlargement.

⚙️ Tech Stack

Python

YOLOv8 (Ultralytics)

OpenCV

NumPy, Pandas

Pydicom

Albumentations

Jupyter Notebook

## 🚀 Run Inference

```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model.predict(
    source="sample_xray.png",
    conf=0.15,
    iou=0.4,
    augment=True
)




📈 Future Improvements

Larger YOLOv8 models (YOLOv8m)

Higher resolution training

Class balancing

Ensemble techniques

📜 Disclaimer

This project is for educational and research purposes only and is not intended for clinical diagnosis.

```md
- GitHub (version control & project hosting)
