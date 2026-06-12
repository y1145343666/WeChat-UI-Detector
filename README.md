# WeChat UI Detector 微信UI检测器

[![License](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)

A YOLOv13l-based object detection model for recognizing UI elements in WeChat chat screenshots. This model identifies 8 categories of interface components including chat bubbles, input fields, contact entries, and navigation elements.

## Model Overview

- **Architecture**: YOLOv13l (Large)
- **Input resolution**: 640x640
- **Output**: Bounding boxes with class labels and confidence scores for 8 UI element types
- **Training framework**: Ultralytics

## Detection Capabilities

The model detects the following UI components in WeChat chat interface screenshots:

| Class ID | Component | Description | Typical count per screenshot |
|:--------:|:----------|:------------|:---------------------------:|
| 0 | contact_item | Contact entries in chat list | ~8 |
| 1 | chat_title | Chat conversation header bar | 1 |
| 2 | left_bubble | Incoming messages from others | ~2 |
| 3 | right_bubble | Outgoing messages from self | ~1 |
| 4 | input_box | Text input area at bottom | 1 |
| 5 | send_btn | Send button | 1 |
| 6 | tab_chat | Bottom navigation tabs | 4 |
| 7 | nav_unread | Unread notification badges | ~3 |

## Performance

Validation was conducted on an augmented test set containing images with random stretching (0.7x-1.4x), rotation (+-10 degrees), HSV color shifts, Gaussian blur, and noise injection to simulate real-world variations.

| Metric | Value | Notes |
|:-------|:-----:|:------|
| mAP@0.5 | 91.67% | Mean average precision at IoU threshold 0.5 |
| mAP@0.5:0.95 | 77.33% | Mean average precision across IoU thresholds 0.5 to 0.95 |
| Precision | 87.05% | Proportion of correct detections among all detections |
| Recall | 87.64% | Proportion of detected targets among all ground truth |
| Training epochs | 87 | Early stopping with patience of 30 epochs |

## Training Data

- **Total**: 7,518 images (7,393 for training, 125 for validation)
- **Total annotations**: 186,696 bounding boxes
- **Sources**: Real WeChat screenshots + synthetic data
- **Augmentation**: stretch (0.7-1.4x), rotation (+-10 deg), HSV shift, Gaussian blur, random noise

## Quick Start

### Installation

```bash
pip install ultralytics opencv-python
```

### Inference

```python
from ultralytics import YOLO

model = YOLO('models/best.pt')

# Run detection
results = model.predict('screenshot.jpg', conf=0.5, iou=0.5, imgsz=640)

# Access results
for box in results[0].boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    print(f'Class {cls_id}: {conf:.2f} at [{x1},{y1},{x2},{y2}]')
```

### Batch Processing

```python
from ultralytics import YOLO

model = YOLO('models/best.pt')
results = model.predict('data/sample/', conf=0.5, save=True)
```

### Custom Training

To fine-tune the model on your own data:

```python
from ultralytics import YOLO

model = YOLO('models/best.pt')
model.train(
    data='your_dataset.yaml',
    epochs=100,
    lr0=0.00005,
    batch=8,
    imgsz=640,
    device=0
)
```

## Training Process

The model was trained for 87 epochs with the following loss progression:

- Initial box loss: 2.65 (epoch 1)
- Final box loss: 1.58 (epoch 86)
- Learning rate schedule: cosine decay from 6e-5 to 4.3e-5

## Deployment

For edge deployment on RK3588 devices:

1. Distill the model to a smaller architecture (e.g., YOLO11n)
2. Export to ONNX format
3. Convert to RKNN format for RK3588 NPU acceleration

## Repository Structure

```
WeChat-UI-Detector/
├── models/
│   └── best.pt          # Trained model weights (159 MB)
├── demo/
│   └── inference.py     # Inference script
├── training/
│   └── augment.py       # Data augmentation utilities
├── data/
│   ├── dataset.yaml     # Dataset configuration
│   └── sample/          # Example images and labels
├── docs/                # Documentation
├── deploy/              # Deployment configurations
├── README.md
└── LICENSE              # AGPL-3.0
```

## License

AGPL-3.0

## Acknowledgments

- Thanks to Trae AI assistant for technical guidance and training optimization
- Ultralytics for the YOLO framework
