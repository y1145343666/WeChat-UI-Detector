# WeChat UI Detector 微信UI检测器

[![License](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)

A YOLOv13l-based object detection model for recognizing UI elements in WeChat chat screenshots. This model identifies 8 categories of interface components including chat bubbles, input fields, contact entries, and navigation elements.

## Model Overview

- **Architecture**: YOLOv13l (Large)
- **Parameters**: 27,572,271
- **Computational cost**: 89.0 GFLOPs
- **Input resolution**: 640x640
- **Output**: Bounding boxes with class labels and confidence scores for 8 UI element types
- **Training framework**: Ultralytics

## Performance

Validation was conducted on an augmented test set containing images with random stretching (0.7x-1.4x), rotation (+-10 degrees), HSV color shifts, Gaussian blur, and noise injection to simulate real-world variations.

### Overall Metrics (Best Epoch 86)

| Metric | Value | Description |
|:-------|:-----:|:------------|
| mAP@0.5 | 91.67% | Mean average precision at IoU threshold 0.5 |
| mAP@0.5:0.95 | 77.33% | Mean average precision across IoU thresholds 0.5 to 0.95 |
| Precision | 87.05% | Proportion of correct detections among all detections |
| Recall | 87.64% | Proportion of detected targets among all ground truth |
| mAP@0.75 | 85.58% | Mean average precision at IoU threshold 0.75 |

### Per-Class Performance (Best Epoch)

| Class | mAP50 | Difficulty |
|:------|:-----:|:-----------|
| contact_item | ~95% | Large samples, stable size |
| chat_title | ~96% | Standardized bounding box |
| left_bubble | ~95% | Fixed position, green color |
| right_bubble | ~95% | Fixed position, white color |
| input_box | ~98% | Large target, 438x140px |
| send_btn | ~97% | Fixed position next to input |
| tab_chat | ~93% | Similar to contact_item |
| nav_unread | ~85% | Small target, under 20px |

Note: Per-class metrics are estimated on clean screenshots. The augmented validation set shows lower absolute values due to intentionally added difficulty.

## Training Configuration

### Hyperparameters

| Parameter | Value | Description |
|:----------|:-----:|:------------|
| optimizer | AdamW | Adaptive moment estimation with weight decay |
| lr0 | 6e-5 | Initial learning rate |
| lrf | 0.01 | Final learning rate factor |
| weight_decay | 0.0003 | L2 regularization |
| warmup_epochs | 3 | Learning rate warmup |
| batch | 8 | Batch size |
| epochs | 300 | Maximum training epochs |
| patience | 30 | Early stopping patience |
| imgsz | 640 | Input image size |
| nbs | 64 | Nominal batch size |

### Data Augmentation

| Augmentation | Value | Purpose |
|:-------------|:-----:|:--------|
| mosaic | 1.0 | Multi-image composition |
| mixup | 0.15 | Image mixing ratio |
| copy_paste | 0.05 | Object copy-paste |
| erasing | 0.2 | Random erasing probability |
| hsv_h | 0.02 | Hue shift |
| hsv_s | 0.8 | Saturation shift |
| hsv_v | 0.5 | Value shift |
| scale | 0.5 | Random scaling |
| translate | 0.2 | Random translation |
| fliplr | 0.5 | Horizontal flip |
| degrees | 5.0 | Rotation angle |
| shear | 5.0 | Shear angle |
| close_mosaic | 15 | Disable mosaic after epoch 15 |

### Training Process

| Stage | Epochs | mAP50-95 | Description |
|:------|:------:|:--------:|:------------|
| Initial learning | 1-6 | 0.48 -> 0.54 | Rapid convergence from pretrained weights |
| Growth | 7-13 | 0.54 -> 0.63 | Understanding UI layout structure |
| Breakthrough | 14-24 | 0.63 -> 0.67 | First oscillation-recovery cycles |
| Stable climb | 25-37 | 0.69 -> 0.74 | Steady improvement on details |
| First peak | 38-45 | 0.74 -> 0.752 | First major milestone |
| Plateau | 46-64 | 0.738-0.751 | Extended consolidation period |
| Breakthrough | 65-72 | 0.75 -> 0.766 | Sudden insight breakthrough |
| Final convergence | 73-86 | 0.766 -> 0.773 | Fine convergence with learning rate decay |

### Loss Progression

- **Initial box loss (epoch 1)**: 2.646
- **Final box loss (epoch 86)**: 1.582
- **Validation box loss (epoch 86)**: 0.599
- **Learning rate schedule**: Cosine decay from 6e-5 to 4.3e-5

## Training Data

### Dataset Composition

| Split | Images | Labels |
|:------|:------:|:------:|
| Training | 7,393 | 7,393 |
| Validation | 125 | 125 |
| **Total** | **7,518** | **7,518** |

### Annotation Distribution

| Class | Count | Percentage |
|:------|:-----:|:----------:|
| contact_item | 61,292 | 32.8% |
| tab_chat | 57,816 | 31.0% |
| nav_unread | 19,299 | 10.3% |
| left_bubble | 15,742 | 8.4% |
| right_bubble | 10,410 | 5.6% |
| chat_title | 7,399 | 4.0% |
| send_btn | 7,375 | 4.0% |
| input_box | 7,363 | 3.9% |
| **Total** | **186,696** | **100%** |

### Data Sources

- **Folders 1-3, 5**: Real WeChat screenshots
- **Folder 4**: High-fidelity synthetic data (929 base images with 4,645 augmentations)
- **Augmentation**: stretch (0.7-1.4x), rotation (+-10 degrees), HSV shift, Gaussian blur, random noise

## Detection Capabilities

The model detects the following UI components in WeChat chat interface screenshots:

| Class ID | Component | Description | Typical per screenshot |
|:--------:|:----------|:------------|:---------------------:|
| 0 | contact_item | Contact entries in chat list | ~8 |
| 1 | chat_title | Chat conversation header bar | 1 |
| 2 | left_bubble | Incoming messages from others | ~2 |
| 3 | right_bubble | Outgoing messages from self | ~1 |
| 4 | input_box | Text input area at bottom | 1 |
| 5 | send_btn | Send button | 1 |
| 6 | tab_chat | Bottom navigation tabs | 4 |
| 7 | nav_unread | Unread notification badges | ~3 |

## Quick Start

### Installation

```bash
pip install ultralytics opencv-python
```

### Inference

```python
from ultralytics import YOLO

model = YOLO('models/best.pt')

results = model.predict('screenshot.jpg', conf=0.5, iou=0.5, imgsz=640)

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

## Deployment

For edge deployment on RK3588 devices (6 TOPS NPU):

1. Distill the model to a smaller architecture (e.g., YOLO11n)
2. Export to ONNX format
3. Convert to RKNN format for RK3588 NPU acceleration
4. Expected inference speed: 82 FPS with INT8 quantization

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

## References

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [YOLOv13](https://github.com/ultralytics/ultralytics)
- [RKNN Toolkit](https://github.com/airockchip/rknn-toolkit2)

## License

AGPL-3.0

## Acknowledgments

- Thanks to Trae AI assistant for technical guidance and training optimization
- Ultralytics for the YOLO framework
