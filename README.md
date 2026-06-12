# WeChat UI Detector

[![License](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)

WeChat UI Detector is a YOLOv13l-based model for detecting UI elements in WeChat chat screenshots, supporting 8 object classes.

## Performance

| Metric | Value | Test Set |
|:-------|:-----:|:---------|
| mAP50 | 91.67% | Augmented validation set (stretch/rotate/noise/blur) |
| mAP50-95 | 77.33% | Augmented validation set |
| Precision | 87.05% | Augmented validation set |
| Recall | 87.64% | Augmented validation set |
| Training epochs | 87 | Early stopping |
| Training data | 7,518 images | 186,696 annotations total |

### Classes

| ID | Class | Description |
|:--:|:------|:------------|
| 0 | contact_item | Contact list entries |
| 1 | chat_title | Chat header bar |
| 2 | left_bubble | Messages from others |
| 3 | right_bubble | Messages from self |
| 4 | input_box | Text input area |
| 5 | send_btn | Send button |
| 6 | tab_chat | Bottom navigation tabs |
| 7 | nav_unread | Unread notification badges |

## Quick Start

### Installation

```
pip install ultralytics opencv-python
```

### Inference

```python
from ultralytics import YOLO

model = YOLO('models/best.pt')
results = model.predict('screenshot.jpg', conf=0.5, iou=0.5, imgsz=640)
results[0].show()
```

### Batch Processing

```python
from ultralytics import YOLO
model = YOLO('models/best.pt')
results = model.predict('data/sample/', conf=0.5, save=True)
```

## Training Data

- Total: 7,518 images (7,393 train + 125 validation)
- Total annotations: 186,696 bounding boxes
- Classes: 8 WeChat UI element types
- Sources: Real screenshots + synthetic data
- Augmentation: stretch (0.7-1.4x), rotate (+-10 deg), HSV shift, blur, noise

## License

AGPL-3.0

## Acknowledgments

- Thanks to Trae AI assistant for technical guidance and training optimization
- Ultralytics for the YOLO framework
