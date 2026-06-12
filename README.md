# WeChat UI Detector 微信UI检测器

[![License](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)

---

**[English](#english) | [中文](#中文)**

---

## <a name="english"></a>English

A YOLOv13l-based object detection model for recognizing UI elements in WeChat chat screenshots. This model identifies 8 categories of interface components including chat bubbles, input fields, contact entries, and navigation elements.

### Model Overview

- **Architecture**: YOLOv13l (Large)
- **Parameters**: 27,572,271
- **Computational cost**: 89.0 GFLOPs
- **Input resolution**: 640x640
- **Output**: Bounding boxes with class labels and confidence scores for 8 UI element types
- **Training framework**: Ultralytics

### Performance

Validation was conducted on an augmented test set containing images with random stretching (0.7x-1.4x), rotation (+-10 degrees), HSV color shifts, Gaussian blur, and noise injection to simulate real-world variations.

| Metric | Value | Description |
|:-------|:-----:|:------------|
| mAP@0.5 | 91.67% | Mean average precision at IoU threshold 0.5 |
| mAP@0.5:0.95 | 77.33% | Mean average precision across IoU thresholds 0.5 to 0.95 |
| Precision | 87.05% | Proportion of correct detections among all detections |
| Recall | 87.64% | Proportion of detected targets among all ground truth |

### Detection Capabilities

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

### Training Configuration

| Parameter | Value | Description |
|:----------|:-----:|:------------|
| optimizer | AdamW | Adaptive moment estimation with weight decay |
| lr0 | 6e-5 | Initial learning rate |
| batch | 8 | Batch size |
| epochs | 300 | Maximum training epochs |
| patience | 30 | Early stopping patience |

### Training Data

- **Total**: 7,518 images (7,393 for training, 125 for validation)
- **Total annotations**: 186,696 bounding boxes
- **Sources**: Real WeChat screenshots + synthetic data
- **Augmentation**: stretch (0.7-1.4x), rotation (+-10 deg), HSV shift, Gaussian blur, random noise

### Quick Start

```bash
pip install ultralytics opencv-python
```

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

### License

AGPL-3.0

---

## <a name="中文"></a>中文

基于 YOLOv13l 的微信聊天界面UI元素检测模型，可识别 8 类界面组件，包括聊天气泡、输入框、联系人条目和导航元素。

### 模型概述

- **架构**: YOLOv13l (Large)
- **参数量**: 27,572,271
- **计算量**: 89.0 GFLOPs
- **输入尺寸**: 640x640
- **输出**: 边界框 + 类别标签 + 置信度
- **训练框架**: Ultralytics

### 性能指标

验证集包含随机拉伸(0.7x-1.4x)、旋转(+-10度)、HSV偏移、高斯模糊、噪声注入等增强，用于模拟真实场景中的各种变化。

| 指标 | 数值 | 说明 |
|:----|:----:|:-----|
| mAP@0.5 | 91.67% | IoU阈值0.5下的平均精度 |
| mAP@0.5:0.95 | 77.33% | IoU阈值0.5至0.95的平均精度 |
| Precision | 87.05% | 检出目标中正确比例 |
| Recall | 87.64% | 真实目标中被检出比例 |

### 检测类别

| ID | 类别 | 说明 | 每张图数量 |
|:--:|:----|:----|:---------:|
| 0 | contact_item | 联系人列表条目 | ~8 |
| 1 | chat_title | 聊天标题栏 | 1 |
| 2 | left_bubble | 对方发送的消息气泡 | ~2 |
| 3 | right_bubble | 自己发送的消息气泡 | ~1 |
| 4 | input_box | 底部输入框 | 1 |
| 5 | send_btn | 发送按钮 | 1 |
| 6 | tab_chat | 底部导航标签 | 4 |
| 7 | nav_unread | 未读消息角标 | ~3 |

### 训练配置

| 参数 | 数值 | 说明 |
|:----|:----:|:-----|
| 优化器 | AdamW | 自适应矩估计+权重衰减 |
| 初始学习率 | 6e-5 | |
| 批大小 | 8 | |
| 最大轮数 | 300 | |
| 早停耐心值 | 30 | 30轮无提升自动停止 |

### 训练数据

- **总图片**: 7,518 张 (7,393 训练 + 125 验证)
- **总标注框**: 186,696 个
- **数据来源**: 真实截图 + 合成数据
- **增强方式**: 拉伸(0.7-1.4x)、旋转(+-10度)、HSV偏移、高斯模糊、随机噪声

### 快速开始

```bash
pip install ultralytics opencv-python
```

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

### 许可证

AGPL-3.0

---

## Acknowledgments 致谢

- Thanks to **Trae AI assistant** for technical guidance and training optimization
- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLO framework
