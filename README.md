# WeChat UI Detector 微信UI检测器

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)

> 一个人 + 一张 RTX 4060Ti = 一个敢跟大厂对标的微信UI检测模型

## 简介

WeChat UI Detector 是一个基于 YOLOv13l 的微信聊天界面UI元素检测模型，能够精准识别微信截图中的 8类核心UI元素。

在地狱级难度验证集（拉伸/旋转/噪声/模糊）上达到 **mAP50-95 = 0.7733**，真实截图预估精度 95%+。

### 检测效果

| 类别 | 说明 | 每张图数量 |
|:----|:----|:---------:|
| contact_item | 联系人条目 | ~8个 |
| chat_title | 聊天标题栏 | 1个 |
| left_bubble | 对方消息气泡 | ~2个 |
| right_bubble | 自己消息气泡 | ~1个 |
| input_box | 输入框 | 1个 |
| send_btn | 发送按钮 | 1个 |
| tab_chat | 底部导航标签 | 4个 |
| nav_unread | 未读角标 | ~3个 |

## 训练成绩

| 指标 | 数值 | 说明 |
|:----|:----:|:-----|
| mAP50 | 91.67% | 地狱级验证集 |
| mAP50-95 | 77.33% | 地狱级验证集 |
| Precision | 87.05% | 精确率 |
| Recall | 87.64% | 召回率 |
| 训练轮数 | 87轮 | 自动早停 |
| 训练数据 | 7,518张 | 内含 186,696 个标注框 |

验证集包含：stretch_h/w (0.7~1.4x)、rotate (+-10度)、brightness/contrast、HSV shift、Gaussian blur、random noise

## 快速开始

### 安装
```
pip install ultralytics opencv-python
```

### 推理
```python
from ultralytics import YOLO

model = YOLO('models/best.pt')
results = model.predict('screenshot.jpg', conf=0.5, iou=0.5, imgsz=640)
results[0].show()

for box in results[0].boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    print(f'Class {cls_id}: {conf:.2f} at [{x1},{y1},{x2},{y2}]')
```

### 批量处理
```python
from ultralytics import YOLO
model = YOLO('models/best.pt')
results = model.predict('data/sample/', conf=0.5, save=True)
```

## 训练数据
- 总图片: 7,518 张 (7,393 训练 + 125 验证)
- 总标注框: 186,696 个
- 类别: 8类微信UI元素
- 数据来源: 真实截图 + 高拟真合成数据

## 学习历程
```
E01~E06:  懵懂期    0.48 -> 0.54
E07~E13:  成长期    0.54 -> 0.63
E14~E24:  震荡突破  0.63 -> 0.67
E25~E37:  稳步爬升  0.69 -> 0.74
E38~E45:  第一巅峰  0.74 -> 0.752
E46~E64:  平台蓄力  思考20轮
E65~E72:  顿悟突破  0.75 -> 0.766
E73~E86:  二次突破  0.766 -> 0.773
```

## 致谢
- Thanks to **Trae AI assistant** for technical guidance and training optimization throughout this project
- Ultralytics for the YOLO framework

---
**如果这个项目对你有帮助，请给一个 Star！**
