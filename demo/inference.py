from ultralytics import YOLO
import cv2
import sys

CLASS_NAMES = ['contact_item', 'chat_title', 'left_bubble', 'right_bubble',
               'input_box', 'send_btn', 'tab_chat', 'nav_unread']

model = YOLO('models/best.pt')

img_path = sys.argv[1] if len(sys.argv) > 1 else 'test.jpg'
results = model.predict(img_path, conf=0.5, iou=0.5, imgsz=640, verbose=False)[0]

img = cv2.imread(img_path)
for box in results.boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f'{CLASS_NAMES[cls_id]} {conf:.2f}', 
                (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    print(f'{CLASS_NAMES[cls_id]:15s} conf={conf:.2f} [{x1},{y1},{x2},{y2}]')

cv2.imwrite('result.jpg', img)
print(f'Results saved to result.jpg')
