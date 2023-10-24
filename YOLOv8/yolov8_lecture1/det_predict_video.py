# 引入必要的库和包
import ultralytics
from ultralytics import YOLO
import cv2

# 查看版本
print(ultralytics.__version__)

# 加载模型
model = YOLO(model="yolov8n.pt")  # yolov8n.pt 是物体检测模型的权重

# 查看模型
print(model.model)

#
video_path = "./西安之行.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    status, frame = cap.read()
    if not status:
        break
    results = model.predict(source=frame, save=True)
    result = results[0]
    anno_frame = result.plot()
    cv2.imshow(winname="frame", mat=anno_frame)
    if cv2.waitKey(delay=1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
