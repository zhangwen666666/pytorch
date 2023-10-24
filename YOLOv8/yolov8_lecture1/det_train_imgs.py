# 引入必要的库和包
from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


if __name__ == "__main__":
    # 构建模型
    model = YOLO(model="yolov8n.yaml")
    # 训练模型
    model.train(data='./objects.yaml', epochs=50, imgsz=640)
    results = model(source='./girl01.jpg', save=True)