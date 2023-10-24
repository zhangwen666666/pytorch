# 引入必要的库和包
from ultralytics import YOLO
import cv2
import os
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == "__main__":
    # 构建模型
    model = YOLO(model="./runs/classify/train/weights/last.pt")
    # 训练模型
    # model.train(data='../datasets/animal', epochs=2000, imgsz=64)
    # 验证模型
    # metrics = model.val()
    # 预测

    results = model(source='../datasets/animal/test/cat/1.png', save=True)
    result = results[0]
    anno_frame = result.plot()
    cv2.imshow(winname="frame", mat=anno_frame)
    cv2.waitKey(0)

