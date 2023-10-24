# 引入必要的库和包
# import torch.cuda
# import ultralytics
from ultralytics import YOLO
import cv2

# 查看版本
# print(ultralytics.__version__)
# print(torch.cuda.is_available())

# 加载模型
model = YOLO(model="yolov8n-cls.pt")  # yolov8n-cls.pt 是分类模型的权重

# 查看模型
# print(model.model)

# 从视频文件中预测
video_path = "./西安之行.mp4"  # 将video_path改为0可以从摄像头获取视频流
cap = cv2.VideoCapture(video_path)

while cap.isOpened():  # 只要文件时打开的我们就可以一直读
    # read()返回两个值，
    # status是一个状态，表示读取是否成功，成功为True
    # 如果读取成功，frame里存的就是图像
    status, frame = cap.read()
    if not status:  # 如果读取不成功，则结束读取
        break

    results = model.predict(source=frame)  # 分类的结果就保存在results中，
    # print(len(results))  # results时一个列表，长度为1，即只有一个元素
    result = results[0]
    anno_frame = result.plot()
    cv2.imshow(winname="frame", mat=anno_frame)

    # delay=10表示每10毫秒检测一下是否有键入，27表示esc键
    # 即如果10毫秒内按了esc键就退出，否则一直继续循环
    if cv2.waitKey(delay=10) == 27:
        break

cap.release()  # 释放所占用的所有资源
cv2.destroyAllWindows()  # 关闭所有开着的窗口
