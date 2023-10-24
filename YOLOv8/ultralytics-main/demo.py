from ultralytics import YOLO

yolo = YOLO("./yolov8n.pt", task='detect')
result = yolo(source='./data/1.jpg',save=True)
