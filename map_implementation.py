from ultralytics import YOLO

model = YOLO('yolov8n.pt')

res = model.train(
    data='data/data.yaml',
    name='test1',
    imgsz=640,
    epochs=100,
    batch=2,
    device='',
    patience=20,
)