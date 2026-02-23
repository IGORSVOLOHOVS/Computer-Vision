from ultralytics import YOLO

# Загрузка модели и источника
model = YOLO('yolov8n.pt') 
source = 'https://ultralytics.com/images/bus.jpg'

# Выполнение инференса
results = model.predict(source)
result = results[0] # Получаем результат для первого (и единственного) изображения

# Извлечение информации о боксах
boxes = result.boxes  

def box_area(box):
    x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
    return (x_max - x_min) * (y_max - y_min)

def get_box_max(box1, box2):
    if box1 is None:
        return box2
    if box2 is None:
        return box1
    if box_area(box1) > box_area(box2):
        return box1
    else:
        return box2

def get_box_info(box):
    box_class_id = box.cls[0].item()
    box_class_name = model.names[box_class_id]
    print(f'Name of class is {box_class_name} with ID:{box_class_id}')

    confidence_score = box.conf[0].item()
    print(f'Confidence score is {confidence_score}')

    area = box_area(box)
    print(f'Area of box is {area}')

max_box = None

for box in boxes:
    # Здесь мы можем получить доступ к свойствам каждого бокса
    coords = box.xyxy[0].tolist()
    class_id = int(box.cls[0].item())
    conf = box.conf[0].item()
    class_name = model.names[class_id]

    max_box = get_box_max(max_box, box)

get_box_info(max_box)
    
    # ... ваш код для заданий ...