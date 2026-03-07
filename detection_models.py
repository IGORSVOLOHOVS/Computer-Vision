import torch
import torchvision
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights, 
    FCOS_ResNet50_FPN_Weights
)
from ultralytics import YOLO
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO


COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Создаём фиксированную палитру цветов
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(COCO_CLASSES), 3), dtype=np.uint8)


def draw_boxes(image, boxes, labels, scores, threshold=0.5, allowed_classes=None):
    image_np = np.array(image.copy())
    for i, box in enumerate(boxes):
        if scores[i] > threshold:
            class_id = int(labels[i])

            # Модели torchvision возвращают 1-индексированные ID (1-91),
            # а YOLO — 0-индексированные (0-79). Мы стандартизировали вывод YOLO
            # к 1-индексному формату, поэтому здесь мы всегда вычитаем 1, чтобы
            # получить правильный индекс для нашего 0-индексированного списка COCO_CLASSES.
            class_index = class_id - 1

            # Проверяем, что индекс в пределах нашего списка (важно для torchvision)
            if class_index < 0 or class_index >= len(COCO_CLASSES): continue
            
            label_text = COCO_CLASSES[class_index]
            
            # Логика фильтрации
            if allowed_classes is not None and label_text not in allowed_classes:
                continue

            color = [int(c) for c in COLORS[class_index]]
            x_min, y_min, x_max, y_max = map(int, box)

            # Адаптивный размер шрифта
            box_width = x_max - x_min
            font_scale = max(0.4, min(1.0, box_width / 200.0))
            thickness = max(1, int(font_scale * 2))

            # Полный текст подписи
            display_text = f"{label_text}: {scores[i]:.2f}"
            
            # Расчёт размера текстового блока
            (text_width, text_height), baseline = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Позиция текста
            text_x = x_min
            text_y = y_min - 5
            
            # Сдвигаем текст, если он выходит за пределы изображения
            if text_y < text_height:
                text_y = y_min + text_height + 5

            # Отрисовка фона для текста
            if show_text:
                cv2.rectangle(
                    image_np, 
                    (text_x, text_y - text_height - baseline), 
                    (text_x + text_width, text_y + baseline), 
                    color, 
                    cv2.FILLED
                )

                cv2.putText(
                    image_np, 
                    display_text, 
                    (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, 
                    (255, 255, 255),
                    thickness
                )

            cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), color, 2)
            
    return Image.fromarray(image_np)

# Загрузка моделей
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


frcnn_weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
frcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=frcnn_weights).to(device).eval()
frcnn_transforms = frcnn_weights.transforms()

fcos_weights = FCOS_ResNet50_FPN_Weights.DEFAULT
fcos_model = torchvision.models.detection.fcos_resnet50_fpn(weights=fcos_weights).to(device).eval()
fcos_transforms = fcos_weights.transforms()

yolov10_model = YOLO('yolov10m.pt').to(device)