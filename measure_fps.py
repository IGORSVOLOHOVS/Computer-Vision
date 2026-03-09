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

def measure_fps(models_dict, transforms_dict, image_pil, num_runs=100, device='cpu'):
    """
    Универсальная функция для измерения и сравнения FPS нескольких моделей.

    Args:
        models_dict (dict): Словарь, где ключ - название модели, значение - сама модель.
        transforms_dict (dict): Словарь, где ключ - название модели, значение - функция трансформации.
        image_pil (Image.Image): Тестовое изображение в формате PIL.
        num_runs (int): Количество запусков для усреднения.
        device (torch.device): Устройство для вычислений.
    """
    results = {}
    
    for model_name, model in models_dict.items():
        # Обрабатываем разные типы моделей
        if "YOLO" in model_name:
            model.predict(image_pil, verbose=False)
        else: # Для моделей torchvision
            transform = transforms_dict[model_name]
            input_tensor = transform(image_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                model(input_tensor)

    for model_name, model in models_dict.items():
        start_time = time.time()

        # Снова обрабатываем разные типы моделей
        if "YOLO" in model_name:
            for _ in range(num_runs):
                model.predict(image_pil, verbose=False)
        else: # Для моделей torchvision
            transform = transforms_dict[model_name]
            input_tensor = transform(image_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                for _ in range(num_runs):
                    model(input_tensor)

        end_time = time.time()
        fps = num_runs / (end_time - start_time)
        results[model_name] = fps

    # Вывод результатов
    print("\n" + "="*20 + " Результаты измерения End-to-End FPS " + "="*20)
    for name, fps in sorted(results.items(), key=lambda item: item[1], reverse=True):
        print(f"{name:<15}: {fps:.2f} FPS")

    return results

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


frcnn_weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
frcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=frcnn_weights).to(device).eval()
frcnn_transforms = frcnn_weights.transforms()

fcos_weights = FCOS_ResNet50_FPN_Weights.DEFAULT
fcos_model = torchvision.models.detection.fcos_resnet50_fpn(weights=fcos_weights).to(device).eval()
fcos_transforms = fcos_weights.transforms()

yolov10_model = YOLO('yolov10m.pt').to(device)
test_scenes = {
    "scene_1_occlusion": "https://farm8.staticflickr.com/7079/6894809750_036c532d00_z.jpg",
    "scene_2_small_objects": "https://farm9.staticflickr.com/8233/8374460136_8d23e517f4_z.jpg",
    "scene_3_standard": "https://farm8.staticflickr.com/7113/8151550021_4388fe2b16_z.jpg",
}

def main():
    models_to_test = {
        "Faster R-CNN": frcnn_model,
        "FCOS": fcos_model,
        "YOLOv10m": yolov10_model,
    }

    # Собираем трансформации в один словарь (для YOLO ставим None)
    transforms_to_test = {
        "Faster R-CNN": frcnn_transforms,
        "FCOS": fcos_transforms,
        "YOLOv10m": None,
    }

    # Загружаем стандартное изображение для теста
    url_standard = test_scenes["scene_3_standard"]
    response = requests.get(url_standard)
    img_pil_standard = Image.open(BytesIO(response.content)).convert("RGB")

    # Запуск измерения
    fps_results = measure_fps(
        models_dict=models_to_test,
        transforms_dict=transforms_to_test,
        image_pil=img_pil_standard,
        num_runs=2,
        device=device
    )


if __name__ == "__main__":
    main()