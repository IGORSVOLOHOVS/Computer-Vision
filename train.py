import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A


def visualize(image, bboxes, category_ids, category_id_to_name):
    """Рисует рамки и подписи на изображении."""
    img_viz = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        
        # Денормализуем координаты из формата YOLO в пиксельные координаты
        img_h, img_w, _ = image.shape
        x_center, y_center, width, height = bbox
        x_min = int((x_center - width / 2) * img_w)
        y_min = int((y_center - height / 2) * img_h)
        x_max = int((x_center + width / 2) * img_w)
        y_max = int((y_center + height / 2) * img_h)
        
        # Рисуем рамку и подпись
        cv2.rectangle(img_viz, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)
        cv2.putText(img_viz, class_name, (x_min, y_min - 10), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9, color=(0, 255, 0), thickness=2)
        
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img_viz, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Определяем пайплайн
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=15, p=0.7),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.5),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
    A.GaussNoise(p=0.2)
], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))

# Готовим исходное изображение
image = np.ones((512, 512, 3), dtype=np.uint8) * 255 # Белый фон
cv2.rectangle(image, (100, 100), (400, 400), color=(255, 0, 0), thickness=-1) # Синий квадрат
cv2.circle(image, (150, 150), 30, color=(0, 0, 255), thickness=-1) # Красный круг

# Исходная разметка в формате YOLO
# [[class_id, x_center, y_center, width, height], ...]
initial_bboxes = [
    [0.488, 0.488, 0.586, 0.586], # Параметры для синего квадрата
    [0.293, 0.293, 0.117, 0.117]  # Параметры для красного круга
]
category_ids = [0, 1]
category_id_to_name = {0: 'square', 1: 'circle'}

# Исходное изображение
visualize(image, initial_bboxes, category_ids, category_id_to_name)


# Передаем в пайплайн и изображение, и его разметку
transformed = transform(image=image, bboxes=initial_bboxes, category_ids=category_ids)

# Извлекаем новые, измененные данные
transformed_image = transformed['image']
transformed_bboxes = transformed['bboxes']
transformed_category_ids = transformed['category_ids']

# Рисуем изображение после аугментаций
visualize(transformed_image, transformed_bboxes, transformed_category_ids, category_id_to_name)