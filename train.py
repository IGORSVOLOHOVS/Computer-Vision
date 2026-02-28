import yaml
import cv2
import os
import random
import matplotlib.pyplot as plt

def verify_and_visualize(data_yaml_path: str):
    """
    Читает data.yaml, выбирает случайное изображение из обучающей выборки
    и визуализирует его вместе с разметкой.
    """
    # Чтение и парсинг data.yaml
    try:
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
    except FileNotFoundError:
        return

    # Извлеките из `data_config` следующие переменные:
    # 1. `class_names` (из ключа 'names')
    # 2. `train_images_rel_path` (из ключа 'train')
    # 3. `dataset_root_path_rel` (из ключа 'path')
    class_names = data_config['names']
    train_images_rel_path = data_config['train']
    dataset_root_path_rel = data_config['path']

    # Построение абсолютных путей
    yaml_dir = os.path.dirname(data_yaml_path)
    dataset_root_abs_path = os.path.abspath(os.path.join(yaml_dir, dataset_root_path_rel))
    train_images_abs_path = os.path.abspath(os.path.join(dataset_root_abs_path, train_images_rel_path))
    
    # Сконструируйте путь к папке с разметкой (`train_labels_abs_path`), 
    # заменив 'images' на 'labels' в `train_images_abs_path`.
    train_labels_abs_path = train_images_abs_path.replace('images', 'labels')

    # Выбор случайного изображения и его разметки
    all_images = [f for f in os.listdir(train_images_abs_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not all_images:
        return
        
    random_image_name = random.choice(all_images)
    image_path = os.path.join(train_images_abs_path, random_image_name)

    # Сформируйте путь к соответствующему файлу разметки (`.txt`).
    # Имя файла должно совпадать, но расширение - другое.
    label_name = random_image_name.split('.')[0] + '.txt' # ... (например, 'my_image.txt') 
    label_path = os.path.join(train_labels_abs_path, label_name)

    # Визуализация
    image = cv2.imread(image_path)
    img_height, img_width, _ = image.shape
    
    if not os.path.exists(label_path):
        print("Для этого изображения нет файла разметки")
    else:
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                
                # Распарсите строку из `.txt` файла.
                # 1. Извлеките `class_id`, `x_center`, `y_center`, `width`, `height`.
                # 2. Не забудьте преобразовать строки в `int` и `float`.
                class_id = int(parts[0])
                x_center, y_center, width, height = [float(p) for p in parts[1:]]
                
                # Денормализация: Преобразуйте нормализованные координаты в пиксельные.
                box_w = int(width * img_width)
                box_h = int(height * img_height)
                x_min = int(x_center * img_width - box_w / 2)
                y_min = int(y_center * img_height - box_h / 2)
                
                # Рисуем рамку и подпись
                class_name = class_names[class_id]
                cv2.rectangle(image, (x_min, y_min), (x_min + box_w, y_min + box_h), (0, 255, 0), 2)
                cv2.putText(image, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Отображаем результат
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Верификация разметки: {random_image_name}")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # Укажите путь к вашему `data.yaml`, который вы создали в Части 1.
    path_to_my_yaml = 'config/data.yaml'
    
    verify_and_visualize(path_to_my_yaml)