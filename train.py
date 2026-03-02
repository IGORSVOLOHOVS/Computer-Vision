import os
import glob
import yaml
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_class_distribution(data_yaml_path: str):
    """
    Анализирует распределение классов в датасете YOLO и строит гистограмму.
    """
    try:
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
    except FileNotFoundError:
        return

    class_names = data_config.get('names')
    train_path_from_yaml = data_config.get('train')

    if not class_names or not train_path_from_yaml:
        return

    # Сконструируйте правильный путь к папке с разметкой для обучающей выборки.
    # Вспомните, что папка `labels` лежит на том же уровне, что и `images`.
    # Используйте os.path.dirname() и os.path.join() для надежности.
    project_root = os.path.dirname(data_yaml_path)
    train_images_path = os.path.join(project_root, train_path_from_yaml)
    train_labels_path = train_images_path.replace('images','labels')

    # Подсчёт количества объектов каждого класса
    class_counter = Counter()

    label_files = glob.glob(os.path.join(train_labels_path, '*.txt'))[:-1]

    if not label_files:
        return

    # Проходимся по всем файлам .txt
    for file_path in label_files:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            # Проходимся по всем детекциям
            for line in lines:
                # Из каждой строки извлеките ID класса. Это первое число в строке.
                class_id = line.split()[0]
                class_counter[class_id] += 1
    
    if not class_counter:
        return

    # Подготовьте данные для графика:
    # - Отсортируйте ID классов, чтобы график был упорядочен.
    # - Создайте список 'labels' с именами классов (используя class_names).
    # - Создайте список 'counts' с количеством объектов для каждого класса.
    sorted_class_ids = sorted(class_counter.keys())# ... (отсортируйте ключи из class_counter)
    labels = [class_names[int(i)] for i in sorted_class_ids]
    counts = [class_counter[i] for i in sorted_class_ids]  # ... (получите значения из class_counter в правильном порядке)

    # Отображение графика
    plt.figure(figsize=(12, 8))
    sns.barplot(x=labels, y=counts)
    plt.title('Распределение классов в обучающем датасете')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    plt.show()

if __name__ == '__main__':
    # Укажите путь к вашему файлу data.yaml.
    path_to_my_yaml = 'datasets/data.yaml'

    analyze_class_distribution(path_to_my_yaml)