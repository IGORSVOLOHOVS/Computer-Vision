import cv2
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from pathlib import Path
from pydantic import DirectoryPath
from pprint import pprint
import matplotlib.pyplot as plt
from PIL import Image
import json
import numpy as np
import pickle
import os


def load_class_map_from_ADE_markup(file: str):
    """Фрагмент функции utils_ade20k.loadAde20K, отвечающий за загрузку маски классов"""

    # Определяем путь до картинки с маской на основе пути до картинки
    fileseg = file.replace(".jpg", "_seg.png")
    with Image.open(fileseg) as io:
        seg = np.array(io)

    # Сохраняем информацию о каждом RGB канале в отдельную переменную
    R = seg[:, :, 0]
    G = seg[:, :, 1]
    B = seg[:, :, 2]
    # Определяем индексы классов, по формуле от авторов датасета
    ObjectClassMasks = (R / 10).astype(np.int32) * 256 + (G.astype(np.int32))
    return ObjectClassMasks


# Формат, к которому мы приводим семплы из датасета ADE20K
@dataclass
class ADESegmentationSample:
    rgb_img: np.ndarray
    classes_map: np.ndarray


class ADE20K_Dataset(Dataset):
    def __init__(self, dataset_root: DirectoryPath) -> None:
        # Считываем пути до всех картинок, которые являются частью датасета, мы знаем, что они начинаются на ADE
        # и имеют расширение .jpg
        file_names = os.listdir(dataset_root)
        ADE_ds_image_names = filter(
            lambda f: f.endswith(".jpg") and f.startswith("ADE"), file_names
        )
        self.ADE_images_paths = [
            os.path.join(dataset_root, fname) for fname in ADE_ds_image_names
        ]

    def __len__(self) -> int:
        return len(self.ADE_images_paths)

    def __getitem__(self, index) -> ADESegmentationSample:
        # Считываем изображение в форматe RGB
        target_img_path = self.ADE_images_paths[index]
        img = cv2.imread(target_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Считываем маску сегментации
        segmentaion_msk = load_class_map_from_ADE_markup(target_img_path)

        # Пакуем наши данные в специальный формат
        output_sample: ADESegmentationSample = ADESegmentationSample(
            img, segmentaion_msk
        )
        return output_sample

def plot_ADE_sample(sample: ADESegmentationSample) -> None:
    fig, (image_ax, segmentation_map_ax) = plt.subplots(1, 2, figsize=(12, 6))
    image_ax.imshow(sample.rgb_img)
    image_ax.set_title("Исходное изображение")

    segmentation_map_ax.imshow(sample.classes_map, cmap="plasma")
    segmentation_map_ax.set_title("Маска классов")

    unique_classes = np.unique(sample.classes_map)
    max_class: int = np.max(unique_classes)
    for i in unique_classes:
        if i == 0:
            continue
        # Обратите внимание, как мы определяем цвет для элемента, который генерирует легенду
        # Наша маска классов содержит числа, которые выходят за границы int8, и чтобы отрисовать цвета
        # Ранее мы использовали палитру cmap=plasma
        # 
        # Теперь, чтобы нарисовать элемент таким же цветом, мы должны взять его из палитры напрямую
        # Для этого мы нормализуем цвет максимальным значением 
        class_color = plt.cm.plasma(i / max_class)
        segmentation_map_ax.scatter(
            0,
            0,
            c=class_color,
            label=ADE_CLASS_MAP[i - 1],
        )

    segmentation_map_ax.legend(loc="upper left")

    # Показываем результат
    fig.tight_layout()
    plt.show()


def main():
    script_folder = os.path.dirname(__file__)
    ade20k_root = os.path.join(os.path.dirname(script_folder), "ADE20K_dataset")

    ds_root = os.path.join(
        ade20k_root,
        "ADE20K_2017_05_30_consistency",
        "images",
        "consistencyanalysis",
        "original_ade20k",
    )
    ds = ADE20K_Dataset(ds_root)
    sample = ds[10]
    plot_ADE_sample(sample)