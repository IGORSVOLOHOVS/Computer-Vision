import os
import numpy as np
from .data.coco_loader import COCOLoader
from .data.data_utils import SegmentationSample, TypedDataset
from .consts import RHINO_DATASET_ROOT
import matplotlib.pyplot as plt
import cv2


RHINO_DATASET_COLORMAP = np.array([[0, 0, 0], [255, 0, 0]], dtype=np.uint8)


def plot_mask_over_image(
    segmentation_sample: SegmentationSample, colormap: np.array
) -> np.ndarray:
    """Функция для отрисовки: накладывает маску на изображение"""
    orig_img = segmentation_sample.img
    class_mask = segmentation_sample.classes_map
    # Генерируем цветную картинку, в которой цвет каждого пикселя определяется
    # значением colormap
    colored_mask = colormap[class_mask]

    # накладываем маску на изображение с помощью функции addWeighted
    # наложение происходит как взвешенная сумма каналов.
    # Вес исходного изображения определяется параметром alpha
    viz = orig_img.copy()
    alpha = 0.6
    mask_over_image_viz = cv2.addWeighted(
        viz,
        alpha,
        colored_mask,
        1 - alpha,
        0,
    )
    return mask_over_image_viz


def load_rhino_dataset() -> TypedDataset[SegmentationSample]:
    """Функция для загрузки датасета с носорогом"""
    rhino_ds_img_root = os.path.join(RHINO_DATASET_ROOT, "images")
    rhino_ds_annotaion_file = os.path.join(
        RHINO_DATASET_ROOT, "annotation", "instances_default.json"
    )
    rhino_dataset: TypedDataset[SegmentationSample] = COCOLoader(
        image_folder=rhino_ds_img_root, coco_anno_file=rhino_ds_annotaion_file
    )
    return rhino_dataset


def demo_rhino_dataset_loading() -> None:
    ds = load_rhino_dataset()
    sample = ds[0]
    sample_viz = plot_mask_over_image(sample, RHINO_DATASET_COLORMAP)
    plt.imshow(sample_viz)
    plt.show()


if __name__ == "__main__":
    demo_rhino_dataset_loading()