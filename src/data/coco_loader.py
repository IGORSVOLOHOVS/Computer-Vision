import os
import cv2
import numpy as np
from typing import List, Dict
from dataclasses import dataclass
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from torch.utils.data import Dataset
from pydantic import DirectoryPath, FilePath
from .data_utils import TypedDataset, SegmentationSample


class COCOLoader(TypedDataset[SegmentationSample]):
    def __init__(self, image_folder: DirectoryPath, coco_anno_file: FilePath) -> None:
        # Инициализируем класс COCO, его методами мы будем пользоваться для чтения разметки
        self._coco_anno_api: COCO = COCO(coco_anno_file)

        self._dataset_images_ids = self._coco_anno_api.getImgIds()
        self._image_folder = image_folder

        # В COCO информация о классе разметки хранится в объекте category.
        # У него есть ID и name. ID генерирует сам COCO, а name — это как раз то, что мы указывали при создании лейбла
        # В аннотациях на category обычно ссылаются через её ID. Чтобы нам было проще перейти к имени
        # заведём маппинг от ID к имени
        self._cat_id2class_name: Dict[int, str] = (
            self._create_cat_id_to_class_name_mapping()
        )

        # На основе маппинга от ID к имени сгенерируем обратный
        self._class_name2cat_id: Dict[str, int] = {
            class_name: cat_id for cat_id, class_name in self._cat_id2class_name.items()
        }

    def _create_cat_id_to_class_name_mapping(self) -> int:
        cat_id2class_name: Dict[int, str] = {}
        # Получаем список всех ID категорий методом getCatIds
        cat_ids = self._coco_anno_api.getCatIds()
        for cat_id in cat_ids:
            # Теперь зная ID, мы можем загрузить сам объект с информацией о категории
            # Это делается методом loadCats, он всегда возвращает список, так как обычно
            # используется с несколькими ID сразу. Но поскольку мы передали только один, то наш
            # список будет содержать один элемент. Сразу достаём его.
            category_info = self._coco_anno_api.loadCats(cat_id)[0]
            cat_id2class_name[cat_id] = category_info["name"]
        return cat_id2class_name

    def __len__(self) -> int:
        return len(self._dataset_images_ids)

    def __getitem__(self, index: int) -> SegmentationSample:
        # Получим информацию о разметке и изображении с помощью COCO api
        sample_img_id: int = self._dataset_images_ids[index]
        sample_image_info: Dict = self._coco_anno_api.loadImgs(sample_img_id)[0]
        sample_anno_ids: List[int] = self._coco_anno_api.getAnnIds(imgIds=sample_img_id)
        sample_annotations = self._coco_anno_api.loadAnns(sample_anno_ids)

        # Считаем иображение
        img_path = os.path.join(self._image_folder, sample_image_info["file_name"])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        segmentation_class_map = np.zeros(
            (sample_image_info["height"], sample_image_info["width"]), dtype=np.uint8
        )

        # Подготовим маску классов в 2 этапа
        # 1. Добавим все классы, которые не являются бекграундом
        background_id: int = self._class_name2cat_id["background"]
        for anno in sample_annotations:
            if anno["category_id"] == background_id:
                continue
            rle_repr: np.ndarray = coco_mask.frPyObjects(
                anno["segmentation"],
                h=sample_image_info["height"],
                w=sample_image_info["width"],
            )
            cur_mask = coco_mask.decode(rle_repr).squeeze()
            segmentation_class_map[cur_mask == 1] = anno["category_id"]

        # 2. Занулим бекграунд
        for anno in sample_annotations:
            if anno["category_id"] != background_id:
                continue
            rle_repr: np.ndarray = coco_mask.frPyObjects(
                anno["segmentation"],
                h=sample_image_info["height"],
                w=sample_image_info["width"],
            )
            cur_mask = coco_mask.decode(rle_repr).squeeze()
            segmentation_class_map[cur_mask == 1] = 0

        output_sample = SegmentationSample(img=img, classes_map=segmentation_class_map)
        return output_sample