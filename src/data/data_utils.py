from typing import TypeVar, Generic
from torch.utils.data import Dataset
from dataclasses import dataclass
import numpy as np


# Спецификация формата семпла для семантической сегментации
@dataclass
class SegmentationSample:
    img: np.ndarray
    classes_map: np.ndarray


SampleType = TypeVar("SampleType")


class TypedDataset(Dataset, Generic[SampleType]):
    """
    Используя Generic из библиотеки typing, мы определяем интерфейс датасета,
    который возвращает семпл определённого типа. Так мы сможем использовать более точную типизацию
    в сигнатурах функций и намм будет проще работать в IDE за счёт более точных подсказок
    Пример:
    dataset: TypedDataset[SegmentationSample] = load_dataset()
    """

    def __getitem__(self, index: int) -> SampleType:
        """Реализация __getitem__ со спецификацией типа семпла"""