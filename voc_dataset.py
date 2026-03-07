import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET
import torchvision.transforms.v2 as transforms

class SimplifiedVOCDataset(Dataset):
    """
    Класс для работы с датасетом, где данные уже разделены по папкам
    train, val, test, и в каждой папке лежат пары .jpg и .xml.
    """
    
    # Список классов для PASCAL VOC-подобных датасетов
    CLASSES = [
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
        "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
        "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]

    def __init__(self, root_dir, image_set='train', transforms=None):
        """
        Args:
            root_dir (str): Путь к основной папке датасета, содержащей train/, val/, test/.
            image_set (str): 'train', 'val' или 'test' для выбора нужной папки.
            transforms: Трансформации для изображений.
        """
        self.root_dir = root_dir
        self.transforms = transforms
        self.image_set = image_set
        
        # Словарь для маппинга классов в ID
        self.class_to_idx = {cls: i + 1 for i, cls in enumerate(self.CLASSES)}

        # Определяем путь к конкретной выборке (train, val или test)
        self.data_dir = os.path.join(root_dir, image_set)
        
        # Сканируем папку и получаем список ID всех изображений.
        # ID - это имя файла без расширения. Мы ищем
        # все .xml файлы и берём их имена.
        self.ids = sorted([
            os.path.splitext(fname)[0] for fname in os.listdir(self.data_dir) 
            if fname.endswith('.xml')
        ])

        # Проверяем, существует ли папка
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Папка для выборки '{image_set}' не найдена по пути: {self.data_dir}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        file_id = self.ids[idx]

        # Пути к файлам теперь строятся относительно папки data_dir
        img_path = os.path.join(self.data_dir, file_id + '.jpg')
        ann_path = os.path.join(self.data_dir, file_id + '.xml')

        # Загружаем изображение
        img = Image.open(img_path).convert("RGB")

        # Парсим XML-файл
        boxes, labels = self._parse_xml(ann_path)

        # Собираем словарь 'target'
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([idx])

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

    def _parse_xml(self, ann_path):
        """Вспомогательная функция для парсинга XML."""
        boxes = []
        labels = []
        tree = ET.parse(ann_path)
        root = tree.getroot()

        for member in root.findall('object'):
            if int(member.find('difficult').text) == 1:
                continue

            class_name = member.find('name').text
            if class_name not in self.class_to_idx:
                continue

            label = self.class_to_idx[class_name]
            bndbox = member.find('bndbox')
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        return torch.as_tensor(boxes, dtype=torch.float32), torch.as_tensor(labels, dtype=torch.int64)

if __name__ == "__main__":
    DATASET_ROOT_PATH = "choco"

    v2_transforms = None
        
    # Создаём TRAIN датасет, указывая image_set='train'
    train_dataset = SimplifiedVOCDataset(
        root_dir=DATASET_ROOT_PATH, 
        image_set='train', 
        transforms=v2_transforms
    )
        
    # Создаём VAL датасет, указывая image_set='val'
    val_dataset = SimplifiedVOCDataset(
        root_dir=DATASET_ROOT_PATH, 
        image_set='valid', 
        transforms=v2_transforms
    )
        
    print(f"Обучающая выборка ('train'): {len(train_dataset)} изображений.")
    print(f"Валидационная выборка ('val'): {len(val_dataset)} изображений.\n")
        
    # Проверяем, что всё работает
    if len(train_dataset) > 0:
        img, target = train_dataset[0] # Берем первый элемент
    
        print(f"ID изображения: {train_dataset.ids[0]}")
        print(f"Найдено объектов: {len(target['boxes'])}")
        print("Координаты боксов:", target['boxes'])
        print("Метки классов:", target['labels'])
    else:
        print("Обучающая выборка пуста или не найдена.")

    dataset = PennFudanDataset('PennFudanPed')

    # Создаём DataLoader
    data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=2,          # Попросим 2 сэмпла за раз
        shuffle=True,          # Перемешивать данные в начале каждой эпохи
        num_workers=2,         # Использовать 2 фоновых процесса для загрузки данных
        collate_fn=collate_fn  # Указываем нашу функцию
    ) 