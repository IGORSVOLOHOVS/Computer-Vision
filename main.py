import torch
import torchvision
from torchvision import transforms, models
from PIL import Image

# Загрузка данных
val_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,      # Указываем, что это НЕ тренировочная выборка
    download=True     # Разрешаем скачивание, если датасета нет
)

# Загрузка модели
weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
model = models.mobilenet_v2(weights=weights)

model.eval()

# Подготовка одного изображения
img_pil, label = val_dataset[0]

preprocess = weights.transforms()

# Применяем трансформации к нашему изображению
input_tensor = preprocess(img_pil)

# Модель ожидает на вход батч изображений. Наш тензор имеет форму [C, H, W].
# Мы должны добавить "батчевое" измерение, чтобы получить [1, C, H, W].
input_batch = input_tensor.unsqueeze(0) 

# Получение предсказания
# `torch.no_grad()` отключает расчёт градиентов, что экономит память и ускоряет инференс
with torch.no_grad():
    # Подаём подготовленный батч в модель. Модель вызывается как функция
    output = model(input_batch)

# `output` содержит "сырые" значения для 1000 классов ImageNet
# Преобразуем их в вероятности с помощью функции softmax
probabilities = torch.nn.functional.softmax(output[0], dim=0)


# Вывод результата
# Найдём топ-5 предсказанных классов и их вероятности
top5_prob, top5_catid = torch.topk(probabilities, 5)

# Получаем список названий классов из метаданных весов
categories = weights.meta["categories"] 

for i in range(top5_prob.size(0)):
    category_name = categories[top5_catid[i]]
    probability = top5_prob[i].item()
    print(f"  {i+1}. Класс: {category_name:<20} | Вероятность: {probability*100:.2f}%")


    py -3.10 -m venv clean_venv 
.\clean_venv\Scripts\activate  
# Для того, чтобы позже собрать mmdetection из исходников,
# нам понадобится более старая версия pip
python -m pip install pip==21.2.3