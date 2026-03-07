import torch
import torchvision
from torchvision import transforms, models
from PIL import Image


val_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,      
    download=True    # Разрешите скачивание (True)
)


weights = models.MobileNet_V2_Weights.IMAGENET1K_V2
model = models.mobilenet_v2(weights) # Загрузите модель models.mobilenet_v2, передав ей веса

model.eval()


# Получите (изображение PIL, метку) первого примера из датасета
img_pil, label = val_dataset[0]

# Получите пайплайн трансформаций, который соответсвует предобученным весам модели
preprocess = weights.transforms()

# Применяем трансформации и добавляем батчевое измерение (batch dimension)
input_tensor = preprocess(img_pil)
input_batch = input_tensor.unsqueeze(0) # -> [1, 3, H, W]


with torch.no_grad():
    output = model(input_batch) # Ваш код здесь. Модель вызывается как функция

# 'output' содержит сырые значения (логиты) для 1000 классов ImageNet
# Преобразуем их в вероятности
probabilities = torch.nn.functional.softmax(output[0], dim=0)


# Найдём топ-5 предсказанных классов
top5_prob, top5_catid = torch.topk(probabilities, 5)
categories = weights.meta["categories"] # Получаем названия классов из метаданных весов


for i in range(top5_prob.size(0)):
    category_name = categories[top5_catid[i]]
    probability = top5_prob[i]
    print(f"  {i+1}. Класс: {category_name:<20} | Вероятность: {probability*100:.2f}%")