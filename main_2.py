from transformers import AutoImageProcessor, AutoModelForImageClassification
from torch.nn import Linear
from rich.console import Console
c = Console()

from transformers import ViTForImageClassification, ViTImageProcessor
from torch import nn

# Выбор предобученной модели
model_name = "google/vit-base-patch16-224"
processor = ViTImageProcessor.from_pretrained(model_name)
# attn_implementation="eager" - реализация, для которой предусмотрена визуализация внимания
model = ViTForImageClassification.from_pretrained(
    model_name, attn_implementation="eager"
)

# Замена последнего слоя на подходящий для задачи бинарной классификации
model.classifier = nn.Linear(model.classifier.in_features, 2)

c.print(
    "Количество параметров:",
    sum([p.numel() for p in model.parameters() if p.requires_grad]),
)

from PIL import Image

image = Image.open('https://www.google.com/imgres?q=graphviz%20download&imgurl=https%3A%2F%2Fgraphviz.org%2FGallery%2Fdirected%2Fcluster.png&imgrefurl=https%3A%2F%2Fgraphviz.org%2F&docid=DaQcWbST_bFZ3M&tbnid=fX4HSEin-ZfKBM&vet=12ahUKEwjQvb2dxveTAxXpUFUIHYMeGs0QnPAOegQIGRAB..i&w=299&h=547&hcb=2&ved=2ahUKEwjQvb2dxveTAxXpUFUIHYMeGs0QnPAOegQIGRAB').convert('RGB')

def vit_transform(image):
    inputs = processor(images=image, return_tensors="pt")
    return inputs["pixel_values"].squeeze(0)
c.print(dir(processor))

# for par in model.parameters():
#     par.requires_grad = False

# for par in model.classifier.parameters():
#     par.requires_grad = True

# import torch 
# # Настройка гиперпараметров
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
# EPOCHS = 1
# best_vloss = 1e5

# # Код обучения
# def train_one_epoch(epoch_index):
#     running_loss = 0.
#     last_loss = 0.

#     for batch_index, data in enumerate(train_loader):
#         # Извлечение батча
#         inputs, labels = data
#         # Обнуление градиентов
#         optimizer.zero_grad()
#         # Прямое распространение
#         outputs = model(inputs)
#         # Подсчёт ошибки
#         loss = criterion(outputs.logits, labels)
#         # Обратное распространение
#         loss.backward()
#         # Обновление весов
#         optimizer.step()

#         # Суммирование ошибки за последние 20 батчей
#         running_loss += loss.item()
#         if batch_index % 20 == 19:
#             last_loss = running_loss / 20. # средняя ошибка за 20 батчей
#             print(f'Эпоха: {epoch_index}, батч: {batch_index}, ошибка {last_loss}')
#             running_loss = 0.

#     return last_loss

# for epoch in range(EPOCHS):
#     print(f'Эпоха {epoch}')

#     # Перевод модели в режим обучения
#     model.train(True)
#     # Эпоха обучения
#     avg_loss = train_one_epoch(epoch)

#     # Перевод модели в режим валидации
#     model.eval()
#     running_vloss = 0.0

#     # Валидация
#     with torch.no_grad():
#         for i, vdata in enumerate(val_loader):
#             vinputs, vlabels = vdata
#             voutputs = model(vinputs)
#             vloss = criterion(voutputs.logits, vlabels)
#             running_vloss += vloss

#     avg_vloss = running_vloss / (i + 1)

    
#     # Сохранение лучшей модели
#     if avg_vloss < best_vloss:
#         best_vloss = avg_vloss
#         model_path = f'classifier_{epoch}.pt'
#         torch.save(model.state_dict(), model_path)

#     print(f'В конце эпохи ошибка train {avg_loss}, ошибка val {avg_vloss}')





# c.print(model.parameters())
# from torch import tensor
# c.print(tensor)

# # c.print(dir(model))
# # c.print(model.named_modules)
# # c.print(model.named_parameters)
# # c.print(model.named_children)
# # c.print(model.named_buffers)
# from torchinfo import summary
# # c.print(model.state_dict().items())
# summary(model, input_size=(1, 3, 384, 384))
