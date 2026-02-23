import cv2
import matplotlib.pyplot as plt

# [x_min, y_min, x_max, y_max]
bounding_box = [650, 400, 1170, 750]
class_name = "car"
confidence_score = 0.95


# Загрузите изображение с помощью OpenCV
image = cv2.imread('test_image.jpg')
# OpenCV загружает изображения в формате BGR. Для Matplotlib нужно его конвертировать в RGB.
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Получите координаты из массива bounding_box
x_min, y_min, x_max, y_max = bounding_box

# Нарисуйте прямоугольник на изображении
# Используйте функцию cv2.rectangle()
cv2.rectangle(image_rgb, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

# Добавьте текст (название класса и уверенность)
# Используйте функцию cv2.putText()
text = f"{class_name}: {confidence_score:.2f}"
cv2.putText(image_rgb, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


# Отобразите результат с помощью Matplotlib
plt.imshow(image_rgb)
plt.axis('off') # Отключить оси
plt.title("Detector Output")
plt.show()
