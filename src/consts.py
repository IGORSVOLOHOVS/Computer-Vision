import os

# Определяем путь до файла consts.py, используя переменную __file__
path_to_file = os.path.abspath(__file__)

# ипользуем dirname, получаем путь до родительской папки consts.py - то есть папки src
SRC_FOLDER = os.path.dirname(path_to_file)
# похожим образом получаем путь до корня проекта
PROJECT_ROOT = os.path.dirname(SRC_FOLDER)
# используя join, объединяем корневую папку проекта с именем "data", чтобы получить полный путь до папки с данными
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

# похожим образом получаем путь до корневой папки нашего датасета с носорогом
RHINO_DATASET_ROOT = os.path.join(DATA_ROOT, "rhino_dataset")
# Проверяем, что путь до папки c датасетом действительно существует
assert os.path.exists(RHINO_DATASET_ROOT)