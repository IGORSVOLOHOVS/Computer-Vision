import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """
        Инициализация свёрточного блока.
        Args:
            c1 (int): количество входных каналов.
            c2 (int): количество выходных каналов.
            k (int): размер ядра свертки (kernel size).
            s (int): шаг свёртки (stride).
            p (int): отступ (padding). Если None, вычисляется автоматически.
            g (int): количество групп (для depth-wise свёрток).
            act (bool): флаг, указывающий, нужно ли применять функцию активации.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

def autopad(k, p=None):
    # Автоматический расчёт padding для "same" свёртки
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Bottleneck(nn.Module):
    # Стандартный Bottleneck блок из ResNet
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """
        Инициализация Bottleneck.
        Args:
            c1 (int): количество входных каналов.
            c2 (int): количество выходных каналов.
            shortcut (bool): флаг, указывающий, нужно ли использовать residual connection.
            g (int): количество групп.
            e (float): коэффициент расширения каналов.
        """
        super().__init__()
        c_ = int(c2 * e)  # скрытые каналы
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        # Если self.add=True, складываем вход с выходом (residual connection)
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    

class C3(nn.Module):
    # Блок C3 из YOLOv5 (CSP Bottleneck with 3 convolutions)
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Инициализация C3.
        Args:
            c1 (int): количество входных каналов.
            c2 (int): количество выходных каналов.
            n (int): количество Bottleneck блоков.
            shortcut (bool): флаг для residual connection в Bottleneck.
            g (int): количество групп.
            e (float): коэффициент расширения.
        """
        super().__init__()
        c_ = int(c2 * e)  # скрытые каналы
        self.cv1 = Conv(c1, c_, 1)
        self.cv2 = Conv(c1, c_, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # финальная свертка
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        # Разделение на два пути и конкатенация
        # Путь 1: x -> cv1 -> m
        # Путь 2: x -> cv2
        # Результат: concat(путь1, путь2) -> cv3
        part1 = self.m(self.cv1(x))
        part2 = self.cv2(x)
        return self.cv3(torch.cat((part1, part2), dim=1))
    
class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """
        Инициализация C2f.
        Args:
            c1 (int): количество входных каналов.
            c2 (int): количество выходных каналов.
            n (int): количество Bottleneck блоков.
            shortcut (bool): флаг для residual connection в Bottleneck.
            g (int): количество групп.
            e (float): коэффициент расширения.
        """
        super().__init__()
        self.c_ = int(c2 * e)  # скрытые каналы
        self.cv1 = Conv(c1, 2 * self.c_, 1, 1) # Увеличиваем каналы для последующего разделения
        self.cv2 = Conv((2 + n) * self.c_, c2, 1)  # Финальная свёртка
        self.m = nn.ModuleList(Bottleneck(self.c_, self.c_, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x):
        x = self.cv1(x)
        y1, y2 = x.split((self.c_, self.c_), 1)

        outputs = [y1, y2]
        current_y = y2
        for bottleneck_module in self.m:
            current_y = bottleneck_module(current_y)
            outputs.append(current_y)
        concatenated_output = torch.cat(outputs, 1)
        return self.cv2(concatenated_output)
    

# --- Твой код здесь ---
# (Определения autopad и Conv)


def example_usage():
    # 1. Создаем входные данные: 1 изображение, 3 канала (RGB), размер 640x640
    input_tensor = torch.randn(1, 3, 640, 640)

    # 2. Инициализируем блок: 
    # c1=3 (вход), c2=16 (выход), k=3 (ядро 3x3), s=2 (шаг 2 для уменьшения размера)
    conv_block = Conv(c1=3, c2=16, k=3, s=2)

    # 3. Прогоняем данные через блок
    output = conv_block(input_tensor)

    # print(input_tensor)

    print(f"Входной размер: {input_tensor.shape}")
    # Благодаря s=2, размер уменьшится вдвое: 640 -> 320
    print(f"Выходной размер: {output.shape}") 

    # Создаем "картинку": 1 штука, 128 каналов, размер 20x20
    input_tensor = torch.randn(1, 128, 20, 20)

    # Создаем блок: на входе 128, на выходе 128, сжатие до 64 (e=0.5)
    model = Bottleneck(c1=128, c2=128, shortcut=True)

    print("--- Старт прохода через Bottleneck ---")
    output = model(input_tensor)
    print(f"Итог: {output.shape}")

    btln = C2f(c1=3,c2=64, n=1)
    output = btln(input_tensor)
    print(f"Входной размер: {input_tensor.shape}")
    # Благодаря s=2, размер уменьшится вдвое: 640 -> 320
    print(f"Выходной размер C2f: {output.shape}") 

example_usage()