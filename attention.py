import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MHSA2D(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, "Число каналов должно быть кратно числу голов"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V за один проход (1×1 conv)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        x: (B, C, H, W)
        return: (B, C, H, W)
        """
        B, C, H, W = x.shape
        N = H * W

        # Получаем Q, K, V
        qkv = self.qkv(x)  # (B, 3C, H, W)
        qkv = rearrange(qkv, 'b (three h d) h0 w0 -> three b h (h0 w0) d', three=3, h=self.num_heads, d=self.head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # каждая: (B, num_heads, N, head_dim)

        # Attention: (B, heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Применяем внимание к значениям
        out = attn @ v  # (B, heads, N, head_dim)
        out = rearrange(out, 'b h n d -> b (h d) n')  # (B, C, N)
        out = out.view(B, C, H, W)  # (B, C, H, W)

        # Финальная проекция
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

import torch
import torch.nn as nn

# Импортируем ваш MHSA2D


def test_output_shape():
    """Проверяет, что выход имеет ту же форму, что и вход"""
    x = torch.randn(2, 64, 16, 16)
    attn = MHSA2D(dim=64, num_heads=4)
    y = attn(x)
    assert y.shape == x.shape, f"Неверная форма: {y.shape}, ожидалось {x.shape}"


def test_heads_divisible():
    """Должно падать, если число каналов не делится на число голов"""
    try:
        MHSA2D(dim=63, num_heads=4)
    except AssertionError:
        pass
    else:
        raise AssertionError("Должна быть ошибка при несовместимых dim и num_heads")


def test_forward_runs():
    """Проверяет, что forward выполняется без ошибок"""
    x = torch.randn(1, 32, 8, 8)
    attn = MHSA2D(dim=32, num_heads=4)
    y = attn(x)
    assert isinstance(y, torch.Tensor), "Выход должен быть тензором"
    assert not torch.isnan(y).any(), "Выход содержит NaN-значения"


def test_gradients():
    """Проверяет, что градиенты проходят"""
    x = torch.randn(2, 32, 8, 8, requires_grad=True)
    attn = MHSA2D(dim=32, num_heads=4)
    y = attn(x).mean()
    y.backward()
    assert x.grad is not None, "Градиенты не прошли назад!"


def test_determinism():
    """Проверяет, что при одинаковых весах результат совпадает"""
    torch.manual_seed(42)
    attn1 = MHSA2D(dim=32, num_heads=4)
    attn2 = MHSA2D(dim=32, num_heads=4)
    attn2.load_state_dict(attn1.state_dict())

    x = torch.randn(1, 32, 8, 8)
    y1 = attn1(x)
    y2 = attn2(x)
    assert torch.allclose(y1, y2, atol=1e-6), "Результаты должны быть одинаковыми при одинаковых весах"


if __name__ == '__main__':
    test_output_shape()
    test_heads_divisible()
    test_forward_runs()
    test_gradients()
    test_determinism()
    print("Все тесты пройдены успешно!")