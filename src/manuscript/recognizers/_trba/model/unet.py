"""
Компактный U-Net для предобработки изображений (очистка фона).

Архитектура: легкий encoder-decoder с skip connections.
Используется перед основным распознавателем для улучшения качества входных изображений.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Два последовательных слоя конволюции с BatchNorm и ReLU."""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling блок: MaxPool + DoubleConv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling блок: ConvTranspose + DoubleConv."""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Padding для совпадения размеров (если есть различия из-за нечетных размеров)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # Concatenate по channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class CompactUNet(nn.Module):
    """
    Компактный U-Net для создания attention mask.

    Задача: формирует бинарную маску (через sigmoid), которая выделяет текст.
    Обучается end-to-end через основной recognition loss - сам учится фильтровать полезную информацию.

    Parameters
    ----------
    in_channels : int
        Число входных каналов (3 для RGB, 1 для grayscale).
    features : list of int, optional
        Число фич на каждом уровне. По умолчанию [32, 64, 128] (увеличено для более тонкой работы).
    bilinear : bool, optional
        Использовать билинейную интерполяцию вместо ConvTranspose2d.
        По умолчанию False (ConvTranspose2d).
    hard_binarize : bool, optional
        Применять жесткую бинаризацию (0 или 1) через straight-through estimator.
        По умолчанию True для четких границ.
    threshold : float, optional
        Порог для жесткой бинаризации. По умолчанию 0.5.
    """

    def __init__(
        self,
        in_channels=3,
        features=None,
        bilinear=False,
        hard_binarize=True,
        threshold=0.5,
    ):
        super().__init__()

        if features is None:
            features = [32, 64, 128]  # Увеличенная версия для более тонкой работы

        self.in_channels = in_channels
        self.hard_binarize = hard_binarize
        self.threshold = threshold

        # Входной слой
        self.inc = DoubleConv(in_channels, features[0])

        # Encoder (downsampling)
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])

        # Bottleneck
        factor = 2 if bilinear else 1
        self.down3 = Down(features[2], features[2] * 2 // factor)

        # Decoder (upsampling)
        self.up1 = Up(features[2] * 2, features[2] // factor, bilinear)
        self.up2 = Up(features[2], features[1] // factor, bilinear)
        self.up3 = Up(features[1], features[0], bilinear)

        # Выходной слой: создаем одноканальную маску
        self.outc = nn.Conv2d(features[0], 1, kernel_size=1)

        # Sigmoid для мягкой бинарной маски [0, 1]
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Входное изображение [B, C, H, W].

        Returns
        -------
        torch.Tensor
            Бинарная маска [B, 1, H, W].
            - При hard_binarize=True: строго 0 или 1
            - При hard_binarize=False: soft [0, 1]
        """
        # Encoder
        x1 = self.inc(x)  # [B, features[0], H, W]
        x2 = self.down1(x1)  # [B, features[1], H/2, W/2]
        x3 = self.down2(x2)  # [B, features[2], H/4, W/4]
        x4 = self.down3(x3)  # [B, features[2]*2, H/8, W/8]

        # Decoder with skip connections
        x_up = self.up1(x4, x3)  # [B, features[2], H/4, W/4]
        x_up = self.up2(x_up, x2)  # [B, features[1], H/2, W/2]
        x_up = self.up3(x_up, x1)  # [B, features[0], H, W]

        # Output: одноканальная маска
        mask = self.outc(x_up)  # [B, 1, H, W]
        mask = self.sigmoid(mask)  # [0, 1]

        # Жесткая бинаризация (straight-through estimator)
        if self.hard_binarize:
            # Forward: hard threshold (0 или 1)
            # Backward: градиенты идут через sigmoid (differentiable)
            mask_hard = (mask > self.threshold).float()
            # Straight-through: используем hard на forward, но градиенты от soft
            mask = mask_hard + mask - mask.detach()

        return mask


class TinyUNet(nn.Module):
    """
    Очень компактная версия U-Net для быстрой предобработки.

    Всего 2 уровня encoder/decoder. Быстрее, но менее выразительная.
    Подходит для случаев, когда скорость критична.
    """

    def __init__(self, in_channels=3, out_channels=3, residual=True):
        super().__init__()

        self.residual = residual

        # Encoder
        self.enc1 = DoubleConv(in_channels, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(32, 64)

        # Decoder
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)  # 64 = 32 (skip) + 32 (up)

        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(32, 16)  # 32 = 16 (skip) + 16 (up)

        # Output
        self.outc = nn.Conv2d(16, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Save input for residual
        x_input = x

        # Encoder
        e1 = self.enc1(x)
        x = self.pool1(e1)

        e2 = self.enc2(x)
        x = self.pool2(e2)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.up1(x)
        x = torch.cat([x, e2], dim=1)
        x = self.dec1(x)

        x = self.up2(x)
        # Padding если нужно
        diffY = e1.size()[2] - x.size()[2]
        diffX = e1.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x, e1], dim=1)
        x = self.dec2(x)

        # Output
        x = self.outc(x)
        x = self.sigmoid(x)

        # Residual
        if self.residual:
            x = 0.7 * x + 0.3 * x_input

        return x


# Функция для создания U-Net по выбору
def create_unet(variant="compact", in_channels=3, out_channels=3, **kwargs):
    """
    Фабрика для создания U-Net модели.

    Parameters
    ----------
    variant : {'compact', 'tiny'}, optional
        Тип архитектуры. Default 'compact'.
    in_channels : int
        Число входных каналов.
    out_channels : int
        Число выходных каналов.
    **kwargs
        Дополнительные параметры для конкретной модели.

    Returns
    -------
    nn.Module
        U-Net модель.
    """
    if variant == "compact":
        return CompactUNet(in_channels, out_channels, **kwargs)
    elif variant == "tiny":
        return TinyUNet(in_channels, out_channels, **kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant}. Choose 'compact' or 'tiny'.")
