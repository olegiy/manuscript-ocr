"""
Вспомогательные loss функции для обучения U-Net на контрастность и бинаризацию.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


class UNetAuxiliaryLoss(nn.Module):
    """
    Вспомогательный loss для U-Net, поощряющий контрастность и бинаризацию.

    Комбинирует несколько компонентов:
    1. Contrast loss - максимизирует дисперсию (разницу фон/текст)
    2. Binarization loss - подталкивает пиксели к 0 или 1
    3. Total Variation loss - сглаживает однородные области
    4. Otsu guidance loss - приближает к Otsu бинаризации (weak learning)

    Parameters
    ----------
    contrast_weight : float
        Вес contrast loss (по умолчанию 1.0)
    binarization_weight : float
        Вес binarization loss (по умолчанию 0.5)
    tv_weight : float
        Вес total variation loss (по умолчанию 0.01)
    otsu_weight : float
        Вес Otsu guidance loss (по умолчанию 0.3)
    """

    def __init__(
        self,
        contrast_weight: float = 1.0,
        binarization_weight: float = 0.5,
        tv_weight: float = 0.01,
        otsu_weight: float = 0.3,
    ):
        super().__init__()
        self.contrast_weight = contrast_weight
        self.binarization_weight = binarization_weight
        self.tv_weight = tv_weight
        self.otsu_weight = otsu_weight

    def contrast_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Максимизирует дисперсию (контрастность) изображения.
        Высокая дисперсия = чёткая граница между фоном и текстом.

        Возвращает отрицательную дисперсию (минимизируем -> максимизируем дисперсию).
        """
        # Считаем дисперсию по batch
        mean = x.mean(dim=[2, 3], keepdim=True)
        variance = ((x - mean) ** 2).mean(dim=[2, 3])

        # Минимизируем отрицательную дисперсию = максимизируем дисперсию
        return -variance.mean()

    def binarization_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Подталкивает пиксели к 0 (фон) или 1 (текст).

        Использует функцию L(p) = 4 * p * (1 - p), которая:
        - минимальна при p=0 и p=1
        - максимальна при p=0.5
        """
        # x в диапазоне [0, 1] после sigmoid
        # 4*p*(1-p) = 0 при p=0 или p=1, максимум при p=0.5
        bin_loss = 4 * x * (1 - x)
        return bin_loss.mean()

    def total_variation_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Сглаживает изображение, уменьшая шум в однородных областях.

        Считает сумму абсолютных разностей соседних пикселей.
        """
        # Разности по горизонтали
        diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        # Разности по вертикали
        diff_v = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])

        tv = diff_h.mean() + diff_v.mean()
        return tv

    def otsu_guidance_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Weak learning: направляет U-Net к Otsu-подобной бинаризации.

        Применяет Otsu на CPU для каждого изображения в batch,
        затем считает MSE между выходом U-Net и Otsu маской.

        Parameters
        ----------
        x : torch.Tensor
            Выход U-Net [B, C, H, W] в диапазоне [0, 1]

        Returns
        -------
        torch.Tensor
            MSE между U-Net выходом и Otsu бинаризацией
        """
        device = x.device
        batch_size = x.size(0)

        otsu_masks = []

        # Применяем Otsu к каждому изображению
        for i in range(batch_size):
            # Берем одноканальное изображение [H, W]
            img = x[i, 0].detach().cpu().numpy()

            # Переводим в uint8 для Otsu
            img_uint8 = (img * 255).astype(np.uint8)

            # Применяем Otsu бинаризацию
            _, otsu_binary = cv2.threshold(
                img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # Нормализуем обратно в [0, 1]
            otsu_normalized = otsu_binary.astype(np.float32) / 255.0

            otsu_masks.append(otsu_normalized)

        # Собираем в batch tensor
        otsu_batch = torch.from_numpy(np.stack(otsu_masks)).unsqueeze(1).to(device)

        # MSE между U-Net выходом и Otsu маской
        mse = F.mse_loss(x, otsu_batch)

        return mse

    def forward(self, unet_output: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет комбинированный loss для U-Net output.

        Parameters
        ----------
        unet_output : torch.Tensor
            Выход U-Net [B, C, H, W] в диапазоне [0, 1]

        Returns
        -------
        torch.Tensor
            Скалярный loss
        """
        loss = 0.0

        if self.contrast_weight > 0:
            loss += self.contrast_weight * self.contrast_loss(unet_output)

        if self.binarization_weight > 0:
            loss += self.binarization_weight * self.binarization_loss(unet_output)

        if self.tv_weight > 0:
            loss += self.tv_weight * self.total_variation_loss(unet_output)

        if self.otsu_weight > 0:
            loss += self.otsu_weight * self.otsu_guidance_loss(unet_output)

        return loss


class CombinedLoss(nn.Module):
    """
    Комбинированный loss: основной recognition loss + вспомогательный U-Net loss.

    Parameters
    ----------
    recognition_criterion : nn.Module
        Основной loss для распознавания (обычно CrossEntropyLoss)
    unet_loss : UNetAuxiliaryLoss, optional
        Вспомогательный loss для U-Net
    unet_loss_weight : float
        Вес U-Net loss относительно recognition loss (по умолчанию 0.1)
    """

    def __init__(
        self,
        recognition_criterion: nn.Module,
        unet_loss: UNetAuxiliaryLoss = None,
        unet_loss_weight: float = 0.1,
    ):
        super().__init__()
        self.recognition_criterion = recognition_criterion
        self.unet_loss = unet_loss
        self.unet_loss_weight = unet_loss_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        unet_output: torch.Tensor = None,
    ) -> tuple:
        """
        Вычисляет комбинированный loss.

        Parameters
        ----------
        logits : torch.Tensor
            Логиты модели [B*T, V] (уже flatten)
        targets : torch.Tensor
            Ground truth токены [B*T] (уже flatten)
        unet_output : torch.Tensor, optional
            Выход U-Net [B, C, H, W]

        Returns
        -------
        tuple
            (total_loss, recognition_loss, unet_loss)
        """
        # Основной recognition loss
        rec_loss = self.recognition_criterion(logits, targets)

        total_loss = rec_loss
        unet_loss_val = torch.tensor(0.0, device=logits.device)

        # Добавляем U-Net loss если доступен
        if self.unet_loss is not None and unet_output is not None:
            unet_loss_val = self.unet_loss(unet_output)
            total_loss = rec_loss + self.unet_loss_weight * unet_loss_val

        return total_loss, rec_loss, unet_loss_val
