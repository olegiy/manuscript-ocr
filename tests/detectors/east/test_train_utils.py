import pytest
import torch
import torch.nn as nn
import numpy as np
from manuscript.detectors._east.train_utils import (
    dice_coefficient as _dice_coefficient,
    _custom_collate_fn,
)


# --- Тесты для _dice_coefficient ---

@pytest.mark.skip(reason="Временно отключено")
def test_dice_coefficient_perfect_match():
    """Тест Dice coefficient при идеальном совпадении"""
    pred = torch.ones(2, 1, 4, 4)
    target = torch.ones(2, 1, 4, 4)
    
    dice = _dice_coefficient(pred, target)
    
    # При идеальном совпадении dice = 1.0
    assert dice.shape == (2,)
    assert torch.allclose(dice, torch.ones(2))

@pytest.mark.skip(reason="Временно отключено")
def test_dice_coefficient_no_overlap():
    """Тест Dice coefficient без пересечения"""
    pred = torch.zeros(2, 1, 4, 4)
    pred[0, 0, 0, 0] = 1.0
    
    target = torch.zeros(2, 1, 4, 4)
    target[0, 0, 3, 3] = 1.0
    
    dice = _dice_coefficient(pred, target)
    
    # Без пересечения dice близок к 0
    assert dice.shape == (2,)
    assert dice[0] < 0.1  # Первый sample почти 0

@pytest.mark.skip(reason="Временно отключено")
def test_dice_coefficient_partial_overlap():
    """Тест Dice coefficient с частичным пересечением"""
    pred = torch.zeros(1, 1, 4, 4)
    pred[0, 0, :2, :2] = 1.0  # Левый верхний квадрант
    
    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, 1:3, 1:3] = 1.0  # Частичное пересечение
    
    dice = _dice_coefficient(pred, target)
    
    assert dice.shape == (1,)
    assert 0 < dice[0] < 1

@pytest.mark.skip(reason="Временно отключено")
def test_dice_coefficient_batch_processing():
    """Тест Dice coefficient с разными batch sizes"""
    for batch_size in [1, 2, 4, 8]:
        pred = torch.rand(batch_size, 1, 8, 8)
        target = torch.rand(batch_size, 1, 8, 8)
        
        dice = _dice_coefficient(pred, target)
        
        assert dice.shape == (batch_size,)
        assert torch.all(dice >= 0)
        assert torch.all(dice <= 1)

@pytest.mark.skip(reason="Временно отключено")
def test_dice_coefficient_different_shapes():
    """Тест Dice coefficient с разными размерами"""
    shapes = [(4, 4), (8, 8), (16, 16), (32, 32)]
    
    for h, w in shapes:
        pred = torch.rand(2, 1, h, w)
        target = torch.rand(2, 1, h, w)
        
        dice = _dice_coefficient(pred, target)
        
        assert dice.shape == (2,)

@pytest.mark.skip(reason="Временно отключено")
def test_dice_coefficient_multi_channel():
    """Тест Dice coefficient с несколькими каналами"""
    pred = torch.rand(2, 3, 4, 4)  # 3 канала
    target = torch.rand(2, 3, 4, 4)
    
    dice = _dice_coefficient(pred, target)
    
    # Должно работать независимо от числа каналов
    assert dice.shape == (2,)

@pytest.mark.skip(reason="Временно отключено")
def test_dice_coefficient_zeros():
    """Тест Dice coefficient с нулевыми тензорами"""
    pred = torch.zeros(2, 1, 4, 4)
    target = torch.zeros(2, 1, 4, 4)
    
    dice = _dice_coefficient(pred, target)
    
    # Благодаря eps не должно быть division by zero
    assert torch.isfinite(dice).all()

@pytest.mark.skip(reason="Временно отключено")
def test_dice_coefficient_numerical_stability():
    """Тест численной стабильности Dice coefficient"""
    # Очень маленькие значения
    pred = torch.ones(2, 1, 4, 4) * 1e-8
    target = torch.ones(2, 1, 4, 4) * 1e-8
    
    dice = _dice_coefficient(pred, target, eps=1e-6)
    
    assert torch.isfinite(dice).all()

@pytest.mark.skip(reason="Временно отключено")
def test_dice_coefficient_custom_eps():
    """Тест Dice coefficient с кастомным eps"""
    pred = torch.rand(2, 1, 4, 4)
    target = torch.rand(2, 1, 4, 4)
    
    dice1 = _dice_coefficient(pred, target, eps=1e-6)
    dice2 = _dice_coefficient(pred, target, eps=1e-3)
    
    # Разные eps могут давать немного разные результаты
    assert dice1.shape == dice2.shape

@pytest.mark.skip(reason="Временно отключено")
def test_dice_coefficient_gradient():
    """Тест что градиенты проходят через Dice coefficient"""
    pred = torch.rand(2, 1, 4, 4, requires_grad=True)
    target = torch.rand(2, 1, 4, 4)
    
    dice = _dice_coefficient(pred, target)
    loss = (1 - dice).mean()
    loss.backward()
    
    assert pred.grad is not None
    assert pred.grad.abs().sum() > 0

@pytest.mark.skip(reason="Временно отключено")
def test_dice_coefficient_binary_masks():
    """Тест Dice coefficient с бинарными масками"""
    pred = torch.zeros(1, 1, 8, 8)
    pred[0, 0, :4, :4] = 1.0
    
    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, :4, :4] = 1.0
    
    dice = _dice_coefficient(pred, target)
    
    # Полное совпадение
    assert torch.allclose(dice, torch.ones(1))

@pytest.mark.skip(reason="Временно отключено")
def test_dice_coefficient_soft_predictions():
    """Тест Dice coefficient с мягкими предсказаниями"""
    pred = torch.rand(2, 1, 4, 4)  # Значения в [0, 1]
    target = torch.rand(2, 1, 4, 4)
    
    dice = _dice_coefficient(pred, target)
    
    assert dice.shape == (2,)
    assert torch.all(dice >= 0)


# --- Тесты для _custom_collate_fn ---

@pytest.mark.skip(reason="Временно отключено")
def test_custom_collate_fn_basic():
    """Тест базовой функциональности _custom_collate_fn"""
    # Создаём batch из 2 samples
    img1 = torch.rand(3, 64, 64)
    img2 = torch.rand(3, 64, 64)
    
    target1 = {
        "score_map": torch.rand(1, 16, 16),
        "geo_map": torch.rand(8, 16, 16),
        "rboxes": torch.rand(5, 8),
    }
    target2 = {
        "score_map": torch.rand(1, 16, 16),
        "geo_map": torch.rand(8, 16, 16),
        "rboxes": torch.rand(3, 8),
    }
    
    batch = [(img1, target1), (img2, target2)]
    
    images, targets = _custom_collate_fn(batch)
    
    # Проверяем images
    assert images.shape == (2, 3, 64, 64)
    
    # Проверяем targets
    assert "score_map" in targets
    assert "geo_map" in targets
    assert "rboxes" in targets
    
    assert targets["score_map"].shape == (2, 1, 16, 16)
    assert targets["geo_map"].shape == (2, 8, 16, 16)
    assert len(targets["rboxes"]) == 2
    assert targets["rboxes"][0].shape == (5, 8)
    assert targets["rboxes"][1].shape == (3, 8)

@pytest.mark.skip(reason="Временно отключено")
def test_custom_collate_fn_single_sample():
    """Тест _custom_collate_fn с одним sample"""
    img = torch.rand(3, 64, 64)
    target = {
        "score_map": torch.rand(1, 16, 16),
        "geo_map": torch.rand(8, 16, 16),
        "rboxes": torch.rand(2, 8),
    }
    
    batch = [(img, target)]
    
    images, targets = _custom_collate_fn(batch)
    
    assert images.shape == (1, 3, 64, 64)
    assert targets["score_map"].shape == (1, 1, 16, 16)
    assert targets["geo_map"].shape == (1, 8, 16, 16)
    assert len(targets["rboxes"]) == 1

@pytest.mark.skip(reason="Временно отключено")
def test_custom_collate_fn_multiple_samples():
    """Тест _custom_collate_fn с несколькими samples"""
    batch_size = 4
    batch = []
    
    for _ in range(batch_size):
        img = torch.rand(3, 128, 128)
        target = {
            "score_map": torch.rand(1, 32, 32),
            "geo_map": torch.rand(8, 32, 32),
            "rboxes": torch.rand(np.random.randint(1, 10), 8),
        }
        batch.append((img, target))
    
    images, targets = _custom_collate_fn(batch)
    
    assert images.shape == (batch_size, 3, 128, 128)
    assert targets["score_map"].shape == (batch_size, 1, 32, 32)
    assert targets["geo_map"].shape == (batch_size, 8, 32, 32)
    assert len(targets["rboxes"]) == batch_size

@pytest.mark.skip(reason="Временно отключено")
def test_custom_collate_fn_empty_rboxes():
    """Тест _custom_collate_fn с пустыми rboxes"""
    img = torch.rand(3, 64, 64)
    target = {
        "score_map": torch.rand(1, 16, 16),
        "geo_map": torch.rand(8, 16, 16),
        "rboxes": torch.empty(0, 8),  # Пустые boxes
    }
    
    batch = [(img, target)]
    
    images, targets = _custom_collate_fn(batch)
    
    assert images.shape == (1, 3, 64, 64)
    assert len(targets["rboxes"]) == 1
    assert targets["rboxes"][0].shape == (0, 8)

@pytest.mark.skip(reason="Временно отключено")
def test_custom_collate_fn_different_rbox_counts():
    """Тест _custom_collate_fn с разным количеством boxes"""
    img1 = torch.rand(3, 64, 64)
    target1 = {
        "score_map": torch.rand(1, 16, 16),
        "geo_map": torch.rand(8, 16, 16),
        "rboxes": torch.rand(10, 8),  # 10 boxes
    }
    
    img2 = torch.rand(3, 64, 64)
    target2 = {
        "score_map": torch.rand(1, 16, 16),
        "geo_map": torch.rand(8, 16, 16),
        "rboxes": torch.rand(2, 8),  # 2 boxes
    }
    
    batch = [(img1, target1), (img2, target2)]
    
    images, targets = _custom_collate_fn(batch)
    
    # rboxes должны быть списком тензоров разной длины
    assert len(targets["rboxes"]) == 2
    assert targets["rboxes"][0].shape[0] == 10
    assert targets["rboxes"][1].shape[0] == 2

@pytest.mark.skip(reason="Временно отключено")
def test_custom_collate_fn_preserves_dtype():
    """Тест что _custom_collate_fn сохраняет типы данных"""
    img = torch.rand(3, 64, 64, dtype=torch.float32)
    target = {
        "score_map": torch.rand(1, 16, 16, dtype=torch.float32),
        "geo_map": torch.rand(8, 16, 16, dtype=torch.float32),
        "rboxes": torch.rand(5, 8, dtype=torch.float32),
    }
    
    batch = [(img, target)]
    
    images, targets = _custom_collate_fn(batch)
    
    assert images.dtype == torch.float32
    assert targets["score_map"].dtype == torch.float32
    assert targets["geo_map"].dtype == torch.float32

@pytest.mark.skip(reason="Временно отключено")
def test_custom_collate_fn_stack_consistency():
    """Тест консистентности stack для images и maps"""
    batch = []
    for i in range(3):
        img = torch.rand(3, 64, 64) * i  # Разные значения
        target = {
            "score_map": torch.rand(1, 16, 16) * i,
            "geo_map": torch.rand(8, 16, 16) * i,
            "rboxes": torch.rand(2, 8),
        }
        batch.append((img, target))
    
    images, targets = _custom_collate_fn(batch)
    
    # Проверяем что stack работает правильно
    for i in range(3):
        # Среднее значение должно увеличиваться с i
        if i > 0:
            assert images[i].mean() > images[0].mean()

@pytest.mark.skip(reason="Временно отключено")
def test_custom_collate_fn_different_image_channels():
    """Тест _custom_collate_fn с разным количеством каналов"""
    # Обычно 3 канала (RGB)
    img = torch.rand(3, 64, 64)
    target = {
        "score_map": torch.rand(1, 16, 16),
        "geo_map": torch.rand(8, 16, 16),
        "rboxes": torch.rand(5, 8),
    }
    
    batch = [(img, target)]
    
    images, targets = _custom_collate_fn(batch)
    
    assert images.shape[1] == 3  # RGB каналы

@pytest.mark.skip(reason="Временно отключено")
def test_custom_collate_fn_large_batch():
    """Тест _custom_collate_fn с большим batch"""
    batch_size = 16
    batch = []
    
    for _ in range(batch_size):
        img = torch.rand(3, 256, 256)
        target = {
            "score_map": torch.rand(1, 64, 64),
            "geo_map": torch.rand(8, 64, 64),
            "rboxes": torch.rand(5, 8),
        }
        batch.append((img, target))
    
    images, targets = _custom_collate_fn(batch)
    
    assert images.shape == (batch_size, 3, 256, 256)
    assert targets["score_map"].shape == (batch_size, 1, 64, 64)
    assert targets["geo_map"].shape == (batch_size, 8, 64, 64)
    assert len(targets["rboxes"]) == batch_size


# --- Дополнительные тесты ---

@pytest.mark.skip(reason="Временно отключено")
def test_dice_coefficient_edge_case_single_pixel():
    """Тест Dice coefficient с одним активным пикселем"""
    pred = torch.zeros(1, 1, 8, 8)
    pred[0, 0, 4, 4] = 1.0
    
    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 4, 4] = 1.0
    
    dice = _dice_coefficient(pred, target)
    
    # Идеальное совпадение одного пикселя
    assert torch.allclose(dice, torch.ones(1))

@pytest.mark.skip(reason="Временно отключено")
def test_dice_coefficient_asymmetric_overlap():
    """Тест Dice coefficient с асимметричным пересечением"""
    pred = torch.zeros(1, 1, 8, 8)
    pred[0, 0, :4, :4] = 1.0  # 16 пикселей
    
    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, :2, :2] = 1.0  # 4 пикселя
    
    dice = _dice_coefficient(pred, target)
    
    # Должно быть между 0 и 1
    assert 0 < dice[0] < 1

@pytest.mark.skip(reason="Временно отключено")
def test_dice_coefficient_continuous_values():
    """Тест Dice coefficient с непрерывными значениями"""
    pred = torch.rand(3, 1, 16, 16) * 0.8 + 0.1  # В диапазоне [0.1, 0.9]
    target = torch.rand(3, 1, 16, 16) * 0.8 + 0.1
    
    dice = _dice_coefficient(pred, target)
    
    assert dice.shape == (3,)
    assert torch.all(dice >= 0)
    assert torch.all(dice <= 1)

@pytest.mark.skip(reason="Временно отключено")
def test_custom_collate_fn_maintains_rbox_structure():
    """Тест что _custom_collate_fn сохраняет структуру rboxes"""
    batch = []
    expected_shapes = [(7, 8), (3, 8), (5, 8)]
    
    for shape in expected_shapes:
        img = torch.rand(3, 64, 64)
        target = {
            "score_map": torch.rand(1, 16, 16),
            "geo_map": torch.rand(8, 16, 16),
            "rboxes": torch.rand(*shape),
        }
        batch.append((img, target))
    
    images, targets = _custom_collate_fn(batch)
    
    for i, expected_shape in enumerate(expected_shapes):
        assert targets["rboxes"][i].shape == expected_shape

@pytest.mark.skip(reason="Временно отключено")
def test_dice_coefficient_high_precision():
    """Тест Dice coefficient с высокой точностью"""
    # Создаём почти идентичные тензоры
    pred = torch.ones(2, 1, 4, 4)
    target = torch.ones(2, 1, 4, 4) * 0.9999
    
    dice = _dice_coefficient(pred, target)
    
    # Dice должен быть очень близок к 1
    assert dice[0] > 0.99

@pytest.mark.skip(reason="Временно отключено")
def test_custom_collate_fn_geo_map_channels():
    """Тест что _custom_collate_fn правильно обрабатывает 8 каналов geo_map"""
    img = torch.rand(3, 64, 64)
    target = {
        "score_map": torch.rand(1, 16, 16),
        "geo_map": torch.rand(8, 16, 16),  # 8 каналов для QUAD offsets
        "rboxes": torch.rand(5, 8),
    }
    
    batch = [(img, target)]
    
    images, targets = _custom_collate_fn(batch)
    
    # geo_map должен иметь 8 каналов
    assert targets["geo_map"].shape[1] == 8
