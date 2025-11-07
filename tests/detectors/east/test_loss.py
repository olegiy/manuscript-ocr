import pytest
import torch
from manuscript.detectors._east.loss import compute_dice_loss, EASTLoss


# --- Тесты для compute_dice_loss ---


def test_compute_dice_loss_perfect_match():
    """Тест Dice loss при идеальном совпадении"""
    gt = torch.ones(2, 1, 4, 4)
    pred = torch.ones(2, 1, 4, 4)
    
    loss = compute_dice_loss(gt, pred)
    
    # При идеальном совпадении loss должен быть близок к 0
    assert loss < 0.01


def test_compute_dice_loss_no_overlap():
    """Тест Dice loss при отсутствии совпадения"""
    gt = torch.zeros(2, 1, 4, 4)
    gt[0, 0, 0, 0] = 1.0
    pred = torch.zeros(2, 1, 4, 4)
    pred[0, 0, 3, 3] = 1.0
    
    loss = compute_dice_loss(gt, pred)
    
    # При отсутствии пересечения loss близок к 1
    assert loss > 0.5


def test_compute_dice_loss_partial_overlap():
    """Тест Dice loss при частичном совпадении"""
    gt = torch.zeros(1, 1, 4, 4)
    gt[0, 0, :2, :2] = 1.0  # Левый верхний квадрант
    
    pred = torch.zeros(1, 1, 4, 4)
    pred[0, 0, 1:3, 1:3] = 1.0  # Пересекается с gt
    
    loss = compute_dice_loss(gt, pred)
    
    # Должно быть между 0 и 1
    assert 0 < loss < 1


def test_compute_dice_loss_stability():
    """Тест стабильности Dice loss (защита от деления на ноль)"""
    gt = torch.zeros(1, 1, 4, 4)
    pred = torch.zeros(1, 1, 4, 4)
    
    loss = compute_dice_loss(gt, pred)
    
    # Должно работать без ошибок благодаря epsilon
    assert torch.isfinite(loss)


def test_compute_dice_loss_different_scales():
    """Тест Dice loss с разными значениями интенсивности"""
    gt = torch.ones(1, 1, 4, 4) * 0.5
    pred = torch.ones(1, 1, 4, 4) * 0.5
    
    loss = compute_dice_loss(gt, pred)
    
    # Dice loss зависит от абсолютных значений, не только от совпадения
    # inter = 0.5*0.5*16 = 4, union = 0.5*16 + 0.5*16 + eps = 16 + eps
    # loss = 1 - 2*4/(16+eps) = 1 - 0.5 = 0.5
    assert abs(loss - 0.5) < 0.01


# --- Тесты для EASTLoss инициализации ---


def test_east_loss_initialization_default():
    """Тест инициализации EASTLoss с параметрами по умолчанию"""
    loss_fn = EASTLoss()
    
    assert loss_fn.use_ohem == False
    assert loss_fn.ohem_ratio == 0.5
    assert loss_fn.use_focal_geo == False
    assert loss_fn.focal_gamma == 2.0


def test_east_loss_initialization_ohem():
    """Тест инициализации EASTLoss с OHEM"""
    loss_fn = EASTLoss(use_ohem=True, ohem_ratio=0.3)
    
    assert loss_fn.use_ohem == True
    assert loss_fn.ohem_ratio == 0.3


def test_east_loss_initialization_focal():
    """Тест инициализации EASTLoss с Focal loss"""
    loss_fn = EASTLoss(use_focal_geo=True, focal_gamma=3.0)
    
    assert loss_fn.use_focal_geo == True
    assert loss_fn.focal_gamma == 3.0


# --- Тесты для EASTLoss forward ---


def test_east_loss_forward_basic():
    """Тест базового forward pass для EASTLoss"""
    loss_fn = EASTLoss()
    
    gt_score = torch.ones(2, 1, 8, 8)
    pred_score = torch.ones(2, 1, 8, 8) * 0.9
    gt_geo = torch.ones(2, 8, 8, 8)
    pred_geo = torch.ones(2, 8, 8, 8) * 0.95
    
    loss = loss_fn(gt_score, pred_score, gt_geo, pred_geo)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar
    assert loss >= 0


def test_east_loss_forward_no_positives():
    """Тест forward когда нет положительных пикселей"""
    loss_fn = EASTLoss()
    
    gt_score = torch.zeros(2, 1, 8, 8)  # Нет положительных
    pred_score = torch.ones(2, 1, 8, 8)
    gt_geo = torch.ones(2, 8, 8, 8)
    pred_geo = torch.ones(2, 8, 8, 8)
    
    loss = loss_fn(gt_score, pred_score, gt_geo, pred_geo)
    
    # Должен вернуть 0 с requires_grad=True
    assert loss == 0.0
    assert loss.requires_grad


def test_east_loss_forward_perfect_prediction():
    """Тест forward при идеальном предсказании"""
    loss_fn = EASTLoss()
    
    gt_score = torch.ones(1, 1, 4, 4)
    pred_score = torch.ones(1, 1, 4, 4)
    gt_geo = torch.ones(1, 8, 4, 4)
    pred_geo = torch.ones(1, 8, 4, 4)
    
    loss = loss_fn(gt_score, pred_score, gt_geo, pred_geo)
    
    # При идеальном предсказании loss близок к 0
    assert loss < 0.1


def test_east_loss_forward_with_ohem():
    """Тест forward с OHEM"""
    loss_fn = EASTLoss(use_ohem=True, ohem_ratio=0.5)
    
    gt_score = torch.ones(2, 1, 8, 8)
    pred_score = torch.rand(2, 1, 8, 8)
    gt_geo = torch.ones(2, 8, 8, 8)
    pred_geo = torch.rand(2, 8, 8, 8)
    
    loss = loss_fn(gt_score, pred_score, gt_geo, pred_geo)
    
    assert isinstance(loss, torch.Tensor)
    assert loss >= 0


def test_east_loss_forward_with_focal():
    """Тест forward с Focal geometry loss"""
    loss_fn = EASTLoss(use_focal_geo=True, focal_gamma=2.0)
    
    gt_score = torch.ones(2, 1, 8, 8)
    pred_score = torch.rand(2, 1, 8, 8)
    gt_geo = torch.ones(2, 8, 8, 8)
    pred_geo = torch.rand(2, 8, 8, 8)
    
    loss = loss_fn(gt_score, pred_score, gt_geo, pred_geo)
    
    assert isinstance(loss, torch.Tensor)
    assert loss >= 0


def test_east_loss_forward_with_ohem_and_focal():
    """Тест forward с OHEM и Focal одновременно"""
    loss_fn = EASTLoss(use_ohem=True, ohem_ratio=0.3, use_focal_geo=True, focal_gamma=3.0)
    
    gt_score = torch.ones(2, 1, 8, 8)
    pred_score = torch.rand(2, 1, 8, 8)
    gt_geo = torch.ones(2, 8, 8, 8)
    pred_geo = torch.rand(2, 8, 8, 8)
    
    loss = loss_fn(gt_score, pred_score, gt_geo, pred_geo)
    
    assert isinstance(loss, torch.Tensor)
    assert loss >= 0


def test_east_loss_gradient_flow():
    """Тест что градиенты проходят через loss"""
    loss_fn = EASTLoss()
    
    gt_score = torch.ones(1, 1, 4, 4)
    pred_score = torch.rand(1, 1, 4, 4, requires_grad=True)
    gt_geo = torch.ones(1, 8, 4, 4)
    pred_geo = torch.rand(1, 8, 4, 4, requires_grad=True)
    
    loss = loss_fn(gt_score, pred_score, gt_geo, pred_geo)
    loss.backward()
    
    # Проверяем что градиенты есть
    assert pred_score.grad is not None
    assert pred_geo.grad is not None
    assert pred_score.grad.abs().sum() > 0
    assert pred_geo.grad.abs().sum() > 0


def test_east_loss_partial_gt_mask():
    """Тест forward с частичной маской gt_score"""
    loss_fn = EASTLoss()
    
    gt_score = torch.zeros(2, 1, 8, 8)
    gt_score[:, :, :4, :4] = 1.0  # Только верхний левый квадрант
    pred_score = torch.rand(2, 1, 8, 8)
    gt_geo = torch.ones(2, 8, 8, 8)
    pred_geo = torch.rand(2, 8, 8, 8)
    
    loss = loss_fn(gt_score, pred_score, gt_geo, pred_geo)
    
    assert loss > 0


def test_east_loss_different_batch_sizes():
    """Тест forward с разными размерами batch"""
    loss_fn = EASTLoss()
    
    for batch_size in [1, 2, 4, 8]:
        gt_score = torch.ones(batch_size, 1, 4, 4)
        pred_score = torch.rand(batch_size, 1, 4, 4)
        gt_geo = torch.ones(batch_size, 8, 4, 4)
        pred_geo = torch.rand(batch_size, 8, 4, 4)
        
        loss = loss_fn(gt_score, pred_score, gt_geo, pred_geo)
        assert loss >= 0


def test_east_loss_large_prediction_error():
    """Тест forward с большой ошибкой предсказания"""
    loss_fn = EASTLoss()
    
    gt_score = torch.ones(1, 1, 4, 4)
    pred_score = torch.zeros(1, 1, 4, 4)
    gt_geo = torch.ones(1, 8, 4, 4)
    pred_geo = torch.zeros(1, 8, 4, 4)
    
    loss = loss_fn(gt_score, pred_score, gt_geo, pred_geo)
    
    # При большой ошибке loss должен быть значительным
    assert loss > 1.0


def test_east_loss_ohem_ratio_extreme_values():
    """Тест OHEM с экстремальными значениями ratio"""
    # Очень низкий ratio
    loss_fn_low = EASTLoss(use_ohem=True, ohem_ratio=0.01)
    
    gt_score = torch.ones(2, 1, 8, 8)
    pred_score = torch.rand(2, 1, 8, 8)
    gt_geo = torch.ones(2, 8, 8, 8)
    pred_geo = torch.rand(2, 8, 8, 8)
    
    loss_low = loss_fn_low(gt_score, pred_score, gt_geo, pred_geo)
    assert torch.isfinite(loss_low)
    
    # Высокий ratio
    loss_fn_high = EASTLoss(use_ohem=True, ohem_ratio=0.99)
    loss_high = loss_fn_high(gt_score, pred_score, gt_geo, pred_geo)
    assert torch.isfinite(loss_high)


def test_east_loss_focal_gamma_variations():
    """Тест Focal loss с разными значениями gamma"""
    gt_score = torch.ones(1, 1, 8, 8)
    pred_score = torch.rand(1, 1, 8, 8)
    gt_geo = torch.ones(1, 8, 8, 8)
    pred_geo = torch.rand(1, 8, 8, 8)
    
    losses = []
    for gamma in [0.5, 1.0, 2.0, 5.0]:
        loss_fn = EASTLoss(use_focal_geo=True, focal_gamma=gamma)
        loss = loss_fn(gt_score, pred_score, gt_geo, pred_geo)
        losses.append(loss.item())
        assert torch.isfinite(loss)
    
    # Все loss должны быть валидными
    assert all(l >= 0 for l in losses)


def test_east_loss_geo_channels_independence():
    """Тест что все 8 каналов geometry учитываются"""
    loss_fn = EASTLoss()
    
    gt_score = torch.ones(1, 1, 4, 4)
    pred_score = torch.ones(1, 1, 4, 4)
    gt_geo = torch.ones(1, 8, 4, 4)
    
    # Предсказание с ошибкой только в одном канале
    pred_geo_one_channel = torch.ones(1, 8, 4, 4)
    pred_geo_one_channel[:, 0, :, :] = 0  # Ошибка в первом канале
    
    loss_one = loss_fn(gt_score, pred_score, gt_geo, pred_geo_one_channel)
    
    # Предсказание с ошибкой во всех каналах
    pred_geo_all_channels = torch.zeros(1, 8, 4, 4)
    
    loss_all = loss_fn(gt_score, pred_score, gt_geo, pred_geo_all_channels)
    
    # Loss с ошибкой во всех каналах должен быть больше
    assert loss_all > loss_one


def test_east_loss_device_cuda():
    """Тест что loss работает на CUDA (если доступна)"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    loss_fn = EASTLoss().cuda()
    
    gt_score = torch.ones(1, 1, 4, 4).cuda()
    pred_score = torch.rand(1, 1, 4, 4).cuda()
    gt_geo = torch.ones(1, 8, 4, 4).cuda()
    pred_geo = torch.rand(1, 8, 4, 4).cuda()
    
    loss = loss_fn(gt_score, pred_score, gt_geo, pred_geo)
    
    assert loss.device.type == "cuda"


def test_east_loss_device_cpu():
    """Тест что loss работает на CPU"""
    loss_fn = EASTLoss()
    
    gt_score = torch.ones(1, 1, 4, 4)
    pred_score = torch.rand(1, 1, 4, 4)
    gt_geo = torch.ones(1, 8, 4, 4)
    pred_geo = torch.rand(1, 8, 4, 4)
    
    loss = loss_fn(gt_score, pred_score, gt_geo, pred_geo)
    
    assert loss.device.type == "cpu"


def test_east_loss_numerical_stability():
    """Тест численной стабильности с экстремальными значениями"""
    loss_fn = EASTLoss()
    
    # Очень большие значения
    gt_score = torch.ones(1, 1, 4, 4) * 1e6
    pred_score = torch.ones(1, 1, 4, 4) * 1e6
    gt_geo = torch.ones(1, 8, 4, 4) * 1e6
    pred_geo = torch.ones(1, 8, 4, 4) * 1e6
    
    loss = loss_fn(gt_score, pred_score, gt_geo, pred_geo)
    assert torch.isfinite(loss)
    
    # Очень маленькие значения
    gt_score_small = torch.ones(1, 1, 4, 4) * 1e-6
    pred_score_small = torch.ones(1, 1, 4, 4) * 1e-6
    gt_geo_small = torch.ones(1, 8, 4, 4) * 1e-6
    pred_geo_small = torch.ones(1, 8, 4, 4) * 1e-6
    
    loss_small = loss_fn(gt_score_small, pred_score_small, gt_geo_small, pred_geo_small)
    assert torch.isfinite(loss_small)


def test_east_loss_single_positive_pixel():
    """Тест с одним положительным пикселем"""
    loss_fn = EASTLoss()
    
    gt_score = torch.zeros(1, 1, 8, 8)
    gt_score[0, 0, 4, 4] = 1.0  # Только один пиксель
    pred_score = torch.rand(1, 1, 8, 8)
    gt_geo = torch.ones(1, 8, 8, 8)
    pred_geo = torch.rand(1, 8, 8, 8)
    
    loss = loss_fn(gt_score, pred_score, gt_geo, pred_geo)
    
    assert torch.isfinite(loss)
    assert loss >= 0


def test_east_loss_ohem_single_batch():
    """Тест OHEM с batch_size=1"""
    loss_fn = EASTLoss(use_ohem=True, ohem_ratio=0.5)
    
    gt_score = torch.ones(1, 1, 8, 8)
    pred_score = torch.rand(1, 1, 8, 8)
    gt_geo = torch.ones(1, 8, 8, 8)
    pred_geo = torch.rand(1, 8, 8, 8)
    
    loss = loss_fn(gt_score, pred_score, gt_geo, pred_geo)
    
    assert torch.isfinite(loss)


def test_east_loss_ohem_multi_batch():
    """Тест OHEM с несколькими batch"""
    loss_fn = EASTLoss(use_ohem=True, ohem_ratio=0.5)
    
    gt_score = torch.ones(4, 1, 8, 8)
    pred_score = torch.rand(4, 1, 8, 8)
    gt_geo = torch.ones(4, 8, 8, 8)
    pred_geo = torch.rand(4, 8, 8, 8)
    
    loss = loss_fn(gt_score, pred_score, gt_geo, pred_geo)
    
    assert torch.isfinite(loss)


def test_compute_dice_loss_gradient():
    """Тест что градиенты проходят через dice loss"""
    gt = torch.ones(1, 1, 4, 4)
    pred = torch.rand(1, 1, 4, 4, requires_grad=True)
    
    loss = compute_dice_loss(gt, pred)
    loss.backward()
    
    assert pred.grad is not None
    assert pred.grad.abs().sum() > 0


def test_east_loss_comparison_ohem_vs_standard():
    """Тест сравнения OHEM vs стандартного loss"""
    gt_score = torch.ones(2, 1, 8, 8)
    pred_score = torch.rand(2, 1, 8, 8)
    gt_geo = torch.ones(2, 8, 8, 8)
    pred_geo = torch.rand(2, 8, 8, 8)
    
    loss_standard = EASTLoss(use_ohem=False)
    loss_ohem = EASTLoss(use_ohem=True, ohem_ratio=0.5)
    
    loss_std_val = loss_standard(gt_score, pred_score, gt_geo, pred_geo)
    loss_ohem_val = loss_ohem(gt_score, pred_score, gt_geo, pred_geo)
    
    # Оба должны быть валидными
    assert torch.isfinite(loss_std_val)
    assert torch.isfinite(loss_ohem_val)
    
    # Значения могут отличаться
    assert loss_std_val >= 0
    assert loss_ohem_val >= 0


def test_east_loss_different_spatial_sizes():
    """Тест с разными пространственными размерами"""
    loss_fn = EASTLoss()
    
    sizes = [(4, 4), (8, 8), (16, 16), (32, 32)]
    
    for h, w in sizes:
        gt_score = torch.ones(1, 1, h, w)
        pred_score = torch.rand(1, 1, h, w)
        gt_geo = torch.ones(1, 8, h, w)
        pred_geo = torch.rand(1, 8, h, w)
        
        loss = loss_fn(gt_score, pred_score, gt_geo, pred_geo)
        assert torch.isfinite(loss)
        assert loss >= 0
