import pytest
import numpy as np
import torch
from shapely.geometry import Polygon
from PIL import Image
from manuscript.detectors._east.utils import (
    poly_iou,
    compute_f1,
    quad_to_rbox,
    tensor_to_image,
    draw_quads,
    read_image,
    box_iou,
    compute_f1_score,
    match_boxes,
)

# Глобальные переменные, которые использует compute_f1
gt_segs = {}
processed_ids = []


def test_identical_polygons():
    seg = [0, 0, 1, 0, 1, 1, 0, 1]
    assert poly_iou(seg, seg) == 1.0


def test_non_overlapping_polygons():
    segA = [0, 0, 1, 0, 1, 1, 0, 1]
    segB = [2, 2, 3, 2, 3, 3, 2, 3]
    assert poly_iou(segA, segB) == 0.0


def test_partial_overlap():
    segA = [0, 0, 2, 0, 2, 2, 0, 2]
    segB = [1, 1, 3, 1, 3, 3, 1, 3]
    expected_iou = (
        Polygon(np.array(segA).reshape(-1, 2))
        .intersection(Polygon(np.array(segB).reshape(-1, 2)))
        .area
        / Polygon(np.array(segA).reshape(-1, 2))
        .union(Polygon(np.array(segB).reshape(-1, 2)))
        .area
    )
    assert pytest.approx(poly_iou(segA, segB), 0.01) == expected_iou


def test_invalid_polygon():
    segA = [0, 0, 1, 1, 1, 1]  # Недопустимое количество точек
    segB = [0, 0, 1, 0, 1, 1, 0, 1]
    assert poly_iou(segA, segB) == 0.0


def test_zero_area_union():
    segA = [0, 0, 0, 0, 0, 0, 0, 0]  # Точки совпадают
    segB = [0, 0, 0, 0, 0, 0, 0, 0]
    assert poly_iou(segA, segB) == 0.0


def test_compute_f1_perfect_match():
    global gt_segs, processed_ids
    gt_segs = {"img1": [[0, 0, 1, 0, 1, 1, 0, 1]]}
    processed_ids = ["img1"]
    preds = [{"image_id": "img1", "segmentation": [0, 0, 1, 0, 1, 1, 0, 1]}]
    assert compute_f1(preds, 0.5, gt_segs, processed_ids) == 1.0


def test_compute_f1_no_match():
    gt_segs = {"img1": [[0, 0, 1, 0, 1, 1, 0, 1]]}
    processed_ids = ["img1"]
    preds = [{"image_id": "img1", "segmentation": [2, 2, 3, 2, 3, 3, 2, 3]}]
    assert compute_f1(preds, 0.5, gt_segs, processed_ids) == 0.0


def test_compute_f1_partial_match():
    gt_segs = {"img1": [[0, 0, 2, 0, 2, 2, 0, 2]]}
    processed_ids = ["img1"]
    preds = [{"image_id": "img1", "segmentation": [1, 1, 3, 1, 3, 3, 1, 3]}]
    iou = poly_iou(preds[0]["segmentation"], gt_segs["img1"][0])
    expected_f1 = 1.0 if iou >= 0.5 else 0.0
    assert compute_f1(preds, 0.5, gt_segs, processed_ids) == expected_f1


# --- Тесты для quad_to_rbox ---


def test_quad_to_rbox_square():
    """Тест quad_to_rbox для квадрата"""
    quad = np.array([0, 0, 10, 0, 10, 10, 0, 10, 0.9])  # Квадрат 10x10
    rbox = quad_to_rbox(quad)
    
    assert rbox.shape == (5,)
    assert isinstance(rbox[0], (float, np.floating))  # cx
    assert isinstance(rbox[1], (float, np.floating))  # cy
    assert rbox[2] > 0  # width
    assert rbox[3] > 0  # height
    # Центр должен быть примерно (5, 5)
    assert abs(rbox[0] - 5) < 1
    assert abs(rbox[1] - 5) < 1


def test_quad_to_rbox_rectangle():
    """Тест quad_to_rbox для прямоугольника"""
    quad = np.array([0, 0, 20, 0, 20, 10, 0, 10])  # Прямоугольник 20x10
    rbox = quad_to_rbox(quad)
    
    assert rbox.shape == (5,)
    # Центр должен быть примерно (10, 5)
    assert abs(rbox[0] - 10) < 1
    assert abs(rbox[1] - 5) < 1


def test_quad_to_rbox_rotated():
    """Тест quad_to_rbox для повёрнутого прямоугольника"""
    # Прямоугольник, повёрнутый на 45 градусов
    angle_rad = np.pi / 4
    w, h = 20, 10
    cx, cy = 50, 50
    
    # Генерируем точки повёрнутого прямоугольника
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    pts = np.array([
        [-w/2, -h/2],
        [w/2, -h/2],
        [w/2, h/2],
        [-w/2, h/2]
    ])
    
    rotated_pts = np.array([
        [cx + pt[0]*cos_a - pt[1]*sin_a, cy + pt[0]*sin_a + pt[1]*cos_a]
        for pt in pts
    ])
    
    quad = rotated_pts.flatten()
    rbox = quad_to_rbox(quad)
    
    assert rbox.shape == (5,)
    assert abs(rbox[0] - cx) < 1  # cx
    assert abs(rbox[1] - cy) < 1  # cy


# --- Тесты для tensor_to_image ---


def test_tensor_to_image_basic():
    """Тест tensor_to_image с обычным тензором"""
    # Создаём тензор 3x64x64
    tensor = torch.rand(3, 64, 64)
    
    img = tensor_to_image(tensor)
    
    assert isinstance(img, np.ndarray)
    assert img.shape == (64, 64, 3)
    assert img.dtype == np.uint8
    assert img.min() >= 0
    assert img.max() <= 255


def test_tensor_to_image_normalized():
    """Тест tensor_to_image с нормализованным тензором"""
    # Тензор с значениями в диапазоне [-1, 1]
    tensor = torch.randn(3, 32, 32)
    
    img = tensor_to_image(tensor)
    
    assert img.shape == (32, 32, 3)
    assert img.dtype == np.uint8


def test_tensor_to_image_zeros():
    """Тест tensor_to_image с нулевым тензором"""
    tensor = torch.zeros(3, 16, 16)
    
    img = tensor_to_image(tensor)
    
    # После денормализации и clipping должны получить валидное изображение
    assert img.shape == (16, 16, 3)
    assert np.all(img >= 0)
    assert np.all(img <= 255)


def test_tensor_to_image_ones():
    """Тест tensor_to_image с единичным тензором"""
    tensor = torch.ones(3, 16, 16)
    
    img = tensor_to_image(tensor)
    
    assert img.shape == (16, 16, 3)
    assert np.all(img >= 0)
    assert np.all(img <= 255)


# --- Тесты для draw_quads ---


def test_draw_quads_empty():
    """Тест draw_quads с пустым массивом boxes"""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    quads = np.array([])
    
    result = draw_quads(img, quads)
    
    assert isinstance(result, Image.Image)
    assert np.array(result).shape == img.shape


def test_draw_quads_single_box():
    """Тест draw_quads с одним box"""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    quads = np.array([[10, 10, 50, 10, 50, 50, 10, 50, 0.9]])
    
    result = draw_quads(img, quads)
    
    assert isinstance(result, Image.Image)
    assert np.array(result).shape == img.shape


def test_draw_quads_multiple_boxes():
    """Тест draw_quads с несколькими boxes"""
    img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    quads = np.array([
        [10, 10, 50, 10, 50, 50, 10, 50, 0.9],
        [60, 60, 100, 60, 100, 100, 60, 100, 0.8],
        [110, 110, 150, 110, 150, 150, 110, 150, 0.7],
    ])
    
    result = draw_quads(img, quads)
    
    assert isinstance(result, Image.Image)


def test_draw_quads_with_tensor():
    """Тест draw_quads с torch.Tensor"""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    quads = torch.tensor([[10, 10, 50, 10, 50, 50, 10, 50, 0.9]])
    
    result = draw_quads(img, quads)
    
    assert isinstance(result, Image.Image)


def test_draw_quads_none():
    """Тест draw_quads с None"""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    result = draw_quads(img, None)
    
    assert isinstance(result, Image.Image)


def test_draw_quads_custom_color():
    """Тест draw_quads с кастомным цветом"""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    quads = np.array([[10, 10, 50, 10, 50, 50, 10, 50]])
    
    result = draw_quads(img, quads, color=(255, 0, 0), thickness=2)
    
    assert isinstance(result, Image.Image)


# --- Тесты для read_image ---


def test_read_image_numpy_array():
    """Тест read_image с numpy array"""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    result = read_image(img)
    
    assert isinstance(result, np.ndarray)
    assert result.shape == img.shape
    assert result.dtype == np.uint8


def test_read_image_pil():
    """Тест read_image с PIL Image - должен вызвать ошибку"""
    pil_img = Image.new('RGB', (100, 100), color=(255, 0, 0))
    
    with pytest.raises(TypeError, match="Unsupported type for image input"):
        read_image(pil_img)


def test_read_image_torch_tensor():
    """Тест read_image с torch.Tensor - должен вызвать ошибку"""
    tensor = torch.rand(3, 100, 100)
    
    with pytest.raises(TypeError, match="Unsupported type for image input"):
        read_image(tensor)


# --- Тесты для box_iou ---


def test_box_iou_identical():
    """Тест box_iou с идентичными boxes"""
    box1 = (10, 10, 50, 50)
    box2 = (10, 10, 50, 50)
    
    iou = box_iou(box1, box2)
    
    assert iou == 1.0


def test_box_iou_no_overlap():
    """Тест box_iou без пересечения"""
    box1 = (0, 0, 10, 10)
    box2 = (20, 20, 30, 30)
    
    iou = box_iou(box1, box2)
    
    assert iou == 0.0


def test_box_iou_partial_overlap():
    """Тест box_iou с частичным пересечением"""
    box1 = (0, 0, 20, 20)  # Площадь 400
    box2 = (10, 10, 30, 30)  # Площадь 400, пересечение 10x10=100
    
    iou = box_iou(box1, box2)
    
    # Пересечение 100, объединение 400+400-100=700
    expected = 100 / 700
    assert abs(iou - expected) < 0.01


def test_box_iou_one_inside_other():
    """Тест box_iou когда один box внутри другого"""
    box1 = (0, 0, 100, 100)  # Большой
    box2 = (25, 25, 75, 75)  # Маленький внутри
    
    iou = box_iou(box1, box2)
    
    # Пересечение = площадь box2 = 2500
    # Объединение = площадь box1 = 10000
    expected = 2500 / 10000
    assert abs(iou - expected) < 0.01


def test_box_iou_edge_touching():
    """Тест box_iou когда boxes касаются краями"""
    box1 = (0, 0, 10, 10)
    box2 = (10, 0, 20, 10)  # Касается правого края box1
    
    iou = box_iou(box1, box2)
    
    # Нет пересечения (только касание)
    assert iou == 0.0


# --- Тесты для compute_f1_score ---


def test_compute_f1_score_perfect():
    """Тест compute_f1_score при идеальных результатах"""
    tp, fp, fn = 10, 0, 0
    
    f1, precision, recall = compute_f1_score(tp, fp, fn)
    
    assert f1 == 1.0
    assert precision == 1.0
    assert recall == 1.0


def test_compute_f1_score_zero():
    """Тест compute_f1_score при нулевых результатах"""
    tp, fp, fn = 0, 10, 10
    
    f1, precision, recall = compute_f1_score(tp, fp, fn)
    
    assert f1 == 0.0
    assert precision == 0.0
    assert recall == 0.0


def test_compute_f1_score_balanced():
    """Тест compute_f1_score со средними результатами"""
    tp, fp, fn = 5, 5, 5
    
    f1, precision, recall = compute_f1_score(tp, fp, fn)
    
    # precision = 5/(5+5) = 0.5
    # recall = 5/(5+5) = 0.5
    # f1 = 2*0.5*0.5/(0.5+0.5) = 0.5
    assert abs(f1 - 0.5) < 0.01
    assert abs(precision - 0.5) < 0.01
    assert abs(recall - 0.5) < 0.01


def test_compute_f1_score_high_precision_low_recall():
    """Тест compute_f1_score с высокой precision, низким recall"""
    tp, fp, fn = 9, 1, 10
    
    f1, precision, recall = compute_f1_score(tp, fp, fn)
    
    # precision = 9/10 = 0.9
    # recall = 9/19 ≈ 0.47
    # f1 = 2*0.9*0.47/(0.9+0.47) ≈ 0.62
    assert 0 < f1 < 1
    assert 0 < precision < 1
    assert 0 < recall < 1


def test_compute_f1_score_all_zeros():
    """Тест compute_f1_score когда все нули"""
    tp, fp, fn = 0, 0, 0
    
    f1, precision, recall = compute_f1_score(tp, fp, fn)
    
    assert f1 == 0.0
    assert precision == 0.0
    assert recall == 0.0


# --- Тесты для match_boxes ---


def test_match_boxes_perfect_match():
    """Тест match_boxes при идеальном совпадении"""
    pred_boxes = [(10, 10, 50, 50), (60, 60, 100, 100)]
    gt_boxes = [(10, 10, 50, 50), (60, 60, 100, 100)]
    
    tp, fp, fn = match_boxes(pred_boxes, gt_boxes, iou_threshold=0.5)
    
    assert tp == 2
    assert fp == 0
    assert fn == 0


def test_match_boxes_no_match():
    """Тест match_boxes без совпадений"""
    pred_boxes = [(0, 0, 10, 10)]
    gt_boxes = [(50, 50, 60, 60)]
    
    tp, fp, fn = match_boxes(pred_boxes, gt_boxes, iou_threshold=0.5)
    
    assert tp == 0
    assert fp == 1
    assert fn == 1


def test_match_boxes_partial():
    """Тест match_boxes с частичным совпадением"""
    pred_boxes = [(0, 0, 20, 20), (50, 50, 70, 70)]
    gt_boxes = [(10, 10, 30, 30)]  # Пересекается только с первым
    
    tp, fp, fn = match_boxes(pred_boxes, gt_boxes, iou_threshold=0.3)
    
    assert tp >= 0
    assert fp >= 0
    assert fn >= 0
    assert tp + fp == len(pred_boxes)
    assert tp + fn == len(gt_boxes)


def test_match_boxes_empty_predictions():
    """Тест match_boxes с пустыми предсказаниями"""
    pred_boxes = []
    gt_boxes = [(10, 10, 50, 50)]
    
    tp, fp, fn = match_boxes(pred_boxes, gt_boxes, iou_threshold=0.5)
    
    assert tp == 0
    assert fp == 0
    assert fn == 1


def test_match_boxes_empty_ground_truth():
    """Тест match_boxes с пустым ground truth"""
    pred_boxes = [(10, 10, 50, 50)]
    gt_boxes = []
    
    tp, fp, fn = match_boxes(pred_boxes, gt_boxes, iou_threshold=0.5)
    
    assert tp == 0
    assert fp == 1
    assert fn == 0


def test_match_boxes_different_thresholds():
    """Тест match_boxes с разными порогами IoU"""
    pred_boxes = [(0, 0, 20, 20)]
    gt_boxes = [(10, 10, 30, 30)]
    
    # Низкий порог - должны совпасть
    tp_low, _, _ = match_boxes(pred_boxes, gt_boxes, iou_threshold=0.1)
    
    # Высокий порог - могут не совпасть
    tp_high, _, _ = match_boxes(pred_boxes, gt_boxes, iou_threshold=0.9)
    
    assert tp_low >= tp_high


# --- Дополнительные тесты ---


def test_quad_to_rbox_dtype():
    """Тест что quad_to_rbox возвращает правильный dtype"""
    quad = np.array([0, 0, 10, 0, 10, 10, 0, 10], dtype=np.float64)
    rbox = quad_to_rbox(quad)
    
    assert rbox.dtype == np.float32


def test_tensor_to_image_gradient():
    """Тест tensor_to_image с градиентом"""
    # Создаём тензор с градиентом
    tensor = torch.linspace(-1, 1, 64*64*3).reshape(3, 64, 64)
    tensor.requires_grad = True
    
    img = tensor_to_image(tensor)
    
    # Должно работать несмотря на requires_grad
    assert img.shape == (64, 64, 3)


def test_draw_quads_different_sizes():
    """Тест draw_quads с разными размерами изображений"""
    sizes = [(100, 100), (200, 300), (512, 512)]
    
    for h, w in sizes:
        img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        quads = np.array([[10, 10, 50, 10, 50, 50, 10, 50]])
        
        result = draw_quads(img, quads)
        
        assert isinstance(result, Image.Image)
        assert np.array(result).shape == (h, w, 3)


def test_box_iou_floating_point():
    """Тест box_iou с floating point координатами"""
    box1 = (10.5, 10.5, 50.7, 50.3)
    box2 = (20.2, 20.8, 60.1, 60.9)
    
    iou = box_iou(box1, box2)
    
    assert 0 <= iou <= 1


def test_compute_f1_score_edge_cases():
    """Тест compute_f1_score с граничными случаями"""
    # Только true positives
    f1, precision, recall = compute_f1_score(10, 0, 0)
    assert f1 == 1.0
    assert precision == 1.0
    assert recall == 1.0
    
    # Только false positives
    f1, precision, recall = compute_f1_score(0, 10, 0)
    assert f1 == 0.0
    
    # Только false negatives
    f1, precision, recall = compute_f1_score(0, 0, 10)
    assert f1 == 0.0
