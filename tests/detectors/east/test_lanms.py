import numpy as np
from manuscript.detectors._east.lanms import (
    polygon_area,
    compute_intersection,
    clip_polygon,
    polygon_intersection,
    polygon_iou,
    should_merge,
    normalize_polygon,
    standard_nms,
    locality_aware_nms,
)

import pytest


# --- Тесты для геометрических функций ---
def test_polygon_area_square():
    poly = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    area = polygon_area(poly)
    np.testing.assert_allclose(area, 1.0, rtol=1e-5)


def test_polygon_area_triangle():
    poly = np.array([[0, 0], [2, 0], [0, 2]], dtype=np.float64)
    area = polygon_area(poly)
    np.testing.assert_allclose(area, 2.0, rtol=1e-5)


def test_compute_intersection():
    # Пересечение двух отрезков, должно вернуть точку (1,1)
    p1 = np.array([0, 0], dtype=np.float64)
    p2 = np.array([2, 2], dtype=np.float64)
    A = np.array([0, 2], dtype=np.float64)
    B = np.array([2, 0], dtype=np.float64)
    inter = compute_intersection(p1, p2, A, B)
    np.testing.assert_allclose(inter, np.array([1, 1], dtype=np.float64), rtol=1e-5)


def test_clip_polygon():
    subject = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=np.float64)
    # Отсечение по линии x = 2
    A = np.array([2, 5], dtype=np.float64)
    B = np.array([2, -1], dtype=np.float64)
    clipped, count = clip_polygon(subject, A, B)
    expected = np.array([[2, 0], [4, 0], [4, 4], [2, 4]], dtype=np.float64)
    np.testing.assert_allclose(clipped, expected, rtol=1e-5)
    assert count == 4


def test_polygon_intersection():
    poly1 = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=np.float64)
    poly2 = np.array([[2, 2], [6, 2], [6, 6], [2, 6]], dtype=np.float64)
    inter_poly = polygon_intersection(poly1, poly2)
    expected = np.array([[2, 2], [4, 2], [4, 4], [2, 4]], dtype=np.float64)
    np.testing.assert_allclose(inter_poly, expected, rtol=1e-5)


def test_polygon_iou():
    poly1 = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=np.float64)
    poly2 = np.array([[2, 2], [6, 2], [6, 6], [2, 6]], dtype=np.float64)
    iou = polygon_iou(poly1, poly2)
    expected = 4 / (16 + 16 - 4)  # 4 / 28 ~ 0.142857
    assert np.isclose(iou, expected, rtol=1e-5)


def test_should_merge():
    poly1 = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=np.float64)
    poly2 = np.array([[2, 2], [6, 2], [6, 6], [2, 6]], dtype=np.float64)
    # Порог IoU 0.1 (0.142857 > 0.1) должен вернуть True
    assert should_merge(poly1, poly2, 0.1)
    # Порог IoU 0.2 (0.142857 < 0.2) должен вернуть False
    assert not should_merge(poly1, poly2, 0.2)


def test_normalize_polygon():
    ref = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=np.float64)
    poly = np.array(
        [[4, 4], [0, 4], [0, 0], [4, 0]], dtype=np.float64
    )  # Переставленные вершины
    normalized = normalize_polygon(ref, poly)
    np.testing.assert_allclose(normalized, ref, rtol=1e-5)


# --- Тесты для функций NMS ---
def test_standard_nms():
    # Три прямоугольника: два пересекаются, один нет
    polys = [
        np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=np.float64),
        np.array([[1, 1], [5, 1], [5, 5], [1, 5]], dtype=np.float64),
        np.array([[10, 10], [14, 10], [14, 14], [10, 14]], dtype=np.float64),
    ]
    scores = [0.9, 0.8, 0.7]
    iou_threshold = 0.1
    kept_polys, kept_scores = standard_nms(polys, scores, iou_threshold)
    # Первые два прямоугольника пересекаются, третий нет => ожидаем 2 оставшихся
    assert len(kept_polys) == 2


def test_locality_aware_nms():
    # Четыре прямоугольника в формате (n,9): [x0,y0,x1,y1,x2,y2,x3,y3,score]
    boxes = np.array(
        [
            [0, 0, 4, 0, 4, 4, 0, 4, 0.9],
            [1, 1, 5, 1, 5, 5, 1, 5, 0.8],
            [10, 10, 14, 10, 14, 14, 10, 14, 0.7],
            [11, 11, 15, 11, 15, 15, 11, 15, 0.6],
        ],
        dtype=np.float32,
    )
    iou_threshold = 0.1
    final_boxes = locality_aware_nms(boxes, iou_threshold)
    # Ожидаем 2 итоговых прямоугольника после слияния соседних пересечений
    assert final_boxes.shape[0] == 2


def test_locality_aware_nms_with_standard_threshold():
    # Тест с отдельным порогом для standard NMS
    boxes = np.array(
        [
            [0, 0, 4, 0, 4, 4, 0, 4, 0.9],
            [1, 1, 5, 1, 5, 5, 1, 5, 0.8],
            [10, 10, 14, 10, 14, 14, 10, 14, 0.7],
            [11, 11, 15, 11, 15, 15, 11, 15, 0.6],
        ],
        dtype=np.float32,
    )
    iou_threshold = 0.1
    iou_threshold_standard = 0.05  # Более строгий порог для standard NMS
    final_boxes = locality_aware_nms(boxes, iou_threshold, iou_threshold_standard)
    # Результат должен быть корректным массивом
    assert final_boxes.shape[1] == 9
    assert final_boxes.dtype == np.float32


def test_polygon_area_degenerate():
    # Менее трех точек => площадь должна быть 0
    poly = np.array([[0, 0], [1, 0]], dtype=np.float64)
    area = polygon_area(poly)
    assert area == pytest.approx(0.0)


def test_compute_intersection_parallel():
    # Параллельные отрезки => возвращает начальную точку
    p1 = np.array([0, 0], dtype=np.float64)
    p2 = np.array([1, 1], dtype=np.float64)
    A = np.array([2, 2], dtype=np.float64)
    B = np.array([3, 3], dtype=np.float64)
    inter = compute_intersection(p1, p2, A, B)
    np.testing.assert_allclose(inter, p1, rtol=1e-5)


def test_polygon_intersection_no_overlap():
    # Нет области пересечения => пустое пересечение
    poly1 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    poly2 = np.array([[2, 2], [3, 2], [3, 3], [2, 3]], dtype=np.float64)
    inter = polygon_intersection(poly1, poly2)
    assert inter.shape == (0, 2)


def test_polygon_iou_extremes():
    poly = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    # Полное пересечение => IoU=1.0
    assert polygon_iou(poly, poly) == pytest.approx(1.0)
    # Нет пересечения => IoU=0.0
    other = np.array([[2, 2], [3, 2], [3, 3], [2, 3]], dtype=np.float64)
    assert polygon_iou(poly, other) == pytest.approx(0.0)


def test_should_merge_at_threshold():
    # На границе порога
    poly = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    assert not should_merge(poly, poly, 1.0)
    assert should_merge(poly, poly, 0.999)


def test_clip_polygon_no_clip():
    # Линия далеко => нет отсечения
    subject = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    A = np.array([100, 0], dtype=np.float64)
    B = np.array([100, 1], dtype=np.float64)
    clipped, count = clip_polygon(subject, A, B)
    np.testing.assert_allclose(clipped, subject, rtol=1e-5)
    assert count == subject.shape[0]


def test_clip_polygon_full_clip():
    # Polygon entirely on one side => empty result
    subject = np.array([[1, 1], [2, 1], [2, 2], [1, 2]], dtype=np.float64)
    A = np.array([0, 0], dtype=np.float64)
    B = np.array([0, 1], dtype=np.float64)
    # Полигон справа от линии x=0 => все точки вне
    clipped, count = clip_polygon(subject, A, B)
    assert clipped.shape == (0, 2)
    assert count == 0


def test_normalize_polygon_variants():
    ref = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    variants = []
    # Все циклические сдвиги и обращения
    for start in range(4):
        variants.append(np.vstack([ref[(i + start) % 4] for i in range(4)]))
        variants.append(np.vstack([ref[(start - i) % 4] for i in range(4)]))
    for var in variants:
        norm = normalize_polygon(ref, var)
        np.testing.assert_allclose(norm, ref, rtol=1e-5)


# --- Дополнительные тесты для повышения покрытия ---


def test_standard_nms_empty_arrays():
    """Тест standard_nms с пустыми массивами"""
    polys = np.array([], dtype=np.float64)
    scores = np.array([], dtype=np.float64)
    
    kept_polys, kept_scores = standard_nms(polys, scores, 0.5)
    
    assert kept_polys.size == 0
    assert kept_scores.size == 0


def test_locality_aware_nms_empty():
    """Тест locality_aware_nms с пустым входом"""
    boxes = np.zeros((0, 9), dtype=np.float32)
    
    result = locality_aware_nms(boxes, 0.5)
    
    assert result.shape == (0, 9)
    assert result.dtype == np.float32


def test_locality_aware_nms_none_input():
    """Тест locality_aware_nms с None входом"""
    result = locality_aware_nms(None, 0.5)
    
    assert result.shape == (0, 9)
    assert result.dtype == np.float32


def test_standard_nms_score_ordering():
    """Тест что standard_nms сохраняет правильный порядок по score"""
    polys = [
        np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64),
        np.array([[100, 0], [110, 0], [110, 10], [100, 10]], dtype=np.float64),
        np.array([[200, 0], [210, 0], [210, 10], [200, 10]], dtype=np.float64),
    ]
    scores = [0.3, 0.9, 0.6]  # Не упорядочены
    
    kept_polys, kept_scores = standard_nms(polys, scores, 0.5)
    
    # Все 3 бокса не пересекаются, все должны остаться
    assert len(kept_polys) == 3
    # Первый в результате должен быть с наивысшим score
    assert kept_scores[0] == 0.9


def test_standard_nms_single_box():
    """Тест standard_nms с одним боксом"""
    polys = [np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)]
    scores = [0.9]
    
    kept_polys, kept_scores = standard_nms(polys, scores, 0.5)
    
    assert len(kept_polys) == 1
    assert kept_scores[0] == 0.9


def test_standard_nms_all_overlapping():
    """Тест standard_nms когда все боксы пересекаются"""
    # Все боксы примерно в одном месте
    polys = [
        np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64),
        np.array([[1, 1], [11, 1], [11, 11], [1, 11]], dtype=np.float64),
        np.array([[2, 2], [12, 2], [12, 12], [2, 12]], dtype=np.float64),
    ]
    scores = [0.9, 0.8, 0.7]
    
    kept_polys, kept_scores = standard_nms(polys, scores, iou_threshold=0.1)
    
    # Должен остаться только один (с наивысшим score)
    assert len(kept_polys) == 1
    assert kept_scores[0] == 0.9


def test_locality_aware_nms_single_box():
    """Тест locality_aware_nms с одним боксом"""
    boxes = np.array([[0, 0, 10, 0, 10, 10, 0, 10, 0.9]], dtype=np.float32)
    
    result = locality_aware_nms(boxes, 0.5)
    
    assert result.shape[0] == 1
    assert result.shape[1] == 9


def test_locality_aware_nms_merging_nearby_boxes():
    """Тест что locality_aware_nms объединяет близкие боксы"""
    # Два очень близких бокса с высоким IoU
    boxes = np.array(
        [
            [0, 0, 10, 0, 10, 10, 0, 10, 0.9],
            [1, 1, 11, 1, 11, 11, 1, 11, 0.85],
        ],
        dtype=np.float32,
    )
    
    result = locality_aware_nms(boxes, iou_threshold=0.3)
    
    # Должны объединиться в один
    assert result.shape[0] == 1


def test_locality_aware_nms_with_nan_protection():
    """Тест защиты от NaN в locality_aware_nms"""
    # Боксы с очень маленькими scores для проверки защиты от деления на ноль
    boxes = np.array(
        [
            [0, 0, 10, 0, 10, 10, 0, 10, 1e-10],
            [1, 1, 11, 1, 11, 11, 1, 11, 1e-10],
        ],
        dtype=np.float32,
    )
    
    result = locality_aware_nms(boxes, iou_threshold=0.3)
    
    # Не должно быть NaN значений
    assert np.isfinite(result).all()


def test_locality_aware_nms_score_max_preservation():
    """Тест что при объединении сохраняется максимальный score"""
    boxes = np.array(
        [
            [0, 0, 10, 0, 10, 10, 0, 10, 0.7],
            [1, 1, 11, 1, 11, 11, 1, 11, 0.95],  # Более высокий score
        ],
        dtype=np.float32,
    )
    
    result = locality_aware_nms(boxes, iou_threshold=0.3)
    
    # Объединенный бокс должен иметь максимальный score
    if result.shape[0] > 0:
        assert result[0, 8] >= 0.95 or np.isclose(result[0, 8], 0.95, rtol=0.01)


def test_polygon_area_large_polygon():
    """Тест площади большого полигона"""
    poly = np.array(
        [[0, 0], [1000, 0], [1000, 500], [0, 500]], dtype=np.float64
    )
    area = polygon_area(poly)
    expected = 1000 * 500  # 500000
    np.testing.assert_allclose(area, expected, rtol=1e-5)


def test_polygon_area_negative_coordinates():
    """Тест площади с отрицательными координатами"""
    poly = np.array([[-10, -10], [10, -10], [10, 10], [-10, 10]], dtype=np.float64)
    area = polygon_area(poly)
    expected = 20 * 20  # 400
    np.testing.assert_allclose(area, expected, rtol=1e-5)


def test_polygon_iou_partial_overlap():
    """Тест IoU с частичным пересечением"""
    poly1 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
    poly2 = np.array([[5, 5], [15, 5], [15, 15], [5, 15]], dtype=np.float64)
    
    iou = polygon_iou(poly1, poly2)
    
    # Пересечение 5x5 = 25, объединение = 100 + 100 - 25 = 175
    expected = 25 / 175
    assert np.isclose(iou, expected, rtol=1e-3)


def test_polygon_iou_one_inside_other():
    """Тест IoU когда один полигон внутри другого"""
    poly1 = np.array([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=np.float64)
    poly2 = np.array([[5, 5], [15, 5], [15, 15], [5, 15]], dtype=np.float64)
    
    iou = polygon_iou(poly1, poly2)
    
    # poly2 полностью внутри poly1
    # Пересечение = 100, poly1 = 400, poly2 = 100
    # IoU = 100 / 400 = 0.25
    assert iou > 0
    assert iou < 1


def test_polygon_intersection_complex_shape():
    """Тест пересечения сложных форм"""
    poly1 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
    poly2 = np.array([[5, -5], [15, -5], [15, 5], [5, 5]], dtype=np.float64)
    
    inter = polygon_intersection(poly1, poly2)
    
    # Должно быть пересечение
    assert inter.shape[0] > 0


def test_normalize_polygon_already_normalized():
    """Тест normalize_polygon с уже нормализованным полигоном"""
    ref = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
    poly = ref.copy()
    
    normalized = normalize_polygon(ref, poly)
    
    np.testing.assert_allclose(normalized, ref, rtol=1e-5)


def test_normalize_polygon_reversed():
    """Тест normalize_polygon с обратным порядком вершин"""
    ref = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
    poly = np.array([[0, 10], [10, 10], [10, 0], [0, 0]], dtype=np.float64)  # Обратный
    
    normalized = normalize_polygon(ref, poly)
    
    # После нормализации должен совпадать с ref
    np.testing.assert_allclose(normalized, ref, rtol=1e-5)


def test_should_merge_exactly_at_threshold():
    """Тест should_merge точно на пороге"""
    poly1 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
    poly2 = np.array([[5, 5], [15, 5], [15, 15], [5, 15]], dtype=np.float64)
    
    iou = polygon_iou(poly1, poly2)
    
    # Проверяем граничный случай
    assert should_merge(poly1, poly2, iou - 0.001)
    assert not should_merge(poly1, poly2, iou + 0.001)


def test_clip_polygon_partial_clip():
    """Тест частичного отсечения полигона"""
    subject = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
    # Отсечение по диагональной линии
    A = np.array([0, 5], dtype=np.float64)
    B = np.array([10, 5], dtype=np.float64)
    
    clipped, count = clip_polygon(subject, A, B)
    
    # Должен получиться новый полигон
    assert count > 0
    assert count <= 20  # Максимальный размер буфера


def test_compute_intersection_edge_case():
    """Тест compute_intersection для граничного случая"""
    p1 = np.array([0, 0], dtype=np.float64)
    p2 = np.array([10, 10], dtype=np.float64)
    A = np.array([0, 10], dtype=np.float64)
    B = np.array([10, 0], dtype=np.float64)
    
    inter = compute_intersection(p1, p2, A, B)
    
    # Должна быть точка пересечения в середине
    assert inter[0] > 0 and inter[0] < 10
    assert inter[1] > 0 and inter[1] < 10


def test_locality_aware_nms_far_apart_boxes():
    """Тест locality_aware_nms с далеко расположенными боксами"""
    boxes = np.array(
        [
            [0, 0, 10, 0, 10, 10, 0, 10, 0.9],
            [1000, 1000, 1010, 1000, 1010, 1010, 1000, 1010, 0.8],
        ],
        dtype=np.float32,
    )
    
    result = locality_aware_nms(boxes, iou_threshold=0.5)
    
    # Оба должны остаться (не пересекаются)
    assert result.shape[0] == 2


def test_locality_aware_nms_different_thresholds():
    """Тест locality_aware_nms с разными порогами"""
    boxes = np.array(
        [
            [0, 0, 10, 0, 10, 10, 0, 10, 0.9],
            [2, 2, 12, 2, 12, 12, 2, 12, 0.8],
            [4, 4, 14, 4, 14, 14, 4, 14, 0.7],
        ],
        dtype=np.float32,
    )
    
    # Низкий порог - больше объединений
    result_low = locality_aware_nms(boxes, iou_threshold=0.01)
    
    # Высокий порог - меньше объединений
    result_high = locality_aware_nms(boxes, iou_threshold=0.99)
    
    # С высоким порогом должно остаться больше боксов
    assert result_high.shape[0] >= result_low.shape[0]
