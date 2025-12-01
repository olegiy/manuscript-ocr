import numpy as np
import pytest

from manuscript.detectors._east.lanms import (
    polygon_area,
    line_intersection,
    clip_polygon_by_edge,
    quad_intersection_area,
    polygon_iou,
    standard_nms,
    locality_aware_nms,
)


def should_merge(poly1, poly2, threshold):
    """Helper function: check if two polygons should merge based on IoU threshold."""
    return polygon_iou(poly1, poly2) > threshold


def _clip_polygon(subject, point_a, point_b):
    """Helper to call clip_polygon_by_edge with a padded buffer."""
    buffer = np.zeros((12, 2), dtype=np.float64)
    count = subject.shape[0]
    buffer[:count] = subject
    clipped, clipped_count = clip_polygon_by_edge(
        buffer,
        count,
        float(point_a[0]),
        float(point_a[1]),
        float(point_b[0]),
        float(point_b[1]),
    )
    return clipped[:clipped_count], clipped_count


@pytest.mark.skip(reason="Временно отключено")
def test_polygon_area_square():
    poly = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    area = polygon_area(poly)
    np.testing.assert_allclose(area, 1.0, rtol=1e-5)

@pytest.mark.skip(reason="Временно отключено")
def test_polygon_area_triangle():
    poly = np.array([[0, 0], [2, 0], [0, 2]], dtype=np.float64)
    area = polygon_area(poly)
    np.testing.assert_allclose(area, 2.0, rtol=1e-5)

@pytest.mark.skip(reason="Временно отключено")
def test_line_intersection():
    # ����������� ���� ��������, ������ ������� ����� (1,1)
    p1 = np.array([0.0, 0.0], dtype=np.float64)
    p2 = np.array([2.0, 2.0], dtype=np.float64)
    A = np.array([0.0, 2.0], dtype=np.float64)
    B = np.array([2.0, 0.0], dtype=np.float64)
    inter = np.array(line_intersection(*p1, *p2, *A, *B), dtype=np.float64)
    np.testing.assert_allclose(inter, np.array([1.0, 1.0], dtype=np.float64), rtol=1e-5)

@pytest.mark.skip(reason="Временно отключено")
def test_clip_polygon():
    subject = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=np.float64)
    # Отсечение по линии x = 2
    A = np.array([2, 5], dtype=np.float64)
    B = np.array([2, -1], dtype=np.float64)
    clipped, count = _clip_polygon(subject, A, B)
    expected = np.array([[2, 0], [4, 0], [4, 4], [2, 4]], dtype=np.float64)
    np.testing.assert_allclose(clipped, expected, rtol=1e-5)
    assert count == 4

@pytest.mark.skip(reason="Временно отключено")
def test_quad_intersection_area_overlap():
    poly1 = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=np.float64)
    poly2 = np.array([[2, 2], [6, 2], [6, 6], [2, 6]], dtype=np.float64)
    area = quad_intersection_area(poly1, poly2)
    expected = 4.0
    np.testing.assert_allclose(area, expected, rtol=1e-5)

@pytest.mark.skip(reason="Временно отключено")
def test_polygon_iou():
    poly1 = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=np.float64)
    poly2 = np.array([[2, 2], [6, 2], [6, 6], [2, 6]], dtype=np.float64)
    iou = polygon_iou(poly1, poly2)
    expected = 4 / (16 + 16 - 4)  # 4 / 28 ~ 0.142857
    assert np.isclose(iou, expected, rtol=1e-5)

@pytest.mark.skip(reason="Временно отключено")
def test_should_merge():
    poly1 = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=np.float64)
    poly2 = np.array([[2, 2], [6, 2], [6, 6], [2, 6]], dtype=np.float64)
    # Порог IoU 0.1 (0.142857 > 0.1) должен вернуть True
    assert should_merge(poly1, poly2, 0.1)
    # Порог IoU 0.2 (0.142857 < 0.2) должен вернуть False
    assert not should_merge(poly1, poly2, 0.2)

@pytest.mark.skip(reason="normalize_polygon function removed - obsolete test")
def test_normalize_polygon_returns_copy():
    ref = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=np.float64)
    poly = np.array([[4, 4], [0, 4], [0, 0], [4, 0]], dtype=np.float64)
    normalized = normalize_polygon(ref, poly)
    np.testing.assert_allclose(normalized, poly, rtol=1e-5)
    assert normalized is not poly


@pytest.mark.skip(reason="normalize_polygon function removed - obsolete test")
def test_normalize_polygon_keeps_input_intact():
    ref = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    poly = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float64)
    original = poly.copy()
    normalized = normalize_polygon(ref, poly)
    np.testing.assert_allclose(normalized, poly, rtol=1e-5)
    np.testing.assert_allclose(poly, original, rtol=1e-5)


@pytest.mark.skip(reason="Временно отключено")
def test_standard_nms():
    # Три прямоугольника: два пересекаются, один нет
    polys = np.array(
        [
            [[0, 0], [4, 0], [4, 4], [0, 4]],
            [[1, 1], [5, 1], [5, 5], [1, 5]],
            [[10, 10], [14, 10], [14, 14], [10, 14]],
        ],
        dtype=np.float64,
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float64)
    iou_threshold = 0.1
    kept_polys, kept_scores = standard_nms(polys, scores, iou_threshold)
    # Первые два прямоугольника пересекаются, третий нет => ожидаем 2 оставшихся
    assert len(kept_polys) == 2

@pytest.mark.skip(reason="Временно отключено")
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

@pytest.mark.skip(reason="Временно отключено")
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

@pytest.mark.skip(reason="Временно отключено")
def test_polygon_area_degenerate():
    # Менее трех точек => площадь должна быть 0
    poly = np.array([[0, 0], [1, 0]], dtype=np.float64)
    area = polygon_area(poly)
    assert area == pytest.approx(0.0)

@pytest.mark.skip(reason="Временно отключено")
def test_line_intersection_parallel_lines():
    # ������������ ������� => ���������� ��������� �����
    p1 = np.array([0.0, 0.0], dtype=np.float64)
    p2 = np.array([1.0, 1.0], dtype=np.float64)
    A = np.array([2.0, 2.0], dtype=np.float64)
    B = np.array([3.0, 3.0], dtype=np.float64)
    inter = np.array(line_intersection(*p1, *p2, *A, *B), dtype=np.float64)
    np.testing.assert_allclose(inter, p1, rtol=1e-5)

@pytest.mark.skip(reason="Временно отключено")
def test_quad_intersection_area_no_overlap():
    poly1 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    poly2 = np.array([[2, 2], [3, 2], [3, 3], [2, 3]], dtype=np.float64)
    area = quad_intersection_area(poly1, poly2)
    assert area == pytest.approx(0.0)
@pytest.mark.skip(reason="Временно отключено")
def test_polygon_iou_extremes():
    poly = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    # Полное пересечение => IoU=1.0
    assert polygon_iou(poly, poly) == pytest.approx(1.0)
    # Нет пересечения => IoU=0.0
    other = np.array([[2, 2], [3, 2], [3, 3], [2, 3]], dtype=np.float64)
    assert polygon_iou(poly, other) == pytest.approx(0.0)

@pytest.mark.skip(reason="Временно отключено")
def test_should_merge_at_threshold():
    # На границе порога
    poly = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    assert not should_merge(poly, poly, 1.0)
    assert should_merge(poly, poly, 0.999)

@pytest.mark.skip(reason="Временно отключено")
def test_clip_polygon_no_clip():
    # Линия далеко => нет отсечения
    subject = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    A = np.array([100, 0], dtype=np.float64)
    B = np.array([100, 1], dtype=np.float64)
    clipped, count = _clip_polygon(subject, A, B)
    np.testing.assert_allclose(clipped, subject, rtol=1e-5)
    assert count == subject.shape[0]


@pytest.mark.skip(reason="Временно отключено")
def test_clip_polygon_full_clip():
    # Polygon entirely on one side => empty result
    subject = np.array([[1, 1], [2, 1], [2, 2], [1, 2]], dtype=np.float64)
    A = np.array([0, 0], dtype=np.float64)
    B = np.array([0, 1], dtype=np.float64)
    # Полигон справа от линии x=0 => все точки вне
    clipped, count = _clip_polygon(subject, A, B)
    assert clipped.shape == (0, 2)
    assert count == 0


@pytest.mark.skip(reason="normalize_polygon function removed - obsolete test")
def test_normalize_polygon_variants():
    ref = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    variants = []
    # ??? ??????????? ?????? ? ?????????
    for start in range(4):
        variants.append(np.vstack([ref[(i + start) % 4] for i in range(4)]))
        variants.append(np.vstack([ref[(start - i) % 4] for i in range(4)]))
    for var in variants:
        norm = normalize_polygon(ref, var)
        np.testing.assert_allclose(norm, var, rtol=1e-5)
        assert norm is not var

@pytest.mark.skip(reason="Временно отключено")
def test_standard_nms_empty_arrays():
    """Тест standard_nms с пустыми массивами"""
    polys = np.zeros((0, 4, 2), dtype=np.float64)
    scores = np.array([], dtype=np.float64)
    
    kept_polys, kept_scores = standard_nms(polys, scores, 0.5)
    
    assert kept_polys.size == 0
    assert kept_scores.size == 0

@pytest.mark.skip(reason="Временно отключено")
def test_locality_aware_nms_empty():
    """Тест locality_aware_nms с пустым входом"""
    boxes = np.zeros((0, 9), dtype=np.float32)
    
    result = locality_aware_nms(boxes, 0.5)
    
    assert result.shape == (0, 9)
    assert result.dtype == np.float32

@pytest.mark.skip(reason="Временно отключено")
def test_locality_aware_nms_none_input():
    """Тест locality_aware_nms с None входом"""
    result = locality_aware_nms(None, 0.5)
    
    assert result.shape == (0, 9)
    assert result.dtype == np.float32

@pytest.mark.skip(reason="Временно отключено")
def test_standard_nms_score_ordering():
    """???? ??? standard_nms ????????? ?????????? ??????? ?? score"""
    polys = np.array(
        [
            [[0, 0], [10, 0], [10, 10], [0, 10]],
            [[100, 0], [110, 0], [110, 10], [100, 10]],
            [[200, 0], [210, 0], [210, 10], [200, 10]],
        ],
        dtype=np.float64,
    )
    scores = np.array([0.3, 0.9, 0.6], dtype=np.float64)  # ?? ???????????
    
    kept_polys, kept_scores = standard_nms(polys, scores, 0.5)
    
    # ??? 3 ????? ?? ????????????, ??? ?????? ????????
    assert len(kept_polys) == 3
    # ?????? ? ?????????? ?????? ???? ? ????????? score
    assert kept_scores[0] == 0.9

@pytest.mark.skip(reason="Временно отключено")
def test_standard_nms_single_box():
    """???? standard_nms ? ????? ??????"""
    polys = np.array([[[0, 0], [10, 0], [10, 10], [0, 10]]], dtype=np.float64)
    scores = np.array([0.9], dtype=np.float64)
    
    kept_polys, kept_scores = standard_nms(polys, scores, 0.5)
    
    assert len(kept_polys) == 1
    assert kept_scores[0] == 0.9

@pytest.mark.skip(reason="Временно отключено")
def test_standard_nms_all_overlapping():
    """???? standard_nms ????? ??? ????? ????????????"""
    # ??? ????? ???????? ? ????? ?????
    polys = np.array(
        [
            [[0, 0], [10, 0], [10, 10], [0, 10]],
            [[1, 1], [11, 1], [11, 11], [1, 11]],
            [[2, 2], [12, 2], [12, 12], [2, 12]],
        ],
        dtype=np.float64,
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float64)
    
    kept_polys, kept_scores = standard_nms(polys, scores, iou_threshold=0.1)
    
    # ?????? ???????? ?????? ???? (? ????????? score)
    assert len(kept_polys) == 1
    assert kept_scores[0] == 0.9

@pytest.mark.skip(reason="Временно отключено")
def test_locality_aware_nms_single_box():
    """Тест locality_aware_nms с одним боксом"""
    boxes = np.array([[0, 0, 10, 0, 10, 10, 0, 10, 0.9]], dtype=np.float32)
    
    result = locality_aware_nms(boxes, 0.5)
    
    assert result.shape[0] == 1
    assert result.shape[1] == 9

@pytest.mark.skip(reason="Временно отключено")
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

@pytest.mark.skip(reason="Временно отключено")
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

@pytest.mark.skip(reason="Временно отключено")
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

@pytest.mark.skip(reason="Временно отключено")
def test_polygon_area_large_polygon():
    """Тест площади большого полигона"""
    poly = np.array(
        [[0, 0], [1000, 0], [1000, 500], [0, 500]], dtype=np.float64
    )
    area = polygon_area(poly)
    expected = 1000 * 500  # 500000
    np.testing.assert_allclose(area, expected, rtol=1e-5)

@pytest.mark.skip(reason="Временно отключено")
def test_polygon_area_negative_coordinates():
    """Тест площади с отрицательными координатами"""
    poly = np.array([[-10, -10], [10, -10], [10, 10], [-10, 10]], dtype=np.float64)
    area = polygon_area(poly)
    expected = 20 * 20  # 400
    np.testing.assert_allclose(area, expected, rtol=1e-5)

@pytest.mark.skip(reason="Временно отключено")
def test_polygon_iou_partial_overlap():
    """Тест IoU с частичным пересечением"""
    poly1 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
    poly2 = np.array([[5, 5], [15, 5], [15, 15], [5, 15]], dtype=np.float64)
    
    iou = polygon_iou(poly1, poly2)
    
    # Пересечение 5x5 = 25, объединение = 100 + 100 - 25 = 175
    expected = 25 / 175
    assert np.isclose(iou, expected, rtol=1e-3)

@pytest.mark.skip(reason="Временно отключено")
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

@pytest.mark.skip(reason="Временно отключено")
def test_quad_intersection_area_complex_shape():
    """???? ??????????? ??????? ????"""
    poly1 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
    poly2 = np.array([[5, -5], [15, -5], [15, 5], [5, 5]], dtype=np.float64)
    area = quad_intersection_area(poly1, poly2)
    assert area > 0.0

@pytest.mark.skip(reason="normalize_polygon function removed - obsolete test")
def test_normalize_polygon_already_normalized():
    """???? normalize_polygon ? ??? ??????????????? ?????????"""
    ref = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
    poly = ref.copy()
    normalized = normalize_polygon(ref, poly)
    np.testing.assert_allclose(normalized, poly, rtol=1e-5)
    assert normalized is not poly

@pytest.mark.skip(reason="normalize_polygon function removed - obsolete test")
def test_normalize_polygon_reversed():
    """???? normalize_polygon ? ???????? ???????? ??????"""
    ref = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
    poly = np.array([[0, 10], [10, 10], [10, 0], [0, 0]], dtype=np.float64)
    normalized = normalize_polygon(ref, poly)
    np.testing.assert_allclose(normalized, poly, rtol=1e-5)

@pytest.mark.skip(reason="Временно отключено")
def test_should_merge_exactly_at_threshold():
    """Тест should_merge точно на пороге"""
    poly1 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
    poly2 = np.array([[5, 5], [15, 5], [15, 15], [5, 15]], dtype=np.float64)
    
    iou = polygon_iou(poly1, poly2)
    
    # Проверяем граничный случай
    assert should_merge(poly1, poly2, iou - 0.001)
    assert not should_merge(poly1, poly2, iou + 0.001)

@pytest.mark.skip(reason="Временно отключено")
def test_clip_polygon_partial_clip():
    """Тест частичного отсечения полигона"""
    subject = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
    # Отсечение по диагональной линии
    A = np.array([0, 5], dtype=np.float64)
    B = np.array([10, 5], dtype=np.float64)
    
    clipped, count = _clip_polygon(subject, A, B)
    
    # Должен получиться новый полигон
    assert count > 0
    assert count <= 20  # Максимальный размер буфера

@pytest.mark.skip(reason="Временно отключено")
def test_line_intersection_edge_case():
    """???? line_intersection ??? ?????????? ??????"""
    p1 = np.array([0.0, 0.0], dtype=np.float64)
    p2 = np.array([10.0, 10.0], dtype=np.float64)
    A = np.array([0.0, 10.0], dtype=np.float64)
    B = np.array([10.0, 0.0], dtype=np.float64)

    inter = np.array(line_intersection(*p1, *p2, *A, *B), dtype=np.float64)

    # ?????? ???? ????? ??????????? ? ????????
    assert 0.0 < inter[0] < 10.0
    assert 0.0 < inter[1] < 10.0

@pytest.mark.skip(reason="Временно отключено")
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

@pytest.mark.skip(reason="Временно отключено")
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
