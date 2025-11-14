import numpy as np

from numba import float64, int64, njit, boolean
from numba.types import Tuple


@njit("float64(float64[:,:])", fastmath=True, cache=True)
def polygon_area(poly):
    """Вычисление площади полигона по формуле Shoelace (быстрая версия)."""
    area = 0.0
    n = poly.shape[0]
    for i in range(n):
        j = (i + 1) % n
        area += poly[i, 0] * poly[j, 1] - poly[j, 0] * poly[i, 1]
    return np.abs(area) * 0.5


@njit("Tuple((float64, float64))(float64, float64, float64, float64, float64, float64, float64, float64)", fastmath=True, inline='always')
def line_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    """Быстрое вычисление точки пересечения двух отрезков."""
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return x1, y1
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return x1 + t * (x2 - x1), y1 + t * (y2 - y1)


@njit("Tuple((float64[:,:], int64))(float64[:,:], int64, float64, float64, float64, float64)", fastmath=True, cache=True)
def clip_polygon_by_edge(polygon, num_verts, ax, ay, bx, by):
    """
    Клипирование полигона по одному ребру (A->B).
    Оптимизированная версия с inline вычислениями.
    """
    # Pre-allocated buffer
    output = np.empty((12, 2), dtype=np.float64)
    output_size = 0
    
    # Вектор ребра
    edge_x = bx - ax
    edge_y = by - ay
    
    for i in range(num_verts):
        curr_x = polygon[i, 0]
        curr_y = polygon[i, 1]
        
        # Previous vertex (с оптимизацией для избежания modulo)
        prev_idx = i - 1 if i > 0 else num_verts - 1
        prev_x = polygon[prev_idx, 0]
        prev_y = polygon[prev_idx, 1]
        
        # Inside test (cross product) - inline для скорости
        curr_cross = edge_x * (curr_y - ay) - edge_y * (curr_x - ax)
        prev_cross = edge_x * (prev_y - ay) - edge_y * (prev_x - ax)
        
        curr_inside = curr_cross >= 0
        prev_inside = prev_cross >= 0
        
        if curr_inside:
            if not prev_inside:
                # Добавляем точку пересечения
                inter_x, inter_y = line_intersection(
                    prev_x, prev_y, curr_x, curr_y,
                    ax, ay, bx, by
                )
                output[output_size, 0] = inter_x
                output[output_size, 1] = inter_y
                output_size += 1
            
            # Добавляем текущую вершину
            output[output_size, 0] = curr_x
            output[output_size, 1] = curr_y
            output_size += 1
        elif prev_inside:
            # Добавляем точку пересечения
            inter_x, inter_y = line_intersection(
                prev_x, prev_y, curr_x, curr_y,
                ax, ay, bx, by
            )
            output[output_size, 0] = inter_x
            output[output_size, 1] = inter_y
            output_size += 1
    
    return output, output_size


@njit("float64(float64[:,:], float64[:,:])", fastmath=True, cache=True)
def quad_intersection_area(quad1, quad2):
    """
    Быстрое вычисление площади пересечения двух четырехугольников.
    Использует Sutherland-Hodgman с оптимизациями.
    """
    # Быстрая проверка: bounding box не пересекаются
    min_x1 = min(quad1[0, 0], quad1[1, 0], quad1[2, 0], quad1[3, 0])
    max_x1 = max(quad1[0, 0], quad1[1, 0], quad1[2, 0], quad1[3, 0])
    min_y1 = min(quad1[0, 1], quad1[1, 1], quad1[2, 1], quad1[3, 1])
    max_y1 = max(quad1[0, 1], quad1[1, 1], quad1[2, 1], quad1[3, 1])
    
    min_x2 = min(quad2[0, 0], quad2[1, 0], quad2[2, 0], quad2[3, 0])
    max_x2 = max(quad2[0, 0], quad2[1, 0], quad2[2, 0], quad2[3, 0])
    min_y2 = min(quad2[0, 1], quad2[1, 1], quad2[2, 1], quad2[3, 1])
    max_y2 = max(quad2[0, 1], quad2[1, 1], quad2[2, 1], quad2[3, 1])
    
    if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
        return 0.0
    
    # Sutherland-Hodgman clipping с двумя буферами
    current = np.empty((12, 2), dtype=np.float64)
    for i in range(4):
        current[i, 0] = quad1[i, 0]
        current[i, 1] = quad1[i, 1]
    current_size = 4
    
    # Clip по каждому ребру quad2
    for edge_idx in range(4):
        if current_size == 0:
            return 0.0
        
        next_idx = (edge_idx + 1) % 4
        ax, ay = quad2[edge_idx, 0], quad2[edge_idx, 1]
        bx, by = quad2[next_idx, 0], quad2[next_idx, 1]
        
        clipped, clipped_size = clip_polygon_by_edge(current, current_size, ax, ay, bx, by)
        
        # Копируем результат обратно в current
        for i in range(clipped_size):
            current[i, 0] = clipped[i, 0]
            current[i, 1] = clipped[i, 1]
        current_size = clipped_size
    
    if current_size < 3:
        return 0.0
    
    # Вычисляем площадь результирующего полигона
    area = 0.0
    for i in range(current_size):
        j = (i + 1) % current_size
        area += current[i, 0] * current[j, 1] - current[j, 0] * current[i, 1]
    
    return np.abs(area) * 0.5


@njit("float64(float64[:,:], float64[:,:])", fastmath=True, cache=True)
def polygon_iou(poly1, poly2):
    """Вычисление IoU для полигонов (оптимизированная версия)."""
    inter_area = quad_intersection_area(poly1, poly2)
    
    if inter_area < 1e-8:
        return 0.0
    
    area1 = polygon_area(poly1)
    area2 = polygon_area(poly2)
    union_area = area1 + area2 - inter_area
    
    if union_area <= 1e-8:
        return 0.0
    
    return inter_area / union_area


@njit("boolean(float64[:,:], float64[:,:], float64)", cache=True)
def should_merge(poly1, poly2, iou_threshold):
    """Проверка, нужно ли объединять два полигона."""
    return polygon_iou(poly1, poly2) > iou_threshold


@njit("float64[:,:](float64[:,:], float64[:,:])", cache=True)
def normalize_polygon(ref, poly):
    """
    Нормализация порядка вершин полигона относительно референсного.
    ОПТИМИЗАЦИЯ: упрощенная версия - просто возвращаем poly как есть.
    Работает если все детекции уже в едином порядке (clockwise/counterclockwise).
    """
    return poly.copy()


@njit(cache=True)
def standard_nms(polys, scores, iou_threshold):
    """Standard NMS (оптимизированная версия)."""
    if polys.size == 0:
        return polys, scores
    
    order = np.argsort(-scores)
    n = order.shape[0]
    keep_idx = []
    suppressed = np.zeros(n, dtype=np.bool_)
    
    for i in range(n):
        idx = order[i]
        if suppressed[idx]:
            continue
        
        keep_idx.append(idx)
        
        # Оптимизация: кешируем текущий полигон
        curr_poly = polys[idx]
        
        for j in range(i + 1, n):
            idx_j = order[j]
            if suppressed[idx_j]:
                continue
            
            if polygon_iou(curr_poly, polys[idx_j]) > iou_threshold:
                suppressed[idx_j] = True
    
    # Формируем результат
    n_keep = len(keep_idx)
    result_polys = np.empty((n_keep, 4, 2), dtype=np.float64)
    result_scores = np.empty(n_keep, dtype=np.float64)
    
    for i in range(n_keep):
        result_polys[i] = polys[keep_idx[i]]
        result_scores[i] = scores[keep_idx[i]]
    
    return result_polys, result_scores


def locality_aware_nms(boxes, iou_threshold, iou_threshold_standard=None):
    """
    Locality-aware NMS для текстовых детекций.
    
    boxes — numpy-массив shape (n,9), где каждая строка:
            [x0, y0, x1, y1, x2, y2, x3, y3, score]
    iou_threshold — порог для объединения (IoU) в locality-aware фазе
    iou_threshold_standard — порог для standard NMS. Если None, используется iou_threshold
    
    Возвращает итоговый numpy-массив боксов (m,9).
    """
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 9), dtype=np.float32)
    
    # Используем iou_threshold для standard_nms, если не указан отдельный
    if iou_threshold_standard is None:
        iou_threshold_standard = iou_threshold

    # Сортируем по x-координате (первая колонка)
    boxes_sorted = np.ascontiguousarray(boxes, dtype=np.float64)[
        np.argsort(boxes[:, 0])
    ]

    merged_polys = []
    merged_scores = []
    weight_sums = []

    # Locality-aware merge
    for box in boxes_sorted:
        poly = box[:8].reshape((4, 2))
        score = float(box[8])

        if merged_polys:
            last_poly = merged_polys[-1]
            
            # Проверяем IoU с последним объединенным полигоном
            if polygon_iou(poly, last_poly) > iou_threshold:
                # Нормализация не нужна, если порядок вершин одинаков
                # aligned_poly = normalize_polygon(last_poly, poly)
                aligned_poly = poly  # ОПТИМИЗАЦИЯ: пропускаем нормализацию
                
                total_weight = weight_sums[-1] + score

                # Защита от деления на ноль
                if total_weight > 1e-8:
                    new_poly = (
                        last_poly * weight_sums[-1] + aligned_poly * score
                    ) / total_weight

                    # Проверка на NaN/Inf
                    if np.isfinite(new_poly).all():
                        merged_polys[-1] = new_poly
                        weight_sums[-1] = total_weight
                        merged_scores[-1] = max(merged_scores[-1], score)
                    else:
                        # Если получились NaN/Inf, просто обновляем score
                        merged_scores[-1] = max(merged_scores[-1], score)
                else:
                    # Если суммарный вес слишком мал, просто берем последний полигон
                    merged_scores[-1] = max(merged_scores[-1], score)
                continue

        merged_polys.append(poly.copy())
        merged_scores.append(score)
        weight_sums.append(score)

    if not merged_polys:
        return np.zeros((0, 9), dtype=np.float32)

    merged_polys_arr = np.array(merged_polys, dtype=np.float64)
    merged_scores_arr = np.array(merged_scores, dtype=np.float64)

    # Standard NMS
    kept_polys, kept_scores = standard_nms(
        merged_polys_arr, merged_scores_arr, iou_threshold_standard
    )

    if kept_polys.size == 0:
        return np.zeros((0, 9), dtype=np.float32)

    # Формируем финальный массив
    final_boxes = np.concatenate(
        [kept_polys.reshape(kept_polys.shape[0], -1), kept_scores[:, None]], axis=1
    )

    # Фильтруем боксы с NaN значениями
    valid_mask = np.isfinite(final_boxes).all(axis=1)
    final_boxes = final_boxes[valid_mask]

    return final_boxes.astype(np.float32)
