from typing import Union, Tuple

import numpy as np
from shapely.geometry import Polygon


def box_iou(
    box1: Union[Tuple[float, float, float, float], np.ndarray],
    box2: Union[Tuple[float, float, float, float], np.ndarray]
) -> float:
    """
    Calculate Intersection over Union (IoU) for two axis-aligned bounding boxes.
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return inter_area / union_area

# нужна ли функция
def quad_iou(quad1: np.ndarray, quad2: np.ndarray) -> float:
    """
    Calculate IoU for two quadrilateral polygons using Shapely.
    """
    try:
        poly1 = Polygon(quad1)
        poly2 = Polygon(quad2)
        
        if not poly1.is_valid or not poly2.is_valid:
            return 0.0
        
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        
        if union <= 0:
            return 0.0
        
        return intersection / union
    except Exception:
        return 0.0
