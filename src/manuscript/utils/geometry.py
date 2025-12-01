"""Geometric utilities for manuscript-ocr."""

from typing import Union, Tuple

import cv2
import numpy as np
from shapely.geometry import Polygon

try:
    import torch
except ImportError:
    torch = None


def box_iou(
    box1: Union[Tuple[float, float, float, float], np.ndarray],
    box2: Union[Tuple[float, float, float, float], np.ndarray]
) -> float:
    """
    Calculate Intersection over Union (IoU) for two axis-aligned bounding boxes.
    
    Parameters
    ----------
    box1 : tuple or np.ndarray
        First box as (x_min, y_min, x_max, y_max).
    box2 : tuple or np.ndarray
        Second box as (x_min, y_min, x_max, y_max).
        
    Returns
    -------
    float
        IoU value in range [0, 1]. Returns 0 if boxes don't intersect.
        
    Examples
    --------
    >>> box1 = (0, 0, 100, 100)
    >>> box2 = (50, 50, 150, 150)
    >>> iou = box_iou(box1, box2)
    >>> round(iou, 2)
    0.14
    
    >>> # No overlap
    >>> box3 = (200, 200, 300, 300)
    >>> box_iou(box1, box3)
    0.0
    
    Notes
    -----
    - Handles edge cases: zero-area boxes, no overlap
    - Commonly used in object detection metrics
    - For rotated boxes, use Shapely polygon IoU instead
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection rectangle
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    # Check if boxes intersect
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    # Calculate areas
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    
    # Handle zero union (shouldn't happen with valid boxes)
    if union_area <= 0:
        return 0.0
    
    return inter_area / union_area


def quad_iou(quad1: np.ndarray, quad2: np.ndarray) -> float:
    """
    Calculate IoU for two quadrilateral polygons using Shapely.
    
    This function is more accurate than box_iou for rotated or irregular boxes.
    
    Parameters
    ----------
    quad1 : np.ndarray
        First quad as array of 4 points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]].
    quad2 : np.ndarray
        Second quad in same format.
        
    Returns
    -------
    float
        IoU value in range [0, 1]. Returns 0 for invalid/non-overlapping polygons.
        
    Examples
    --------
    >>> quad1 = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
    >>> quad2 = np.array([[50, 50], [150, 50], [150, 150], [50, 150]])
    >>> iou = quad_iou(quad1, quad2)
    >>> round(iou, 2)
    0.14
    
    Notes
    -----
    - Handles rotated and arbitrary quadrilaterals
    - More computationally expensive than box_iou
    - Returns 0 for invalid polygons (self-intersecting, etc.)
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
        # Handle any Shapely errors (degenerate polygons, etc.)
        return 0.0
