"""Sorting and postprocessing utilities for manuscript-ocr."""

from typing import List, Tuple

import numpy as np


def resolve_intersections(
    boxes: List[Tuple[float, float, float, float]]
) -> List[Tuple[float, float, float, float]]:
    """
    Resolve intersecting boxes by shrinking them iteratively.
    
    This function is useful for cleaning up overlapping detections before
    applying reading order sorting.
    
    Parameters
    ----------
    boxes : list of tuple
        List of boxes in format (x_min, y_min, x_max, y_max).
        
    Returns
    -------
    list of tuple
        List of resolved boxes with reduced overlaps.
        
    Examples
    --------
    >>> boxes = [(10, 10, 55, 30), (50, 10, 100, 30)]  # Overlapping
    >>> resolved = resolve_intersections(boxes)
    >>> len(resolved)
    2
    
    Notes
    -----
    - Iteratively shrinks boxes by 10% when overlaps detected
    - Maximum 50 iterations to prevent infinite loops
    - May not eliminate all overlaps for complex cases
    """
    def intersect(b1, b2):
        """Check if two boxes intersect."""
        return not (
            b1[2] <= b2[0] or b2[2] <= b1[0] or b1[3] <= b2[1] or b2[3] <= b1[1]
        )
    
    resolved = list(boxes)
    max_iterations = 50
    
    for _ in range(max_iterations):
        changed = False
        for i in range(len(resolved)):
            for j in range(i + 1, len(resolved)):
                if intersect(resolved[i], resolved[j]):
                    # Shrink both boxes by 10%
                    x0, y0, x1, y1 = resolved[i]
                    x0b, y0b, x1b, y1b = resolved[j]
                    
                    resolved[i] = (
                        x0,
                        y0,
                        int(x1 - (x1 - x0) * 0.1),
                        int(y1 - (y1 - y0) * 0.1),
                    )
                    resolved[j] = (
                        x0b,
                        y0b,
                        int(x1b - (x1b - x0b) * 0.1),
                        int(y1b - (y1b - y0b) * 0.1),
                    )
                    changed = True
        
        if not changed:
            break
    
    return resolved


def sort_boxes_reading_order(
    boxes: List[Tuple[float, float, float, float]],
    y_tol_ratio: float = 0.6,
    x_gap_ratio: float = np.inf,
) -> List[Tuple[float, float, float, float]]:
    """
    Sort boxes in natural reading order (left-to-right, top-to-bottom).
    
    Groups boxes into lines based on vertical proximity, then sorts each line
    horizontally. This approximates the natural reading order for documents.
    
    Parameters
    ----------
    boxes : list of tuple
        List of boxes in format (x_min, y_min, x_max, y_max).
    y_tol_ratio : float, default=0.6
        Vertical tolerance as a ratio of average box height for grouping boxes
        into the same line. Boxes within this vertical distance are considered
        part of the same line.
    x_gap_ratio : float, default=np.inf
        Maximum horizontal gap as a ratio of average box height for boxes to be
        considered part of the same line. Use np.inf for no horizontal constraint.
        
    Returns
    -------
    list of tuple
        Boxes sorted in reading order.
        
    Examples
    --------
    >>> boxes = [(10, 10, 50, 30), (60, 10, 100, 30), (10, 50, 50, 70)]
    >>> sorted_boxes = sort_boxes_reading_order(boxes)
    >>> sorted_boxes[0]  # First box (top-left)
    (10, 10, 50, 30)
    
    >>> # With stricter horizontal gap constraint
    >>> sorted_boxes = sort_boxes_reading_order(boxes, x_gap_ratio=2.0)
    
    Notes
    -----
    - Lines are identified by vertical center proximity
    - Within each line, boxes are sorted left-to-right
    - Useful for OCR postprocessing to get reading order
    """
    if not boxes:
        return []
    
    # Calculate average box height
    avg_h = np.mean([b[3] - b[1] for b in boxes])
    lines = []
    
    # Sort boxes by vertical position first
    for b in sorted(boxes, key=lambda b: (b[1] + b[3]) / 2):
        cy = (b[1] + b[3]) / 2  # Center Y
        placed = False
        
        # Try to add to existing line
        for ln in lines:
            line_cy = np.mean([(v[1] + v[3]) / 2 for v in ln])
            last_x1 = max(v[2] for v in ln)
            
            # Check vertical proximity and horizontal gap
            if (
                abs(cy - line_cy) <= avg_h * y_tol_ratio
                and (b[0] - last_x1) <= avg_h * x_gap_ratio
            ):
                ln.append(b)
                placed = True
                break
        
        # Create new line if not placed
        if not placed:
            lines.append([b])
    
    # Sort lines by vertical position
    lines.sort(key=lambda ln: np.mean([(b[1] + b[3]) / 2 for b in ln]))
    
    # Sort boxes within each line by horizontal position
    for ln in lines:
        ln.sort(key=lambda b: b[0])
    
    # Flatten to single list
    return [b for ln in lines for b in ln]


def sort_boxes_reading_order_with_resolutions(
    boxes: List[Tuple[float, float, float, float]],
    y_tol_ratio: float = 0.6,
    x_gap_ratio: float = np.inf,
) -> List[Tuple[float, float, float, float]]:
    """
    Sort boxes in reading order after resolving intersections.
    
    This function first resolves overlapping boxes by shrinking them, then applies
    reading order sorting. Useful when boxes may overlap slightly.
    
    Parameters
    ----------
    boxes : list of tuple
        List of boxes in format (x_min, y_min, x_max, y_max).
    y_tol_ratio : float, default=0.6
        Vertical tolerance for line grouping (see sort_boxes_reading_order).
    x_gap_ratio : float, default=np.inf
        Maximum horizontal gap for line continuity (see sort_boxes_reading_order).
        
    Returns
    -------
    list of tuple
        Original boxes sorted in reading order (intersections resolved internally).
        
    Examples
    --------
    >>> boxes = [(10, 10, 55, 30), (50, 10, 100, 30)]  # Overlapping
    >>> sorted_boxes = sort_boxes_reading_order_with_resolutions(boxes)
    >>> sorted_boxes[0]
    (10, 10, 55, 30)
    
    Notes
    -----
    - First resolves overlaps by iterative shrinking
    - Then applies reading order sorting
    - Returns original (non-shrunk) boxes in sorted order
    - Maintains original box dimensions in output
    """
    # Resolve intersections
    compressed = resolve_intersections(boxes)
    
    # Create mapping from compressed to original
    mapping = {c: o for c, o in zip(compressed, boxes)}
    
    # Sort compressed boxes
    sorted_compressed = sort_boxes_reading_order(
        compressed, y_tol_ratio=y_tol_ratio, x_gap_ratio=x_gap_ratio
    )
    
    # Return original boxes in sorted order
    return [mapping[b] for b in sorted_compressed]
