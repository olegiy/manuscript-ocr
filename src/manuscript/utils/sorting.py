from typing import List, Tuple

import numpy as np


def resolve_intersections(
    boxes: List[Tuple[float, float, float, float]]
) -> List[Tuple[float, float, float, float]]:
    """
    Resolve intersecting boxes by shrinking them iteratively.

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


def sort_into_lines(
    boxes: List[Tuple[float, float, float, float]],
    y_tol_ratio: float = 0.6,
    x_gap_ratio: float = np.inf,
) -> List[List[Tuple[float, float, float, float]]]:
    """
    Sort boxes into text lines with reading order.
    
    Groups boxes into lines based on vertical proximity, resolving any overlaps first.
    Each line contains boxes sorted left-to-right, and lines are ordered top-to-bottom.
    
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
    list of list of tuple
        List of lines, where each line is a list of boxes sorted left-to-right.
        Lines are sorted top-to-bottom. Original box coordinates are preserved.
        
    Examples
    --------
    >>> boxes = [(10, 10, 50, 30), (60, 10, 100, 30), (10, 50, 50, 70)]
    >>> lines = sort_into_lines(boxes)
    >>> len(lines)  # Two lines detected
    2
    >>> len(lines[0])  # First line has 2 boxes
    2
    >>> lines[0][0]  # First box in first line
    (10, 10, 50, 30)
    
    Notes
    -----
    - First resolves box overlaps by iterative shrinking
    - Groups boxes into lines based on vertical center proximity
    - Within each line, boxes are sorted left-to-right
    - Returns original (non-shrunk) boxes in grouped structure
    - Useful for converting flat detection output to line-based structure
    """
    if not boxes:
        return []
    
    # Resolve intersections
    compressed = resolve_intersections(boxes)
    
    # Create mapping from compressed to original
    mapping = {c: o for c, o in zip(compressed, boxes)}
    
    # Calculate average box height
    avg_h = np.mean([b[3] - b[1] for b in compressed])
    lines = []
    
    # Sort boxes by vertical position first
    for b in sorted(compressed, key=lambda b: (b[1] + b[3]) / 2):
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
    
    # Sort boxes within each line by horizontal position and map back to originals
    result = []
    for ln in lines:
        ln.sort(key=lambda b: b[0])
        original_line = [mapping[b] for b in ln]
        result.append(original_line)
    
    return result
