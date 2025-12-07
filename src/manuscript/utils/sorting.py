from typing import List, Tuple

import numpy as np

from ..data import Block, Line, Page, Word


def resolve_intersections(
    boxes: List[Tuple[float, float, float, float]],
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


def _find_gaps(boxes, start, end) -> List[int]:
    """Find centers of empty vertical regions within a range."""
    segs = [
        (max(b[0], start), min(b[2], end))
        for b in boxes
        if not (b[2] <= start or b[0] >= end)
    ]
    if not segs:
        return []
    segs.sort()
    merged = [segs[0]]
    for s, e in segs[1:]:
        ms, me = merged[-1]
        if s <= me:
            merged[-1] = (ms, max(me, e))
        else:
            merged.append((s, e))
    gaps = []
    prev_end = start
    for s, e in merged:
        if s > prev_end:
            gaps.append((prev_end, s))
        prev_end = e
    if prev_end < end:
        gaps.append((prev_end, end))
    return [(a + b) // 2 for a, b in gaps if b - a > 1]


def _emptiness(boxes, start, end) -> float:
    """Calculate emptiness ratio of a column region."""
    col = [b for b in boxes if b[0] >= start and b[2] <= end]
    if not col:
        return 1.0
    min_y, min_x = min(b[1] for b in col), max(b[3] for b in col)
    rect = (end - start) * (min_x - min_y)
    area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in col)
    return (rect - area) / rect if rect else 1.0


def segment_columns(
    boxes: List[Tuple[int, int, int, int]],
    max_splits: int = 10,
) -> List[List[Tuple[int, int, int, int]]]:
    """
    Segment boxes into columns using vertical gaps.

    Parameters
    ----------
    boxes : list of tuple
        List of boxes in format (x_min, y_min, x_max, y_max).
    max_splits : int, default=10
        Maximum number of column splits to detect.

    Returns
    -------
    list of list of tuple
        List of columns, where each column is a list of boxes.
        Columns are sorted left-to-right.

    Notes
    -----
    - Finds optimal vertical split points by analyzing empty regions
    - Filters out empty columns
    - Useful for multi-column document layouts
    """
    if not boxes:
        return []

    img_width = max(b[2] for b in boxes)
    segments = [(0, img_width)]
    separators: List[int] = []

    for _ in range(max_splits or img_width):
        best = None
        for idx, (s, e) in enumerate(segments):
            for x in _find_gaps(boxes, s, e):
                if not (
                    any(b[2] <= x and b[0] >= s for b in boxes)
                    and any(b[0] >= x and b[2] <= e for b in boxes)
                ):
                    continue
                score = _emptiness(boxes, s, x) + _emptiness(boxes, x, e)
                if best is None or score < best[0]:
                    best = (score, x, idx)
        if not best:
            break
        _, x_split, idx = best
        s, e = segments.pop(idx)
        separators.append(x_split)
        segments.insert(idx, (s, x_split))
        segments.insert(idx + 1, (x_split, e))
        segments.sort()

    parts = [(0, img_width)]
    for x in separators:
        new_parts: List[Tuple[int, int]] = []
        for s, e in parts:
            if s < x < e:
                new_parts += [(s, x), (x, e)]
            else:
                new_parts.append((s, e))
        parts = new_parts

    cols: List[List[Tuple[int, int, int, int]]] = []
    for s, e in parts:
        col = [b for b in boxes if b[0] >= s and b[2] <= e]
        cols.append(col)

    cols = [c for c in cols if c]
    if not cols:
        return []

    return sorted(cols, key=lambda c: min(b[0] for b in c))


def sort_into_lines(
    boxes: List[Tuple[float, float, float, float]],
    y_tol_ratio: float = 0.6,
    x_gap_ratio: float = np.inf,
    use_columns: bool = True,
    max_columns: int = 10,
) -> List[List[Tuple[float, float, float, float]]]:
    """
    Sort boxes into text lines with reading order, optionally segmenting into columns first.

    Groups boxes into lines based on vertical proximity, resolving any overlaps first.
    Each line contains boxes sorted left-to-right, and lines are ordered top-to-bottom.
    If use_columns=True, first segments boxes into columns, then sorts each column separately.

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
    use_columns : bool, default=True
        If True, first segment boxes into columns using vertical gaps, then sort
        each column independently. Each column becomes a separate block.
    max_columns : int, default=10
        Maximum number of columns to detect when use_columns=True.

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

    compressed = resolve_intersections(boxes)
    mapping = {c: o for c, o in zip(compressed, boxes)}

    if use_columns:
        columns = segment_columns(compressed, max_splits=max_columns)
        all_lines = []
        for column_boxes in columns:
            column_lines = _sort_column_into_lines(
                column_boxes, mapping, y_tol_ratio, x_gap_ratio
            )
            all_lines.extend(column_lines)
        return all_lines
    else:
        return _sort_column_into_lines(compressed, mapping, y_tol_ratio, x_gap_ratio)


def _sort_column_into_lines(
    compressed: List[Tuple[int, int, int, int]],
    mapping: dict,
    y_tol_ratio: float,
    x_gap_ratio: float,
) -> List[List[Tuple[float, float, float, float]]]:
    """Helper function to sort a single column of boxes into lines."""
    if not compressed:
        return []

    avg_h = np.mean([b[3] - b[1] for b in compressed])
    lines = []

    for b in sorted(compressed, key=lambda b: (b[1] + b[3]) / 2):
        cy = (b[1] + b[3]) / 2
        placed = False

        for ln in lines:
            line_cy = np.mean([(v[1] + v[3]) / 2 for v in ln])
            last_x1 = max(v[2] for v in ln)

            if (
                abs(cy - line_cy) <= avg_h * y_tol_ratio
                and (b[0] - last_x1) <= avg_h * x_gap_ratio
            ):
                ln.append(b)
                placed = True
                break

        if not placed:
            lines.append([b])

    lines.sort(key=lambda ln: np.mean([(b[1] + b[3]) / 2 for b in ln]))

    result = []
    for ln in lines:
        ln.sort(key=lambda b: b[0])
        original_line = [mapping[b] for b in ln]
        result.append(original_line)

    return result


def organize_page(
    page: Page,
    max_splits: int = 10,
    use_columns: bool = True,
) -> Page:
    """
    Organize words in a Page into structured Blocks, Lines, and reading order.

    Takes a Page with unorganized words and returns a new Page where:
    - Words are grouped into columns (Blocks)
    - Each Block contains Lines of Words
    - Words within Lines are ordered left-to-right
    - Lines within Blocks are ordered top-to-bottom
    - Blocks are ordered left-to-right (for columns)

    Parameters
    ----------
    page : Page
        Input Page object. Can contain either:
        - Words in unstructured blocks/lines
        - Direct list of words without proper organization
    max_splits : int, optional
        Maximum number of column splits to attempt when segmenting.
        Higher values allow more columns to be detected. Default is 10.
    use_columns : bool, optional
        If True, segments the page into columns (separate Blocks).
        If False, treats entire page as single column. Default is True.

    Returns
    -------
    Page
        New Page object with organized Blocks, Lines, and reading order set.

    Examples
    --------
    >>> from manuscript.detectors import EAST
    >>> from manuscript.utils import organize_page
    >>>
    >>> detector = EAST()
    >>> result = detector.predict("image.jpg", sort_reading_order=False)
    >>> page = result["page"]
    >>>
    >>> # Organize into structured reading order
    >>> organized_page = organize_page(page, max_splits=5)
    >>>
    >>> # Access first word in first line of first block
    >>> first_word = organized_page.blocks[0].lines[0].words[0]
    >>> print(f"Word order: {first_word.order}")

    Notes
    -----
    This function extracts all words from the input Page (regardless of their
    current organization), converts them to bounding boxes, performs column
    segmentation and line sorting, then rebuilds a clean Page structure.

    The function preserves all Word attributes (polygon, confidence, text, etc.)
    while updating the `order` field for reading sequence.
    """
    # Extract all words from the page
    all_words: List[Word] = []
    for block in page.blocks:
        for line in block.lines:
            all_words.extend(line.words)

    # If no words found, return empty page
    if not all_words:
        return Page(blocks=[Block(lines=[Line(words=[], order=0)], order=0)])

    # Convert words to boxes for sorting
    word_to_box = {}
    boxes = []
    for word in all_words:
        poly = np.array(word.polygon, dtype=np.int32)
        x_min, y_min = np.min(poly, axis=0)
        x_max, y_max = np.max(poly, axis=0)
        box = (int(x_min), int(y_min), int(x_max), int(y_max))
        boxes.append(box)
        word_to_box[box] = word

    # Segment into columns if requested
    if use_columns:
        columns = segment_columns(boxes, max_splits=max_splits)
    else:
        columns = [boxes]

    # Create blocks from columns
    blocks: List[Block] = []
    for block_idx, column_boxes in enumerate(columns):
        # Sort this column into lines
        lines_in_column = sort_into_lines(column_boxes, use_columns=False)

        # Convert to Line objects with ordered Words
        lines: List[Line] = []
        for line_idx, line_boxes in enumerate(lines_in_column):
            line_words = []
            for word_idx, box in enumerate(line_boxes):
                word = word_to_box[box]
                # Set word order within the line
                word.order = word_idx
                line_words.append(word)

            # Create Line with order
            line = Line(words=line_words, order=line_idx)
            lines.append(line)

        # Create Block for this column
        block = Block(lines=lines, order=block_idx)
        blocks.append(block)

    # Create Page with organized Blocks
    return Page(blocks=blocks)
