"""Common utilities for manuscript-ocr."""

# I/O utilities
from .io import read_image, tensor_to_image

# Geometry utilities
from .geometry import (
    box_iou,
    quad_iou,
)

# Visualization utilities
from .visualization import (
    draw_quads,
    visualize_page,
)

# Sorting and postprocessing utilities
from .sorting import resolve_intersections, sort_into_lines, segment_columns

# Metrics and evaluation utilities
from .metrics import (
    match_boxes,
    compute_f1_score,
    evaluate_dataset,
)

# Training utilities
from .training import set_seed


__all__ = [
    # I/O
    "read_image",
    "tensor_to_image",
    # Geometry
    "box_iou",
    "quad_iou",
    # Visualization
    "draw_quads",
    "draw_boxes",
    "visualize_page",
    # Sorting/Postprocessing
    "resolve_intersections",
    "sort_into_lines",
    "segment_columns",
    # Metrics
    "match_boxes",
    "compute_f1_score",
    "evaluate_dataset",
    # Training
    "set_seed",
]
