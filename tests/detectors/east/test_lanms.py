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


def test_polygon_area_square():
    poly = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    area = polygon_area(poly)
    np.testing.assert_allclose(area, 1.0, rtol=1e-5)

def test_polygon_area_triangle():
    poly = np.array([[0, 0], [2, 0], [0, 2]], dtype=np.float64)
    area = polygon_area(poly)
    np.testing.assert_allclose(area, 2.0, rtol=1e-5)

def test_line_intersection():
    # Intersecting lines: expect intersection point (1, 1)
    p1 = np.array([0.0, 0.0], dtype=np.float64)
    p2 = np.array([2.0, 2.0], dtype=np.float64)
    A = np.array([0.0, 2.0], dtype=np.float64)
    B = np.array([2.0, 0.0], dtype=np.float64)
    inter = np.array(line_intersection(*p1, *p2, *A, *B), dtype=np.float64)
    np.testing.assert_allclose(inter, np.array([1.0, 1.0], dtype=np.float64), rtol=1e-5)

def test_clip_polygon():
    subject = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=np.float64)
    # Clipping by the line x = 2
    A = np.array([2, 5], dtype=np.float64)
    B = np.array([2, -1], dtype=np.float64)
    clipped, count = _clip_polygon(subject, A, B)
    expected = np.array([[2, 0], [4, 0], [4, 4], [2, 4]], dtype=np.float64)
    np.testing.assert_allclose(clipped, expected, rtol=1e-5)
    assert count == 4

def test_quad_intersection_area_overlap():
    poly1 = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=np.float64)
    poly2 = np.array([[2, 2], [6, 2], [6, 6], [2, 6]], dtype=np.float64)
    area = quad_intersection_area(poly1, poly2)
    expected = 4.0
    np.testing.assert_allclose(area, expected, rtol=1e-5)

def test_polygon_iou():
    poly1 = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=np.float64)
    poly2 = np.array([[2, 2], [6, 2], [6, 6], [2, 6]], dtype=np.float64)
    iou = polygon_iou(poly1, poly2)
    expected = 4 / (16 + 16 - 4)  # 4 / 28 ~ 0.142857
    assert np.isclose(iou, expected, rtol=1e-5)

def test_should_merge():
    poly1 = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=np.float64)
    poly2 = np.array([[2, 2], [6, 2], [6, 6], [2, 6]], dtype=np.float64)
    # IoU threshold 0.1 (0.142857 > 0.1) should return True
    assert should_merge(poly1, poly2, 0.1)
    # IoU threshold 0.2 (0.142857 < 0.2) should return False
    assert not should_merge(poly1, poly2, 0.2)

def test_standard_nms():
    # Three rectangles: two overlap, one does not
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
    # The first two rectangles overlap, the third does not => expect 2 remaining
    assert len(kept_polys) == 2

def test_locality_aware_nms():
    # Four rectangles in (n,9) format: [x0,y0,x1,y1,x2,y2,x3,y3,score]
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
    # Expect 2 final rectangles after merging neighboring intersections
    assert final_boxes.shape[0] == 2

def test_locality_aware_nms_with_standard_threshold():
    # Test with a separate threshold for standard NMS
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
    iou_threshold_standard = 0.05  # More strict threshold for standard NMS
    final_boxes = locality_aware_nms(boxes, iou_threshold, iou_threshold_standard)
    # Result should be a correct array
    assert final_boxes.shape[1] == 9
    assert final_boxes.dtype == np.float32

def test_polygon_area_degenerate():
    # Less than three points => area should be 0
    poly = np.array([[0, 0], [1, 0]], dtype=np.float64)
    area = polygon_area(poly)
    assert area == pytest.approx(0.0)

def test_line_intersection_parallel_lines():
    # Parallel lines => no intersection point
    p1 = np.array([0.0, 0.0], dtype=np.float64)
    p2 = np.array([1.0, 1.0], dtype=np.float64)
    A = np.array([2.0, 2.0], dtype=np.float64)
    B = np.array([3.0, 3.0], dtype=np.float64)
    inter = np.array(line_intersection(*p1, *p2, *A, *B), dtype=np.float64)
    np.testing.assert_allclose(inter, p1, rtol=1e-5)

def test_quad_intersection_area_no_overlap():
    poly1 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    poly2 = np.array([[2, 2], [3, 2], [3, 3], [2, 3]], dtype=np.float64)
    area = quad_intersection_area(poly1, poly2)
    assert area == pytest.approx(0.0)

def test_polygon_iou_extremes():
    poly = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    # Full overlap => IoU=1.0
    assert polygon_iou(poly, poly) == pytest.approx(1.0)
    # No overlap => IoU=0.0
    other = np.array([[2, 2], [3, 2], [3, 3], [2, 3]], dtype=np.float64)
    assert polygon_iou(poly, other) == pytest.approx(0.0)

def test_should_merge_at_threshold():
    # At the threshold boundary
    poly = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    assert not should_merge(poly, poly, 1.0)
    assert should_merge(poly, poly, 0.999)

def test_clip_polygon_no_clip():
    # Line far away => no clipping
    subject = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    A = np.array([100, 0], dtype=np.float64)
    B = np.array([100, 1], dtype=np.float64)
    clipped, count = _clip_polygon(subject, A, B)
    np.testing.assert_allclose(clipped, subject, rtol=1e-5)
    assert count == subject.shape[0]

def test_clip_polygon_full_clip():
    # Polygon entirely on one side => empty result
    subject = np.array([[1, 1], [2, 1], [2, 2], [1, 2]], dtype=np.float64)
    A = np.array([0, 0], dtype=np.float64)
    B = np.array([0, 1], dtype=np.float64)
    
    # Polygon to the right of the line x=0 => all points outside
    clipped, count = _clip_polygon(subject, A, B)
    assert clipped.shape == (0, 2)
    assert count == 0

def test_standard_nms_empty_arrays():
    """Test standard_nms with empty массивами"""
    polys = np.zeros((0, 4, 2), dtype=np.float64)
    scores = np.array([], dtype=np.float64)
    
    kept_polys, kept_scores = standard_nms(polys, scores, 0.5)
    
    assert kept_polys.size == 0
    assert kept_scores.size == 0

def test_locality_aware_nms_empty():
    """Test locality_aware_nms с пустым входом"""
    boxes = np.zeros((0, 9), dtype=np.float32)
    
    result = locality_aware_nms(boxes, 0.5)
    
    assert result.shape == (0, 9)
    assert result.dtype == np.float32

def test_locality_aware_nms_none_input():
    """Test locality_aware_nms с None входом"""
    result = locality_aware_nms(None, 0.5)
    
    assert result.shape == (0, 9)
    assert result.dtype == np.float32

def test_standard_nms_score_ordering():
    """Test that standard_nms returns polygons sorted by score"""
    polys = np.array(
        [
            [[0, 0], [10, 0], [10, 10], [0, 10]],
            [[100, 0], [110, 0], [110, 10], [100, 10]],
            [[200, 0], [210, 0], [210, 10], [200, 10]],
        ],
        dtype=np.float64,
    )
    scores = np.array([0.3, 0.9, 0.6], dtype=np.float64)  # Scores for each polygon
    
    kept_polys, kept_scores = standard_nms(polys, scores, 0.5)
    
    # Expect 3 polygons to be kept since they do not overlap significantly
    assert len(kept_polys) == 3
    # The first polygon should have the highest score
    assert kept_scores[0] == 0.9

def test_standard_nms_single_box():
    """Test standard_nms with a single box"""
    polys = np.array([[[0, 0], [10, 0], [10, 10], [0, 10]]], dtype=np.float64)
    scores = np.array([0.9], dtype=np.float64)
    
    kept_polys, kept_scores = standard_nms(polys, scores, 0.5)
    
    assert len(kept_polys) == 1
    assert kept_scores[0] == 0.9

def test_standard_nms_all_overlapping():
    """Test standard_nms with all overlapping boxes"""
    # Three overlapping boxes with different scores
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
    
    # Expect only one polygon to be kept (the one with the highest score)
    assert len(kept_polys) == 1
    assert kept_scores[0] == 0.9

def test_locality_aware_nms_single_box():
    """Test locality_aware_nms with a single box"""
    boxes = np.array([[0, 0, 10, 0, 10, 10, 0, 10, 0.9]], dtype=np.float32)
    
    result = locality_aware_nms(boxes, 0.5)
    
    assert result.shape[0] == 1
    assert result.shape[1] == 9

def test_locality_aware_nms_merging_nearby_boxes():
    """Test that locality_aware_nms merges nearby boxes"""
    # Two very close boxes with high IoU
    boxes = np.array(
        [
            [0, 0, 10, 0, 10, 10, 0, 10, 0.9],
            [1, 1, 11, 1, 11, 11, 1, 11, 0.85],
        ],
        dtype=np.float32,
    )
    
    result = locality_aware_nms(boxes, iou_threshold=0.3)
    
    # They should merge into one
    assert result.shape[0] == 1

def test_locality_aware_nms_with_nan_protection():
    """Test NaN protection in locality_aware_nms"""
    # Boxes with very small scores to test division by zero protection
    boxes = np.array(
        [
            [0, 0, 10, 0, 10, 10, 0, 10, 1e-10],
            [1, 1, 11, 1, 11, 11, 1, 11, 1e-10],
        ],
        dtype=np.float32,
    )
    
    result = locality_aware_nms(boxes, iou_threshold=0.3)
    
    # There should be no NaN values
    assert np.isfinite(result).all()

def test_locality_aware_nms_score_max_preservation():
    """Test that the maximum score is preserved when merging"""
    boxes = np.array(
        [
            [0, 0, 10, 0, 10, 10, 0, 10, 0.7],
            [1, 1, 11, 1, 11, 11, 1, 11, 0.95],  # Higher score
        ],
        dtype=np.float32,
    )
    
    result = locality_aware_nms(boxes, iou_threshold=0.3)
    
    # The merged box should have the maximum score
    if result.shape[0] > 0:
        assert result[0, 8] >= 0.95 or np.isclose(result[0, 8], 0.95, rtol=0.01)

def test_polygon_area_large_polygon():
    """Test area of a large polygon"""
    poly = np.array(
        [[0, 0], [1000, 0], [1000, 500], [0, 500]], dtype=np.float64
    )
    area = polygon_area(poly)
    expected = 1000 * 500  # 500000
    np.testing.assert_allclose(area, expected, rtol=1e-5)

def test_polygon_area_negative_coordinates():
    """Test area with negative coordinates"""
    poly = np.array([[-10, -10], [10, -10], [10, 10], [-10, 10]], dtype=np.float64)
    area = polygon_area(poly)
    expected = 20 * 20  # 400
    np.testing.assert_allclose(area, expected, rtol=1e-5)

def test_polygon_iou_partial_overlap():
    """Test IoU with partial overlap"""
    poly1 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
    poly2 = np.array([[5, 5], [15, 5], [15, 15], [5, 15]], dtype=np.float64)
    
    iou = polygon_iou(poly1, poly2)
    
    # Intersection 5x5 = 25, union = 100 + 100 - 25 = 175
    expected = 25 / 175
    assert np.isclose(iou, expected, rtol=1e-3)

def test_polygon_iou_one_inside_other():
    """Test IoU when one polygon is inside another"""
    poly1 = np.array([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=np.float64)
    poly2 = np.array([[5, 5], [15, 5], [15, 15], [5, 15]], dtype=np.float64)
    
    iou = polygon_iou(poly1, poly2)
    
    # poly2 is completely inside poly1
    # Intersection = 100, poly1 = 400, poly2 = 100
    # IoU = 100 / 400 = 0.25
    assert iou > 0
    assert iou < 1

def test_quad_intersection_area_complex_shape():
    """Test intersection of complex shapes"""
    poly1 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
    poly2 = np.array([[5, -5], [15, -5], [15, 5], [5, 5]], dtype=np.float64)
    area = quad_intersection_area(poly1, poly2)
    assert area > 0.0


def test_should_merge_exactly_at_threshold():
    """Test should_merge exactly at the threshold"""
    poly1 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
    poly2 = np.array([[5, 5], [15, 5], [15, 15], [5, 15]], dtype=np.float64)
    
    iou = polygon_iou(poly1, poly2)
    
    # Check boundary case
    assert should_merge(poly1, poly2, iou - 0.001)
    assert not should_merge(poly1, poly2, iou + 0.001)

def test_clip_polygon_partial_clip():
    """Test partial clipping of a polygon"""
    subject = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
    # Clipping along a diagonal line
    A = np.array([0, 5], dtype=np.float64)
    B = np.array([10, 5], dtype=np.float64)
    
    clipped, count = _clip_polygon(subject, A, B)
    
    # A new polygon should be formed
    assert count > 0
    assert count <= 20  # Maximum buffer size

def test_line_intersection_edge_case():
    """Test line_intersection with edge case"""
    p1 = np.array([0.0, 0.0], dtype=np.float64)
    p2 = np.array([10.0, 10.0], dtype=np.float64)
    A = np.array([0.0, 10.0], dtype=np.float64)
    B = np.array([10.0, 0.0], dtype=np.float64)

    inter = np.array(line_intersection(*p1, *p2, *A, *B), dtype=np.float64)

    # The intersection point should be within the bounds of the line segments
    assert 0.0 < inter[0] < 10.0
    assert 0.0 < inter[1] < 10.0

def test_locality_aware_nms_far_apart_boxes():
    """Test locality_aware_nms with far apart boxes"""
    boxes = np.array(
        [
            [0, 0, 10, 0, 10, 10, 0, 10, 0.9],
            [1000, 1000, 1010, 1000, 1010, 1010, 1000, 1010, 0.8],
        ],
        dtype=np.float32,
    )
    
    result = locality_aware_nms(boxes, iou_threshold=0.5)
    
    # Both should remain (do not overlap)
    assert result.shape[0] == 2

def test_locality_aware_nms_different_thresholds():
    """Test locality_aware_nms with different thresholds"""
    boxes = np.array(
        [
            [0, 0, 10, 0, 10, 10, 0, 10, 0.9],
            [2, 2, 12, 2, 12, 12, 2, 12, 0.8],
            [4, 4, 14, 4, 14, 14, 4, 14, 0.7],
        ],
        dtype=np.float32,
    )
    
    # Low threshold - more merges
    result_low = locality_aware_nms(boxes, iou_threshold=0.01)
    
    # High threshold - fewer merges
    result_high = locality_aware_nms(boxes, iou_threshold=0.99)
    
    # With a high threshold, more boxes should remain
    assert result_high.shape[0] >= result_low.shape[0]

