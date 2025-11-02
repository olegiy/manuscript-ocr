"""
EAST Detector Utilities

Содержит утилиты для работы с EAST детектором:
- Визуализация результатов
- Конвертация форматов боксов
- Оценка точности детекции (F1, Precision, Recall)

Пример оценки точности:
    >>> from manuscript.detectors._east.utils import evaluate_detection
    >>> pred_boxes = [(10, 10, 50, 50), (60, 60, 100, 100)]
    >>> gt_boxes = [(12, 12, 48, 48), (61, 61, 99, 99)]
    >>> metrics = evaluate_detection(pred_boxes, gt_boxes, iou_threshold=0.5)
    >>> print(f"F1: {metrics['f1']:.4f}")

Пример оценки датасета:
    >>> from manuscript.detectors._east.utils import evaluate_dataset
    >>> predictions = {"img1.jpg": [(10, 10, 50, 50)]}
    >>> ground_truths = {"img1.jpg": [(12, 12, 48, 48)]}
    >>> metrics = evaluate_dataset(predictions, ground_truths)
    >>> print(f"F1@0.5: {metrics['f1@0.5']:.4f}")
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from shapely.geometry import Polygon
from tqdm import tqdm


"""
def convert_rboxes_to_quad_boxes(rboxes, scores=None):
    quad_boxes = []
    if scores is None:
        scores = np.ones(len(rboxes), dtype=np.float32)
    for i, r in enumerate(rboxes):
        cx, cy, w, h, angle = r
        pts = cv2.boxPoints(((cx, cy), (w, h), angle))
        quad = np.concatenate([pts.flatten(), [scores[i]]]).astype(np.float32)
        quad_boxes.append(quad)
    return np.array(quad_boxes, dtype=np.float32)
"""


def quad_to_rbox(quad):
    pts = quad[:8].reshape(4, 2).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    (cx, cy), (w, h), angle = rect
    return np.array([cx, cy, w, h, angle], dtype=np.float32)


def tensor_to_image(tensor):
    img = tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = (img * 0.5) + 0.5
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def draw_quads(
    image: np.ndarray,
    quads: np.ndarray,
    color: tuple = (0, 0, 0),
    thickness: int = 1,
    dark_alpha: float = 0.5,
    blur_ksize: int = 11,
) -> Image.Image:
    """
    Draw quadrilateral boxes over the input image and return a PIL visualization.

    Args:
        image: Source image in RGB (preferred) or BGR order with shape HxWx3.
        quads: Array-like collection of quadrangles shaped `N x 9` or `N x 8`
            where the first eight values are `[x1, y1, ..., x4, y4]`.
        color: Polyline color passed to OpenCV.
        thickness: Polyline thickness in pixels.
        dark_alpha: Blending factor for darkening the background under polygons.
        blur_ksize: Odd kernel size for Gaussian blur applied to the soft mask.

    Returns:
        PIL.Image.Image: Image containing the visualization overlay.
    """
    img = image.copy()
    if quads is None or len(quads) == 0:
        return Image.fromarray(img)

    if isinstance(quads, torch.Tensor):
        quads = quads.detach().cpu().numpy()

    h, w = img.shape[:2]
    dark_bg = (img.astype(np.float32) * (1 - dark_alpha)).astype(np.uint8)

    mask = np.zeros((h, w), dtype=np.float32)
    for q in quads:
        pts = q[:8].reshape(4, 2).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1.0)

    k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    mask = np.clip(mask, 0.0, 1.0)
    mask_3 = mask[:, :, None]

    out = img.astype(np.float32) * mask_3 + dark_bg.astype(np.float32) * (1 - mask_3)
    out = np.clip(out, 0, 255).astype(np.uint8)

    for q in quads:
        pts = q[:8].reshape(4, 2).astype(np.int32)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)

    return Image.fromarray(out)


def visualize_page(
    image,
    page,
    *,
    show_order=False,
    color=(0, 0, 255),
    thickness=2,
    dark_alpha=0.3,
    blur_ksize=11,
    line_color=(0, 255, 0),
    number_color=(255, 255, 255),
    number_bg=(0, 0, 0),
) -> Image.Image:
    """
    Unified visualization function for Page objects with optional reading order display.

    This function draws text detection boxes on the image using the same style as EAST detector,
    with optional numbered boxes and connecting lines to show reading order.

    Parameters
    ----------
    image : np.ndarray or PIL.Image.Image
        Input image (RGB format preferred).
    page : Page
        Page object containing detected text blocks and words.
    show_order : bool, optional
        Whether to display reading order (numbered boxes with connecting lines), by default False.
    color : tuple, optional
        Color for polygon borders in RGB format, by default (0, 0, 255) - blue.
    thickness : int, optional
        Thickness of polygon borders in pixels, by default 2.
    dark_alpha : float, optional
        Blending factor for darkening background (0.0 to 1.0), by default 0.3.
    blur_ksize : int, optional
        Kernel size for Gaussian blur (must be odd), by default 11.
    line_color : tuple, optional
        Color for connecting lines in RGB format, by default (0, 255, 0) - green.
    number_color : tuple, optional
        Color for box numbers text in RGB format, by default (255, 255, 255) - white.
    number_bg : tuple, optional
        Background color for box numbers in RGB format, by default (0, 0, 0) - black.

    Returns
    -------
    PIL.Image.Image
        Visualized image with detection boxes and optional reading order annotations.

    Examples
    --------
    Basic visualization without reading order:

    >>> from manuscript import EAST, visualize_page
    >>> detector = EAST()
    >>> result = detector.predict("document.jpg")
    >>> vis = visualize_page(result["vis_image"], result["page"])
    >>> vis.save("output.jpg")

    Visualization with reading order display:

    >>> vis = visualize_page(
    ...     result["vis_image"],
    ...     result["page"],
    ...     show_order=True,
    ...     color=(255, 0, 0),
    ...     thickness=3
    ... )
    """
    from PIL import ImageDraw

    # Convert to numpy array if PIL Image
    if isinstance(image, Image.Image):
        img = np.array(image.convert("RGB"))
    else:
        img = image.copy()

    # Collect all quads and words in order
    quads = []
    words_in_order = []

    for block in page.blocks:
        for w in block.words:
            poly = np.array(w.polygon).reshape(-1)
            quads.append(poly)
            words_in_order.append(w)

    if len(quads) == 0:
        return Image.fromarray(img) if isinstance(image, np.ndarray) else image

    quads = np.stack(quads, axis=0)

    # Draw polygons using EAST style
    out = draw_quads(
        image=img,
        quads=quads,
        color=color,
        thickness=thickness,
        dark_alpha=dark_alpha,
        blur_ksize=blur_ksize,
    )

    # Add reading order visualization if requested
    if show_order:
        draw = ImageDraw.Draw(out)

        # Calculate centers of all words
        centers = []
        for w in words_in_order:
            xs = [p[0] for p in w.polygon]
            ys = [p[1] for p in w.polygon]
            centers.append((sum(xs) / len(xs), sum(ys) / len(ys)))

        # Draw connecting lines between consecutive words
        if len(centers) > 1:
            for p, c in zip(centers, centers[1:]):
                draw.line([p, c], fill=line_color, width=3)

        # Draw numbered boxes at centers
        for idx, c in enumerate(centers, start=1):
            cx, cy = c
            draw.rectangle(
                [cx - 12, cy - 12, cx + 12, cy + 12],
                fill=number_bg,
            )
            draw.text((cx - 6, cy - 8), str(idx), fill=number_color)

    return out


def draw_rboxes(image, rboxes, color=(0, 255, 0), thickness=2, alpha=0.5):
    img = image.copy()
    if rboxes is None or len(rboxes) == 0:
        return img
    if isinstance(rboxes, torch.Tensor):
        rboxes = rboxes.detach().cpu().numpy()
    overlay = img.copy()
    for r in rboxes:
        cx, cy, w, h, angle = r
        pts = cv2.boxPoints(((cx, cy), (w, h), angle))
        pts = np.int32(pts)
        cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=thickness)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2, alpha=0.5):
    if boxes is None or len(boxes) == 0:
        return image
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu().numpy()
    # detect format by length
    first = boxes[0]
    if len(first) == 5:
        return draw_rboxes(image, boxes, color=color, thickness=thickness, alpha=alpha)
    elif len(first) in (8, 9):
        # quad with or without score
        quad_img = draw_quads(
            image, boxes, color=color, thickness=thickness, dark_alpha=alpha
        )
        return np.array(quad_img)
    else:
        raise ValueError(f"Unsupported box format with length {len(first)}")


def create_collage(
    img_tensor,
    gt_score_map,
    gt_geo_map,
    gt_rboxes,
    pred_score_map=None,
    pred_geo_map=None,
    pred_rboxes=None,
    cell_size=640,
):
    n_rows, n_cols = 2, 10
    collage = np.full((cell_size * n_rows, cell_size * n_cols, 3), 255, dtype=np.uint8)
    orig = tensor_to_image(img_tensor)

    # GT
    gt_img = draw_boxes(orig, gt_rboxes, color=(0, 255, 0))
    gt_score = (
        gt_score_map.detach().cpu().numpy().squeeze()
        if isinstance(gt_score_map, torch.Tensor)
        else gt_score_map
    )
    gt_score_vis = cv2.applyColorMap(
        (gt_score * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    gt_geo = (
        gt_geo_map.detach().cpu().numpy()
        if isinstance(gt_geo_map, torch.Tensor)
        else gt_geo_map
    )
    gt_cells = [gt_img, gt_score_vis]
    for i in range(gt_geo.shape[2]):
        ch = gt_geo[:, :, i]
        norm = cv2.normalize(ch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        gt_cells.append(cv2.applyColorMap(norm, cv2.COLORMAP_JET))

    # Pred
    if pred_score_map is not None and pred_geo_map is not None:
        pred_img = draw_boxes(orig, pred_rboxes, color=(0, 0, 255))
        pred_score = (
            pred_score_map.detach().cpu().numpy().squeeze()
            if isinstance(pred_score_map, torch.Tensor)
            else pred_score_map
        )
        pred_score_vis = cv2.applyColorMap(
            (pred_score * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        pred_geo = (
            pred_geo_map.detach().cpu().numpy()
            if isinstance(pred_geo_map, torch.Tensor)
            else pred_geo_map
        )
        pred_cells = [pred_img, pred_score_vis]
        for i in range(pred_geo.shape[2]):
            ch = pred_geo[:, :, i]
            norm = cv2.normalize(ch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            pred_cells.append(cv2.applyColorMap(norm, cv2.COLORMAP_JET))
    else:
        pred_cells = [np.zeros((cell_size, cell_size, 3), dtype=np.uint8)] * n_cols

    # assemble
    for r in range(n_rows):
        cells = gt_cells if r == 0 else pred_cells
        for c in range(n_cols):
            cell = cv2.resize(cells[c], (cell_size, cell_size))
            y0, y1 = r * cell_size, (r + 1) * cell_size
            x0, x1 = c * cell_size, (c + 1) * cell_size
            collage[y0:y1, x0:x1] = cell

    return collage


def decode_quads_from_maps(
    score_map: np.ndarray,
    geo_map: np.ndarray,
    score_thresh: float,
    scale: float,
    quantization: int = 1,
    profile=False,
) -> np.ndarray:
    if score_map.ndim == 3 and score_map.shape[0] == 1:
        score_map = score_map.squeeze(0)

    t0 = time.time()
    ys, xs = np.where(score_map > score_thresh)
    if profile:
        print(f"    Find pixels > thresh: {time.time() - t0:.3f}s ({len(ys)} pixels)")

    if len(ys) == 0:
        return np.zeros((0, 9), dtype=np.float32)

    if quantization > 1:
        t0_quant = time.time()
        ys_quant = (ys // quantization) * quantization + quantization // 2
        xs_quant = (xs // quantization) * quantization + quantization // 2

        coords = np.column_stack([ys_quant, xs_quant])
        unique_coords = np.unique(coords, axis=0)

        ys = unique_coords[:, 0]
        xs = unique_coords[:, 1]

        if profile:
            print(
                f"    Quantization (step={quantization}): {time.time() - t0_quant:.3f}s"
            )
            print(
                f"    Points after quantization: {len(ys)} (removed {len(coords) - len(ys)})"
            )

    t0 = time.time()
    quads = []
    for y, x in zip(ys, xs):
        offs = geo_map[y, x]
        verts = []
        for i in range(4):
            dx_map, dy_map = offs[2 * i], offs[2 * i + 1]
            vx = x * scale + dx_map * scale
            vy = y * scale + dy_map * scale
            verts.extend([vx, vy])
        quads.append(verts + [float(score_map[y, x])])

    if profile:
        print(f"    Decode coordinates: {time.time() - t0:.3f}s ({len(quads)} quads)")

    return np.array(quads, dtype=np.float32)


def expand_boxes(
    quads: np.ndarray,
    expand_w: float = 0.0,
    expand_h: float = 0.0,
    expand_power: float = 1.0,
) -> np.ndarray:
    """
    Расширяет боксы с нелинейным масштабированием.

    Parameters
    ----------
    quads : np.ndarray
        Массив четырехугольников shape (N, 9) - 8 координат + score
    expand_w : float
        Коэффициент расширения по ширине
    expand_h : float
        Коэффициент расширения по высоте
    expand_power : float
        Степень для нелинейного масштабирования:
        - 1.0 = линейное (маленькие и большие боксы увеличиваются одинаково)
        - <1.0 = маленькие боксы увеличиваются сильнее (например, 0.5)
        - >1.0 = большие боксы увеличиваются сильнее

    Returns
    -------
    np.ndarray
        Расширенные боксы

    Notes
    -----
    Нелинейное увеличение полезно, так как маленькие боксы (символы) требуют
    больше "запаса" для распознавания, чем большие боксы (слова).
    """
    if len(quads) == 0 or (expand_w == 0 and expand_h == 0):
        return quads

    coords = quads[:, :8].reshape(-1, 4, 2)
    scores = quads[:, 8:9]

    x, y = coords[:, :, 0], coords[:, :, 1]
    area = np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)
    sign = np.sign(area).reshape(-1, 1, 1)
    sign[sign == 0] = 1

    p_prev = np.roll(coords, 1, axis=1)
    p_curr = coords
    p_next = np.roll(coords, -1, axis=1)

    edge1 = p_curr - p_prev
    edge2 = p_next - p_curr
    len1 = np.linalg.norm(edge1, axis=2, keepdims=True)
    len2 = np.linalg.norm(edge2, axis=2, keepdims=True)

    n1 = sign * np.stack([edge1[..., 1], -edge1[..., 0]], axis=2) / (len1 + 1e-6)
    n2 = sign * np.stack([edge2[..., 1], -edge2[..., 0]], axis=2) / (len2 + 1e-6)
    n_avg = n1 + n2
    norm = np.linalg.norm(n_avg, axis=2, keepdims=True)

    # Normalize, but if norm is too small (degenerate case), use n1 as fallback
    n_normalized = np.divide(n_avg, norm, out=np.zeros_like(n_avg), where=norm > 1e-6)
    degenerate_mask = (norm <= 1e-6).squeeze(-1)  # shape: (N, 4)
    n_normalized[degenerate_mask] = n1[
        degenerate_mask
    ]  # fallback to n1 when degenerate

    offset = np.minimum(len1, len2)

    # Нелинейное масштабирование: маленькие боксы увеличиваются сильнее при expand_power < 1.0
    # Нормализуем offset для стабильности (используем среднее значение как референс)
    if expand_power != 1.0:
        # Применяем степенную функцию к нормализованному offset
        # Это делает малые значения offset относительно больше при power < 1
        offset_scaled = np.power(offset, expand_power)
    else:
        offset_scaled = offset

    scale_xy = np.array([1 + expand_w, 1 + expand_h], dtype=np.float32).reshape(1, 1, 2)
    delta = (scale_xy - 1.0) * offset_scaled

    new_coords = p_curr + delta * n_normalized

    expanded = np.hstack([new_coords.reshape(-1, 8), scores])
    return expanded.astype(np.float32)


def poly_iou(segA, segB):
    A = Polygon(np.array(segA).reshape(-1, 2))
    B = Polygon(np.array(segB).reshape(-1, 2))
    if not A.is_valid or not B.is_valid:
        return 0.0
    inter = A.intersection(B).area
    union = A.union(B).area
    return inter / union if union > 0 else 0.0


def compute_f1(preds, thresh, gt_segs, processed_ids):
    gt_polys = {
        iid: [Polygon(np.array(seg).reshape(-1, 2)) for seg in gt_segs.get(iid, [])]
        for iid in processed_ids
    }
    pred_polys = [
        {
            "image_id": p["image_id"],
            "polygon": Polygon(np.array(p["segmentation"]).reshape(-1, 2)),
        }
        for p in preds
    ]

    used = {iid: [False] * len(gt_polys.get(iid, [])) for iid in processed_ids}
    tp = fp = 0
    for p, pred_poly in zip(preds, pred_polys):
        image_id = p["image_id"]
        pred_polygon = pred_poly["polygon"]
        if not pred_polygon.is_valid:
            fp += 1
            continue
        best_iou, bj = 0, -1
        for j, gt_polygon in enumerate(gt_polys.get(image_id, [])):
            if used[image_id][j] or not gt_polygon.is_valid:
                continue
            inter = pred_polygon.intersection(gt_polygon).area
            union = pred_polygon.union(gt_polygon).area
            iou = inter / union if union > 0 else 0
            if iou > best_iou:
                best_iou, bj = iou, j
        if best_iou >= thresh:
            tp += 1
            used[image_id][bj] = True
        else:
            fp += 1
    total_gt = sum(len(v) for v in gt_polys.values())
    fn = total_gt - tp
    prec = tp / (tp + fp) if tp + fp > 0 else 0
    rec = tp / (tp + fn) if tp + fn > 0 else 0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0


def read_image(img_or_path):
    if isinstance(img_or_path, (str, Path)):
        img = cv2.imread(str(img_or_path))
        if img is None:
            try:
                with Image.open(str(img_or_path)) as pil_img:
                    img = np.array(pil_img.convert("RGB"))
            except Exception as e:
                raise FileNotFoundError(
                    f"Cannot read image with cv2 or PIL: {img_or_path}. Error: {e}"
                )
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    elif isinstance(img_or_path, np.ndarray):
        img = img_or_path

    else:
        raise TypeError(f"Unsupported type for image input: {type(img_or_path)}")

    return img


def resolve_intersections(boxes):
    """
    Resolve intersecting boxes by shrinking them iteratively.

    Parameters
    ----------
    boxes : list of tuple
        List of boxes in format (x_min, y_min, x_max, y_max).

    Returns
    -------
    list of tuple
        List of resolved boxes.
    """

    def intersect(b1, b2):
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


def sort_boxes_reading_order(boxes, y_tol_ratio=0.6, x_gap_ratio=np.inf):
    """
    Sort boxes in natural reading order (left-to-right, top-to-bottom).

    Groups boxes into lines based on vertical proximity, then sorts each line
    horizontally. This approximates the natural reading order for documents.

    Parameters
    ----------
    boxes : list of tuple
        List of boxes in format (x_min, y_min, x_max, y_max).
    y_tol_ratio : float, optional
        Vertical tolerance as a ratio of average box height for grouping boxes
        into the same line, by default 0.6.
    x_gap_ratio : float, optional
        Maximum horizontal gap as a ratio of average box height for boxes to be
        considered part of the same line, by default np.inf (no limit).

    Returns
    -------
    list of tuple
        Boxes sorted in reading order.

    Examples
    --------
    >>> boxes = [(10, 10, 50, 30), (60, 10, 100, 30), (10, 50, 50, 70)]
    >>> sorted_boxes = sort_boxes_reading_order(boxes)
    """
    if not boxes:
        return []

    avg_h = np.mean([b[3] - b[1] for b in boxes])
    lines = []

    for b in sorted(boxes, key=lambda b: (b[1] + b[3]) / 2):
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
    for ln in lines:
        ln.sort(key=lambda b: b[0])

    return [b for ln in lines for b in ln]


def sort_boxes_reading_order_with_resolutions(
    boxes, y_tol_ratio=0.6, x_gap_ratio=np.inf
):
    """
    Sort boxes in reading order after resolving intersections.

    This function first resolves overlapping boxes by shrinking them, then applies
    reading order sorting. Useful when boxes may overlap slightly.

    Parameters
    ----------
    boxes : list of tuple
        List of boxes in format (x_min, y_min, x_max, y_max).
    y_tol_ratio : float, optional
        Vertical tolerance for line grouping, by default 0.6.
    x_gap_ratio : float, optional
        Maximum horizontal gap for line continuity, by default np.inf.

    Returns
    -------
    list of tuple
        Boxes sorted in reading order with intersections resolved.

    Examples
    --------
    >>> boxes = [(10, 10, 55, 30), (50, 10, 100, 30)]  # Overlapping
    >>> sorted_boxes = sort_boxes_reading_order_with_resolutions(boxes)
    """
    compressed = resolve_intersections(boxes)
    mapping = {c: o for c, o in zip(compressed, boxes)}

    sorted_compressed = sort_boxes_reading_order(
        compressed, y_tol_ratio=y_tol_ratio, x_gap_ratio=x_gap_ratio
    )
    return [mapping[b] for b in sorted_compressed]


"""
def load_gt(gt_path):
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_coco = json.load(f)
    gt_segs = defaultdict(list)
    for ann in gt_coco["annotations"]:
        seg = ann.get("segmentation", [])
        if seg:
            gt_segs[ann["image_id"]].append(seg[0])
    return gt_segs


def load_preds(pred_path):
    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    preds_list = data.get("annotations", data)
    preds = []
    for p in preds_list:
        seg = p.get("segmentation", [])
        if not seg:
            continue
        preds.append(
            {
                "image_id": p["image_id"],
                "segmentation": seg[0],
                "score": p.get("score", 1.0),
            }
        )
    return preds
"""


def box_iou(box1, box2):
    """
    Вычисляет IoU между двумя прямоугольными боксами.

    Parameters
    ----------
    box1 : tuple or list
        Бокс в формате (x_min, y_min, x_max, y_max)
    box2 : tuple or list
        Бокс в формате (x_min, y_min, x_max, y_max)

    Returns
    -------
    float
        IoU значение в диапазоне [0, 1]

    Examples
    --------
    >>> iou = box_iou((10, 10, 50, 50), (30, 30, 70, 70))
    >>> print(f"IoU: {iou:.3f}")
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # Union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def match_boxes(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Сопоставляет предсказанные боксы с ground truth боксами.

    Parameters
    ----------
    pred_boxes : list of tuple
        Список предсказанных боксов в формате (x_min, y_min, x_max, y_max)
    gt_boxes : list of tuple
        Список ground truth боксов в формате (x_min, y_min, x_max, y_max)
    iou_threshold : float
        Порог IoU для считывания match

    Returns
    -------
    tuple
        (true_positives, false_positives, false_negatives)

    Examples
    --------
    >>> pred = [(10, 10, 50, 50), (60, 60, 100, 100)]
    >>> gt = [(12, 12, 48, 48)]
    >>> tp, fp, fn = match_boxes(pred, gt, iou_threshold=0.5)
    """
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 0, 0, 0

    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0

    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)

    # Матрица IoU
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i, pred in enumerate(pred_boxes):
        for j, gt in enumerate(gt_boxes):
            iou_matrix[i, j] = box_iou(pred, gt)

    # Greedy matching: каждый pred сопоставляется с лучшим gt
    matched_gt = set()
    matched_pred = set()

    # Сортируем все пары по убыванию IoU
    matches = []
    for i in range(len(pred_boxes)):
        for j in range(len(gt_boxes)):
            if iou_matrix[i, j] >= iou_threshold:
                matches.append((iou_matrix[i, j], i, j))

    matches.sort(reverse=True)

    # Выбираем лучшие совпадения
    for iou_val, i, j in matches:
        if i not in matched_pred and j not in matched_gt:
            matched_pred.add(i)
            matched_gt.add(j)

    tp = len(matched_pred)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)

    return tp, fp, fn


def compute_f1_score(true_positives, false_positives, false_negatives):
    """
    Вычисляет F1 score по TP, FP, FN.

    Parameters
    ----------
    true_positives : int
        Количество истинных положительных
    false_positives : int
        Количество ложных положительных
    false_negatives : int
        Количество ложных отрицательных

    Returns
    -------
    tuple
        (f1_score, precision, recall)

    Examples
    --------
    >>> f1, prec, rec = compute_f1_score(80, 10, 10)
    >>> print(f"F1: {f1:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}")
    """
    if true_positives == 0:
        return 0.0, 0.0, 0.0

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )

    if precision + recall == 0:
        return 0.0, precision, recall

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1, precision, recall


def evaluate_detection(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Оценивает качество детекции для одного изображения.

    Parameters
    ----------
    pred_boxes : list of tuple
        Список предсказанных боксов в формате (x_min, y_min, x_max, y_max)
    gt_boxes : list of tuple
        Список ground truth боксов в формате (x_min, y_min, x_max, y_max)
    iou_threshold : float
        Порог IoU для считывания match

    Returns
    -------
    dict
        {"tp": int, "fp": int, "fn": int, "f1": float, "precision": float, "recall": float}

    Examples
    --------
    >>> pred = [(10, 10, 50, 50), (60, 60, 100, 100)]
    >>> gt = [(12, 12, 48, 48), (61, 61, 99, 99)]
    >>> metrics = evaluate_detection(pred, gt, iou_threshold=0.5)
    """
    tp, fp, fn = match_boxes(pred_boxes, gt_boxes, iou_threshold)
    f1, precision, recall = compute_f1_score(tp, fp, fn)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def evaluate_detection_multi_iou(pred_boxes, gt_boxes, iou_thresholds=None):
    """
    Оценивает качество детекции для одного изображения с несколькими порогами IoU.

    Parameters
    ----------
    pred_boxes : list of tuple
        Список предсказанных боксов в формате (x_min, y_min, x_max, y_max)
    gt_boxes : list of tuple
        Список ground truth боксов в формате (x_min, y_min, x_max, y_max)
    iou_thresholds : list of float, optional
        Список порогов IoU. По умолчанию [0.5, 0.55, 0.6, ..., 0.95]

    Returns
    -------
    dict
        Метрики для каждого порога + средние значения

    Examples
    --------
    >>> pred = [(10, 10, 50, 50)]
    >>> gt = [(12, 12, 48, 48)]
    >>> metrics = evaluate_detection_multi_iou(pred, gt)
    >>> print(f"F1@0.5: {metrics['f1@0.5']:.3f}")
    >>> print(f"F1@0.5:0.95: {metrics['f1@0.5:0.95']:.3f}")
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05).tolist()

    results = {}
    f1_scores = []

    for threshold in iou_thresholds:
        metrics = evaluate_detection(pred_boxes, gt_boxes, iou_threshold=threshold)
        results[f"f1@{threshold:.2f}"] = metrics["f1"]
        results[f"precision@{threshold:.2f}"] = metrics["precision"]
        results[f"recall@{threshold:.2f}"] = metrics["recall"]
        f1_scores.append(metrics["f1"])

    # Специальные метрики
    results["f1@0.5"] = results.get("f1@0.50", 0.0)
    results["f1@0.5:0.95"] = float(np.mean(f1_scores))

    return results


def _evaluate_image_worker(args):
    """
    Рабочая функция для параллельной оценки одного изображения.

    Parameters
    ----------
    args : tuple
        (image_id, pred_boxes, gt_boxes, iou_thresholds)

    Returns
    -------
    dict
        Словарь {threshold: (tp, fp, fn)} для всех порогов
    """
    image_id, pred_boxes, gt_boxes, iou_thresholds = args
    results = {}

    for threshold in iou_thresholds:
        tp, fp, fn = match_boxes(pred_boxes, gt_boxes, iou_threshold=threshold)
        results[threshold] = (tp, fp, fn)

    return results


def evaluate_dataset(
    predictions, ground_truths, iou_thresholds=None, verbose=True, n_jobs=None
):
    """
    Оценивает качество детекции на всем датасете.

    Parameters
    ----------
    predictions : dict
        Словарь {image_id: list of boxes}, где boxes в формате (x_min, y_min, x_max, y_max)
    ground_truths : dict
        Словарь {image_id: list of boxes}, где boxes в формате (x_min, y_min, x_max, y_max)
    iou_thresholds : list of float, optional
        Список порогов IoU
    verbose : bool
        Печатать прогресс
    n_jobs : int, optional
        Количество процессов для параллельной обработки.
        None - использовать все доступные CPU
        1 - последовательная обработка
        >1 - указанное количество процессов

    Returns
    -------
    dict
        Агрегированные метрики по всему датасету

    Examples
    --------
    >>> preds = {"img1": [(10, 10, 50, 50)], "img2": [(20, 20, 60, 60)]}
    >>> gts = {"img1": [(12, 12, 48, 48)], "img2": [(22, 22, 58, 58)]}
    >>> metrics = evaluate_dataset(preds, gts)
    >>> print(f"Dataset F1@0.5: {metrics['f1@0.5']:.3f}")
    >>> # С параллельной обработкой (работает на Windows)
    >>> metrics = evaluate_dataset(preds, gts, n_jobs=4)
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05).tolist()

    # Собираем метрики по каждому порогу
    total_tp = {th: 0 for th in iou_thresholds}
    total_fp = {th: 0 for th in iou_thresholds}
    total_fn = {th: 0 for th in iou_thresholds}

    all_image_ids = list(set(list(predictions.keys()) + list(ground_truths.keys())))

    # Определяем, использовать ли параллельную обработку
    use_parallel = n_jobs is None or n_jobs > 1

    if use_parallel:
        import multiprocessing as mp
        from multiprocessing import Pool

        # Для Windows используем 'spawn' контекст
        if n_jobs is None:
            n_jobs = mp.cpu_count()

        # Подготавливаем аргументы для воркеров
        worker_args = [
            (
                image_id,
                predictions.get(image_id, []),
                ground_truths.get(image_id, []),
                iou_thresholds,
            )
            for image_id in all_image_ids
        ]

        # Используем spawn для совместимости с Windows
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_jobs) as pool:
            if verbose:
                # tqdm с imap для отображения прогресса
                image_results = list(
                    tqdm(
                        pool.imap(_evaluate_image_worker, worker_args),
                        total=len(worker_args),
                        desc="Evaluating images",
                    )
                )
            else:
                image_results = pool.map(_evaluate_image_worker, worker_args)

        # Агрегируем результаты
        for result in image_results:
            for threshold, (tp, fp, fn) in result.items():
                total_tp[threshold] += tp
                total_fp[threshold] += fp
                total_fn[threshold] += fn
    else:
        # Последовательная обработка
        iterator = (
            tqdm(all_image_ids, desc="Evaluating images") if verbose else all_image_ids
        )

        for image_id in iterator:
            pred_boxes = predictions.get(image_id, [])
            gt_boxes = ground_truths.get(image_id, [])

            for threshold in iou_thresholds:
                tp, fp, fn = match_boxes(pred_boxes, gt_boxes, iou_threshold=threshold)
                total_tp[threshold] += tp
                total_fp[threshold] += fp
                total_fn[threshold] += fn

    # Вычисляем агрегированные метрики
    results = {}
    f1_scores = []

    for threshold in iou_thresholds:
        tp = total_tp[threshold]
        fp = total_fp[threshold]
        fn = total_fn[threshold]

        f1, precision, recall = compute_f1_score(tp, fp, fn)

        results[f"f1@{threshold:.2f}"] = f1
        results[f"precision@{threshold:.2f}"] = precision
        results[f"recall@{threshold:.2f}"] = recall
        results[f"tp@{threshold:.2f}"] = tp
        results[f"fp@{threshold:.2f}"] = fp
        results[f"fn@{threshold:.2f}"] = fn

        f1_scores.append(f1)

    # Специальные метрики
    results["f1@0.5"] = results.get("f1@0.50", 0.0)
    results["precision@0.5"] = results.get("precision@0.50", 0.0)
    results["recall@0.5"] = results.get("recall@0.50", 0.0)
    results["f1@0.5:0.95"] = float(np.mean(f1_scores))
    results["num_images"] = len(all_image_ids)
    results["num_predictions"] = sum(len(boxes) for boxes in predictions.values())
    results["num_ground_truths"] = sum(len(boxes) for boxes in ground_truths.values())

    return results
