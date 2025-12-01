"""Visualization utilities for manuscript-ocr."""

from typing import Tuple, Optional, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw

try:
    import torch
except ImportError:
    torch = None


def draw_quads(
    image: Union[np.ndarray, Image.Image],
    quads: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    dark_alpha: float = 0.3,
    blur_ksize: int = 5,
) -> Image.Image:
    """
    Draw quadrilateral boxes on an image with semi-transparent overlay.
    
    Parameters
    ----------
    image : np.ndarray or PIL.Image
        Input image (RGB).
    quads : np.ndarray
        Array of quad boxes with shape (N, 8) or (N, 9).
        Each row contains [x1, y1, x2, y2, x3, y3, x4, y4] or with score.
    color : tuple of int, default=(0, 255, 0)
        RGB color for drawing boxes.
    thickness : int, default=2
        Line thickness in pixels.
    dark_alpha : float, default=0.3
        Alpha value for darkening the image (0=no darkening, 1=fully dark).
    blur_ksize : int, default=5
        Kernel size for Gaussian blur (must be odd, 0=no blur).
        
    Returns
    -------
    PIL.Image.Image
        Image with drawn quadrilaterals.
        
    Examples
    --------
    >>> import numpy as np
    >>> from PIL import Image
    >>> img = np.zeros((480, 640, 3), dtype=np.uint8)
    >>> quads = np.array([[100, 100, 200, 100, 200, 150, 100, 150]])
    >>> result = draw_quads(img, quads, color=(255, 0, 0))
    """
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        img = np.array(image.convert("RGB"))
    else:
        img = image.copy()
    
    # Apply darkening if requested
    if dark_alpha > 0:
        overlay = (img * (1 - dark_alpha)).astype(np.uint8)
    else:
        overlay = img
    
    # Apply blur if requested
    if blur_ksize > 0:
        overlay = cv2.GaussianBlur(overlay, (blur_ksize, blur_ksize), 0)
    
    # Draw each quad
    for quad in quads:
        coords = quad[:8].reshape(4, 2).astype(np.int32)
        cv2.polylines(overlay, [coords], isClosed=True, color=color, thickness=thickness)
    
    return Image.fromarray(overlay)


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Draw quad boxes on an image.
    
    This is a convenience wrapper for draw_quads that maintains backward compatibility.
    
    Parameters
    ----------
    image : np.ndarray
        Input image (RGB).
    boxes : np.ndarray or torch.Tensor
        Array of quad boxes with shape (N, 8) or (N, 9).
        Each row: [x1, y1, x2, y2, x3, y3, x4, y4] with optional score.
    color : tuple of int, default=(0, 255, 0)
        RGB color for drawing boxes.
    thickness : int, default=2
        Line thickness in pixels.
    alpha : float, default=0.5
        Transparency of the overlay (used as dark_alpha in draw_quads).
        
    Returns
    -------
    np.ndarray
        Image with drawn boxes.
        
    Raises
    ------
    ValueError
        If box format is not recognized (not 8 or 9 values per box).
        
    Examples
    --------
    >>> quads = np.array([[10, 10, 50, 10, 50, 30, 10, 30]])
    >>> img = draw_boxes(image, quads)
    """
    if boxes is None or len(boxes) == 0:
        return image
    
    # Handle PyTorch tensors
    if torch is not None and isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu().numpy()
    
    # Validate format
    first = boxes[0]
    if len(first) not in (8, 9):
        raise ValueError(
            f"Unsupported box format with length {len(first)}. "
            "Expected quad boxes with 8 or 9 values per box."
        )
    
    # Use draw_quads for rendering
    quad_img = draw_quads(
        image, boxes, color=color, thickness=thickness, dark_alpha=alpha
    )
    return np.array(quad_img)


def visualize_page(
    image: Union[np.ndarray, Image.Image],
    page: "Page",  # type: ignore  # noqa: F821
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    dark_alpha: float = 0.3,
    blur_ksize: int = 5,
    show_order: bool = False,
    line_color: Tuple[int, int, int] = (255, 165, 0),
    number_bg: Tuple[int, int, int] = (255, 255, 255),
    number_color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """
    Visualize a Page object with detected words/blocks.
    
    This function draws all words from the Page structure on the image,
    optionally showing reading order with numbered markers and connecting lines.
    
    Parameters
    ----------
    image : np.ndarray or PIL.Image
        Input image (RGB).
    page : Page
        Page object from manuscript.data containing detected blocks/words.
    color : tuple of int, default=(0, 255, 0)
        RGB color for word boundaries.
    thickness : int, default=2
        Line thickness for word boundaries.
    dark_alpha : float, default=0.3
        Alpha for darkening background (0=no darkening, 1=fully dark).
    blur_ksize : int, default=5
        Gaussian blur kernel size (0=no blur, must be odd).
    show_order : bool, default=False
        If True, display reading order with numbers and connecting lines.
    line_color : tuple of int, default=(255, 165, 0)
        RGB color for connecting lines between words.
    number_bg : tuple of int, default=(255, 255, 255)
        Background color for order number boxes.
    number_color : tuple of int, default=(0, 0, 0)
        Text color for order numbers.
        
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
    
    Notes
    -----
    This function was moved from manuscript.detectors._east.utils to be model-agnostic,
    as it works with the generic Page structure from manuscript.data.
    """
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
    
    # Draw polygons
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
