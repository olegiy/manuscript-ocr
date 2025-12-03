from typing import Tuple, Optional, Union
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .io import read_image

try:
    import torch
except ImportError:
    torch = None


def draw_quads(
    image: Union[str, Path, np.ndarray, Image.Image],
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
    image : str, Path, np.ndarray, or PIL.Image
        Input image. Can be:
        - Path to image file (str or Path)
        - RGB numpy array with shape (H, W, 3)
        - PIL Image object
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
    >>> # From numpy array
    >>> img = np.zeros((480, 640, 3), dtype=np.uint8)
    >>> quads = np.array([[100, 100, 200, 100, 200, 150, 100, 150]])
    >>> result = draw_quads(img, quads, color=(255, 0, 0))
    
    >>> # From file path
    >>> result = draw_quads("document.jpg", quads, color=(255, 0, 0))
    """
    # Load image using universal reader
    if isinstance(image, (str, Path)):
        img = read_image(image)
    elif isinstance(image, Image.Image):
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


def visualize_page(
    image: Union[str, Path, np.ndarray, Image.Image],
    page: "Page",  # type: ignore  # noqa: F821
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    dark_alpha: float = 0.3,
    blur_ksize: int = 5,
    show_order: bool = True,
    line_color: Tuple[int, int, int] = (255, 165, 0),
    number_bg: Tuple[int, int, int] = (255, 255, 255),
    number_color: Tuple[int, int, int] = (0, 0, 0),
    max_size: Optional[int] = 4096,
) -> Image.Image:
    """
    Visualize a Page object with detected words/blocks.
    
    This function draws all words from the Page structure on the image,
    optionally showing reading order with numbered markers and connecting lines.
    When show_order=True, it also visualizes blocks with semi-transparent
    bounding boxes, each block having a distinct color.
    
    Parameters
    ----------
    image : str, Path, np.ndarray, or PIL.Image
        Input image. Can be:
        - Path to image file (str or Path) - supports Unicode paths
        - RGB numpy array with shape (H, W, 3)
        - PIL Image object
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
    show_order : bool, default=True
        If True, display reading order with numbers and connecting lines.
        Also colors different text lines with different colors and shows
        semi-transparent block boundaries with different colors per block.
    line_color : tuple of int, default=(255, 165, 0)
        RGB color for connecting lines between words.
    number_bg : tuple of int, default=(255, 255, 255)
        Background color for order number boxes.
    number_color : tuple of int, default=(0, 0, 0)
        Text color for order numbers.
    max_size : int or None, default=4096
        Maximum size for the longer dimension of the output image.
        Image will be resized proportionally if larger. Set to None to
        keep original size.
        
    Returns
    -------
    PIL.Image.Image
        Visualized image with detection boxes and optional reading order annotations.
        When show_order=True, also includes semi-transparent block boundaries.
        
    Examples
    --------
    Basic visualization without reading order:
    
    >>> from manuscript import EAST
    >>> from manuscript.utils import visualize_page
    >>> detector = EAST()
    >>> result = detector.predict("document.jpg")
    >>> # Can pass path directly
    >>> vis = visualize_page("document.jpg", result["page"])
    >>> vis.save("output.jpg")
    
    Visualization with reading order and block boundaries:
    
    >>> # Can also use numpy array or PIL Image
    >>> from manuscript.utils import read_image
    >>> img = read_image("document.jpg")
    >>> vis = visualize_page(
    ...     img,
    ...     result["page"],
    ...     show_order=True,
    ...     color=(255, 0, 0),
    ...     thickness=3
    ... )
    """
    # Load image using universal reader
    if isinstance(image, (str, Path)):
        img = read_image(image)
    elif isinstance(image, Image.Image):
        img = np.array(image.convert("RGB"))
    else:
        img = image.copy()
    
    # Resize image if needed
    if max_size is not None:
        h, w = img.shape[:2]
        max_dim = max(h, w)
        if max_dim > max_size:
            scale = max_size / max_dim
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
    else:
        scale = 1.0
    
    # Color palette for different lines (HSV-based for better distinction)
    def get_line_color(line_idx: int) -> Tuple[int, int, int]:
        """Generate distinct colors for different lines using HSV color space."""
        # Use golden ratio for better color distribution
        golden_ratio = 0.618033988749895
        hue = (line_idx * golden_ratio) % 1.0
        
        # Convert HSV to RGB
        h = int(hue * 179)  # OpenCV uses 0-179 for hue
        s = 220  # High saturation
        v = 255  # Full brightness
        
        hsv = np.uint8([[[h, s, v]]])
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0][0]
        return tuple(map(int, rgb))
    
    def get_block_color(block_idx: int) -> Tuple[int, int, int]:
        """Generate distinct colors for different blocks using HSV color space."""
        # Use different offset for blocks to distinguish from lines
        golden_ratio = 0.618033988749895
        hue = ((block_idx * golden_ratio) + 0.5) % 1.0
        
        # Convert HSV to RGB
        h = int(hue * 179)  # OpenCV uses 0-179 for hue
        s = 200  # Slightly lower saturation for blocks
        v = 255  # Full brightness
        
        hsv = np.uint8([[[h, s, v]]])
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0][0]
        return tuple(map(int, rgb))
    
    # Collect all quads and words in order, grouped by lines
    lines_data = []  # List of (line_quads, line_words, line_index)
    blocks_data = []  # List of (block_quads, block_index) for block visualization
    
    global_line_idx = 0  # Global line counter across all blocks
    
    for block_idx, block in enumerate(page.blocks):
        block_quads = []  # All quads in this block
        
        # Support both new (lines-based) and legacy (words-based) structure
        if block.lines:
            # New structure: iterate through lines
            for line in block.lines:
                line_quads = []
                line_words = []
                for w in line.words:
                    poly = np.array(w.polygon)
                    if scale != 1.0:
                        poly = poly * scale
                    quad = poly.reshape(-1)
                    line_quads.append(quad)
                    line_words.append(w)
                    block_quads.append(quad)  # Add to block quads
                if line_quads:
                    lines_data.append((line_quads, line_words, global_line_idx))
                    global_line_idx += 1
        elif block.words:
            # Legacy structure: direct words list - treat as single line
            line_quads = []
            line_words = []
            for w in block.words:
                poly = np.array(w.polygon)
                if scale != 1.0:
                    poly = poly * scale
                quad = poly.reshape(-1)
                line_quads.append(quad)
                line_words.append(w)
                block_quads.append(quad)  # Add to block quads
            if line_quads:
                lines_data.append((line_quads, line_words, global_line_idx))
                global_line_idx += 1
        
        # Store block data if it has any quads
        if block_quads:
            blocks_data.append((block_quads, block_idx))
    
    if len(lines_data) == 0:
        return Image.fromarray(img) if isinstance(image, np.ndarray) else image
    
    # Create mask for text areas to preserve them during darkening/blurring
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for line_quads, line_words, line_idx in lines_data:
        for quad in line_quads:
            coords = quad[:8].reshape(4, 2).astype(np.int32)
            cv2.fillPoly(mask, [coords], 255)
    
    # Apply darkening and/or blur only to background (outside boxes)
    overlay = img.copy()
    
    if dark_alpha > 0 or blur_ksize > 0:
        # Create background version (darkened/blurred)
        background = img.copy()
        
        # Apply darkening to background
        if dark_alpha > 0:
            background = (background * (1 - dark_alpha)).astype(np.uint8)
        
        # Apply blur to background
        if blur_ksize > 0:
            background = cv2.GaussianBlur(background, (blur_ksize, blur_ksize), 0)
        
        # Composite: use original image inside boxes, darkened/blurred outside
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        overlay = np.where(mask_3ch > 0, img, background)
    
    # Draw quads with line-specific colors if show_order is True
    # First draw block bounding boxes if show_order is True
    if show_order:
        for block_quads, block_idx in blocks_data:
            # Calculate bounding box for all quads in this block
            all_points = []
            for quad in block_quads:
                coords = quad[:8].reshape(4, 2)
                all_points.extend(coords)
            
            if all_points:
                all_points = np.array(all_points)
                min_x = int(np.min(all_points[:, 0]))
                min_y = int(np.min(all_points[:, 1]))
                max_x = int(np.max(all_points[:, 0]))
                max_y = int(np.max(all_points[:, 1]))
                
                # Get color for this block
                block_color = get_block_color(block_idx)
                
                # Create a semi-transparent overlay for this block
                block_overlay = overlay.copy()
                cv2.rectangle(block_overlay, (min_x, min_y), (max_x, max_y), 
                             block_color, -1)  # Filled rectangle
                
                # Blend with main image (high alpha = almost transparent)
                alpha = 0.15  # 15% opacity (85% transparent)
                overlay = cv2.addWeighted(overlay, 1 - alpha, block_overlay, alpha, 0)
    
    # Then draw individual word quads
    for line_quads, line_words, line_idx in lines_data:
        # Choose color: different per line if show_order, otherwise use default
        draw_color = get_line_color(line_idx) if show_order else color
        
        # Draw each quad in this line
        for quad in line_quads:
            coords = quad[:8].reshape(4, 2).astype(np.int32)
            cv2.polylines(overlay, [coords], isClosed=True, color=draw_color, thickness=thickness)
    
    out = Image.fromarray(overlay)
    
    # Add reading order visualization if requested
    if show_order:
        draw = ImageDraw.Draw(out)
        
        # Collect all words in reading order
        all_words = []
        for line_quads, line_words, line_idx in lines_data:
            all_words.extend(line_words)
        
        # Calculate centers of all words
        centers = []
        for w in all_words:
            xs = [p[0] * scale for p in w.polygon]
            ys = [p[1] * scale for p in w.polygon]
            centers.append((sum(xs) / len(xs), sum(ys) / len(ys)))
        
        # Draw connecting lines between consecutive words
        if len(centers) > 1:
            for p, c in zip(centers, centers[1:]):
                draw.line([p, c], fill=line_color, width=3)
        
        # Draw numbered boxes at centers with transparency
        # Create a transparent overlay for the number backgrounds
        overlay = Image.new('RGBA', out.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        for idx, c in enumerate(centers, start=1):
            cx, cy = c
            # Draw semi-transparent background rectangle
            bg_with_alpha = number_bg + (128,)  # Add alpha=128 (0.5 * 255)
            overlay_draw.rectangle(
                [cx - 12, cy - 12, cx + 12, cy + 12],
                fill=bg_with_alpha,
            )
        
        # Composite the overlay onto the main image
        out = out.convert('RGBA')
        out = Image.alpha_composite(out, overlay)
        out = out.convert('RGB')
        
        # Now draw text on top (opaque)
        draw = ImageDraw.Draw(out)
        for idx, c in enumerate(centers, start=1):
            cx, cy = c
            draw.text((cx - 6, cy - 8), str(idx), fill=number_color)
    
    return out
