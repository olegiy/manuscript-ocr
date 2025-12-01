from pydantic import BaseModel, Field
from typing import List, Tuple, Optional


class Word(BaseModel):
    polygon: List[Tuple[float, float]] = Field(
        ...,
        min_items=4,
        description="Polygon vertices (x, y), ordered clockwise. For quadrilateral text regions: TL → TR → BR → BL."
    )
    detection_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Text detection confidence score from detector"
    )
    text: Optional[str] = Field(
        None, description="Recognized text content (populated by OCR pipeline)"
    )
    recognition_confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Text recognition confidence score from recognizer"
    )
    order: Optional[int] = Field(
        None,
        description="Reading-order position assigned after sorting. None before sorting; list order remains authoritative."
    )

class Block(BaseModel):
    """
    A text block, which may consist of several words (Word).
    """
    words: List[Word]
    order: Optional[int] = Field(
        None,
        description="Block reading-order position after sorting. None before sorting."
    )

class Page(BaseModel):
    """
    A document page containing one or multiple text blocks.
    """

    blocks: List[Block]
