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
        description="Word position inside the line after sorting. None before sorting."
    )


class Line(BaseModel):
    """
    A single text line containing one or more words.
    """
    words: List[Word]
    order: Optional[int] = Field(
        None,
        description="Line position inside a block or page after sorting. None before sorting."
    )


class Block(BaseModel):
    """
    A logical text block (e.g., paragraph, column).
    """
    lines: List[Line]
    order: Optional[int] = Field(
        None,
        description="Block reading-order position after sorting. None before sorting."
    )


class Page(BaseModel):
    """
    A document page containing blocks of text.
    """
    blocks: List[Block]
