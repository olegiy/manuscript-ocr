"""
Data structures for manuscript OCR.

This package contains the core data structures used to represent OCR results
throughout the manuscript-ocr library.
"""

from .structures import Word, Line, Block, Page

__all__ = ["Word", "Line", "Block", "Page"]
