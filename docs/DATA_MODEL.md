```mermaid
mindmap
  root((OCR Data Model))
    Page
      blocks[List of Block]
    Block
      words[List of Word]
      order("Block reading-order position\nNone before sorting")
    Word
      polygon("Polygon vertices (x, y)\n≥ 4 points, clockwise order")
      detection_confidence("Detection score 0.0–1.0")
      text("Recognized text (optional)")
      recognition_confidence("Recognition score 0.0–1.0")
      order("Word reading-order position\nNone before sorting")
```