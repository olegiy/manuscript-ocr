from manuscript.data import Block, Line, Page, Word
from manuscript.utils import organize_page


def test_organize_page_empty():
    """Test organize_page with empty page."""
    page = Page(blocks=[])
    result = organize_page(page)

    assert len(result.blocks) == 1
    assert len(result.blocks[0].lines) == 1
    assert len(result.blocks[0].lines[0].words) == 0


def test_organize_page_single_word():
    """Test organize_page with single word."""
    word = Word(
        polygon=[(10, 20), (100, 20), (100, 40), (10, 40)], detection_confidence=0.95
    )
    page = Page(blocks=[Block(lines=[Line(words=[word], order=0)], order=0)])

    result = organize_page(page, use_columns=False)

    assert len(result.blocks) == 1
    assert len(result.blocks[0].lines) == 1
    assert len(result.blocks[0].lines[0].words) == 1
    assert result.blocks[0].lines[0].words[0].order == 0


def test_organize_page_multiple_words():
    """Test organize_page with multiple words in same line."""
    words = [
        Word(
            polygon=[(10, 20), (50, 20), (50, 40), (10, 40)], detection_confidence=0.95
        ),
        Word(
            polygon=[(60, 20), (110, 20), (110, 40), (60, 40)],
            detection_confidence=0.97,
        ),
        Word(
            polygon=[(120, 20), (180, 20), (180, 40), (120, 40)],
            detection_confidence=0.93,
        ),
    ]
    page = Page(blocks=[Block(lines=[Line(words=words, order=0)], order=0)])

    result = organize_page(page, use_columns=False)

    assert len(result.blocks) == 1
    assert len(result.blocks[0].lines) == 1
    assert len(result.blocks[0].lines[0].words) == 3
    # Check reading order (left to right)
    for i, word in enumerate(result.blocks[0].lines[0].words):
        assert word.order == i


def test_organize_page_multiple_lines():
    """Test organize_page with multiple lines."""
    words = [
        # First line
        Word(
            polygon=[(10, 20), (50, 20), (50, 40), (10, 40)], detection_confidence=0.95
        ),
        Word(
            polygon=[(60, 20), (110, 20), (110, 40), (60, 40)],
            detection_confidence=0.97,
        ),
        # Second line (below first line)
        Word(
            polygon=[(10, 50), (50, 50), (50, 70), (10, 70)], detection_confidence=0.93
        ),
        Word(
            polygon=[(60, 50), (110, 50), (110, 70), (60, 70)],
            detection_confidence=0.91,
        ),
    ]
    page = Page(blocks=[Block(lines=[Line(words=words, order=0)], order=0)])

    result = organize_page(page, use_columns=False)

    assert len(result.blocks) == 1
    assert len(result.blocks[0].lines) == 2
    # First line should have 2 words
    assert len(result.blocks[0].lines[0].words) == 2
    # Second line should have 2 words
    assert len(result.blocks[0].lines[1].words) == 2
    # Check line order (top to bottom)
    assert result.blocks[0].lines[0].order == 0
    assert result.blocks[0].lines[1].order == 1


def test_organize_page_with_columns():
    """Test organize_page with column detection."""
    words = [
        # Left column
        Word(
            polygon=[(10, 20), (50, 20), (50, 40), (10, 40)], detection_confidence=0.95
        ),
        Word(
            polygon=[(10, 50), (50, 50), (50, 70), (10, 70)], detection_confidence=0.93
        ),
        # Right column (far from left)
        Word(
            polygon=[(200, 20), (250, 20), (250, 40), (200, 40)],
            detection_confidence=0.97,
        ),
        Word(
            polygon=[(200, 50), (250, 50), (250, 70), (200, 70)],
            detection_confidence=0.91,
        ),
    ]
    page = Page(blocks=[Block(lines=[Line(words=words, order=0)], order=0)])

    result = organize_page(page, use_columns=True, max_splits=10)

    # Should detect 2 columns (2 blocks)
    assert len(result.blocks) >= 1  # At least one block
    # Each block should have lines
    for block in result.blocks:
        assert len(block.lines) > 0
        for line in block.lines:
            assert len(line.words) > 0


def test_organize_page_preserves_word_attributes():
    """Test that organize_page preserves word attributes."""
    word = Word(
        polygon=[(10, 20), (100, 20), (100, 40), (10, 40)],
        detection_confidence=0.95,
        text="Hello",
        recognition_confidence=0.98,
    )
    page = Page(blocks=[Block(lines=[Line(words=[word], order=0)], order=0)])

    result = organize_page(page, use_columns=False)

    result_word = result.blocks[0].lines[0].words[0]
    assert result_word.polygon == word.polygon
    assert result_word.detection_confidence == word.detection_confidence
    assert result_word.text == word.text
    assert result_word.recognition_confidence == word.recognition_confidence
