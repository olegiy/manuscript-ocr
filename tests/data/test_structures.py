import pytest
from pydantic import ValidationError

from manuscript.data import Word, Line, Block, Page


# ============================================================================
# Тесты для Word
# ============================================================================


class TestWord:
    """Тесты для класса Word"""

    def test_word_creation_minimal(self):
        """Создание Word с минимальными параметрами"""
        word = Word(
            polygon=[(10.0, 20.0), (100.0, 20.0), (100.0, 40.0), (10.0, 40.0)],
            detection_confidence=0.95,
        )
        assert len(word.polygon) == 4
        assert word.detection_confidence == 0.95
        assert word.text is None
        assert word.recognition_confidence is None
        assert word.order is None

    def test_word_creation_full(self):
        """Создание Word со всеми параметрами"""
        word = Word(
            polygon=[(10.0, 20.0), (100.0, 20.0), (100.0, 40.0), (10.0, 40.0)],
            detection_confidence=0.95,
            text="Hello",
            recognition_confidence=0.98,
            order=0,
        )
        assert len(word.polygon) == 4
        assert word.detection_confidence == 0.95
        assert word.text == "Hello"
        assert word.recognition_confidence == 0.98
        assert word.order == 0

    def test_word_polygon_quadrilateral(self):
        """Проверка четырёхугольного полигона"""
        word = Word(
            polygon=[(0, 0), (10, 0), (10, 5), (0, 5)],
            detection_confidence=0.9,
        )
        assert len(word.polygon) == 4
        assert word.polygon[0] == (0, 0)
        assert word.polygon[1] == (10, 0)
        assert word.polygon[2] == (10, 5)
        assert word.polygon[3] == (0, 5)

    def test_word_polygon_more_than_4_points(self):
        """Полигон может иметь больше 4 точек"""
        word = Word(
            polygon=[(0, 0), (5, 0), (10, 0), (10, 5), (5, 5), (0, 5)],
            detection_confidence=0.9,
        )
        assert len(word.polygon) == 6

    def test_word_invalid_polygon_too_few_points(self):
        """Полигон должен иметь минимум 4 точки"""
        with pytest.raises(ValidationError) as exc_info:
            Word(
                polygon=[(0, 0), (10, 0), (10, 5)],  # Только 3 точки
                detection_confidence=0.9,
            )
        error_msg = str(exc_info.value).lower()
        # Pydantic v2 использует "too_short" вместо "min_items"
        assert "too_short" in error_msg or "min_items" in error_msg or "at least 4" in error_msg

    def test_word_detection_confidence_valid_range(self):
        """detection_confidence должен быть в диапазоне [0.0, 1.0]"""
        # Валидные значения
        Word(polygon=[(0, 0), (1, 0), (1, 1), (0, 1)], detection_confidence=0.0)
        Word(polygon=[(0, 0), (1, 0), (1, 1), (0, 1)], detection_confidence=0.5)
        Word(polygon=[(0, 0), (1, 0), (1, 1), (0, 1)], detection_confidence=1.0)

    def test_word_detection_confidence_invalid_negative(self):
        """detection_confidence < 0.0 должен вызывать ошибку"""
        with pytest.raises(ValidationError):
            Word(
                polygon=[(0, 0), (1, 0), (1, 1), (0, 1)], detection_confidence=-0.1
            )

    def test_word_detection_confidence_invalid_greater_than_one(self):
        """detection_confidence > 1.0 должен вызывать ошибку"""
        with pytest.raises(ValidationError):
            Word(polygon=[(0, 0), (1, 0), (1, 1), (0, 1)], detection_confidence=1.1)

    def test_word_recognition_confidence_valid_range(self):
        """recognition_confidence должен быть в диапазоне [0.0, 1.0]"""
        word = Word(
            polygon=[(0, 0), (1, 0), (1, 1), (0, 1)],
            detection_confidence=0.9,
            recognition_confidence=0.85,
        )
        assert word.recognition_confidence == 0.85

    def test_word_recognition_confidence_invalid(self):
        """Невалидный recognition_confidence"""
        with pytest.raises(ValidationError):
            Word(
                polygon=[(0, 0), (1, 0), (1, 1), (0, 1)],
                detection_confidence=0.9,
                recognition_confidence=1.5,
            )

    def test_word_text_optional(self):
        """text опциональный"""
        word = Word(
            polygon=[(0, 0), (1, 0), (1, 1), (0, 1)], detection_confidence=0.9
        )
        assert word.text is None

        word_with_text = Word(
            polygon=[(0, 0), (1, 0), (1, 1), (0, 1)],
            detection_confidence=0.9,
            text="Test",
        )
        assert word_with_text.text == "Test"

    def test_word_order_optional(self):
        """order опциональный"""
        word = Word(
            polygon=[(0, 0), (1, 0), (1, 1), (0, 1)], detection_confidence=0.9
        )
        assert word.order is None

        word_with_order = Word(
            polygon=[(0, 0), (1, 0), (1, 1), (0, 1)],
            detection_confidence=0.9,
            order=5,
        )
        assert word_with_order.order == 5


# ============================================================================
# Тесты для Line
# ============================================================================


class TestLine:
    """Тесты для класса Line"""

    def test_line_creation_single_word(self):
        """Создание Line с одним словом"""
        word = Word(
            polygon=[(0, 0), (10, 0), (10, 5), (0, 5)], detection_confidence=0.9
        )
        line = Line(words=[word])

        assert len(line.words) == 1
        assert line.words[0] == word
        assert line.order is None

    def test_line_creation_multiple_words(self):
        """Создание Line с несколькими словами"""
        word1 = Word(
            polygon=[(0, 0), (10, 0), (10, 5), (0, 5)], detection_confidence=0.9
        )
        word2 = Word(
            polygon=[(15, 0), (25, 0), (25, 5), (15, 5)], detection_confidence=0.95
        )
        word3 = Word(
            polygon=[(30, 0), (40, 0), (40, 5), (30, 5)], detection_confidence=0.92
        )

        line = Line(words=[word1, word2, word3])

        assert len(line.words) == 3
        assert line.words[0] == word1
        assert line.words[1] == word2
        assert line.words[2] == word3

    def test_line_with_order(self):
        """Создание Line с порядковым номером"""
        word = Word(
            polygon=[(0, 0), (10, 0), (10, 5), (0, 5)], detection_confidence=0.9
        )
        line = Line(words=[word], order=3)

        assert line.order == 3

    def test_line_empty_words_list(self):
        """Line с пустым списком слов"""
        line = Line(words=[])
        assert len(line.words) == 0


# ============================================================================
# Тесты для Block
# ============================================================================


class TestBlock:
    """Тесты для класса Block"""

    def test_block_creation_with_lines(self):
        """Создание Block с линиями"""
        word1 = Word(
            polygon=[(0, 0), (10, 0), (10, 5), (0, 5)], detection_confidence=0.9
        )
        word2 = Word(
            polygon=[(0, 10), (10, 10), (10, 15), (0, 15)], detection_confidence=0.95
        )

        line1 = Line(words=[word1])
        line2 = Line(words=[word2])

        block = Block(lines=[line1, line2])

        assert len(block.lines) == 2
        assert block.lines[0] == line1
        assert block.lines[1] == line2
        assert block.order is None

    def test_block_creation_with_words_legacy(self):
        """Создание Block со словами (legacy API) - автоматически создаёт линию"""
        word1 = Word(
            polygon=[(0, 0), (10, 0), (10, 5), (0, 5)], detection_confidence=0.9
        )
        word2 = Word(
            polygon=[(15, 0), (25, 0), (25, 5), (15, 5)], detection_confidence=0.95
        )

        block = Block(words=[word1, word2])

        # Должна создаться одна линия с обоими словами
        assert len(block.lines) == 1
        assert len(block.lines[0].words) == 2
        assert block.lines[0].words[0] == word1
        assert block.lines[0].words[1] == word2
        assert len(block.words) == 2  # Legacy поле тоже сохраняется

    def test_block_creation_empty(self):
        """Создание пустого Block"""
        block = Block()

        assert len(block.lines) == 0
        assert len(block.words) == 0

    def test_block_with_order(self):
        """Создание Block с порядковым номером"""
        word = Word(
            polygon=[(0, 0), (10, 0), (10, 5), (0, 5)], detection_confidence=0.9
        )
        line = Line(words=[word])
        block = Block(lines=[line], order=2)

        assert block.order == 2

    def test_block_lines_take_priority_over_words(self):
        """Если заданы и lines и words, lines имеет приоритет"""
        word1 = Word(
            polygon=[(0, 0), (10, 0), (10, 5), (0, 5)], detection_confidence=0.9
        )
        word2 = Word(
            polygon=[(15, 0), (25, 0), (25, 5), (15, 5)], detection_confidence=0.95
        )

        line = Line(words=[word1])

        # Передаём и lines, и words
        block = Block(lines=[line], words=[word2])

        # lines должен иметь приоритет
        assert len(block.lines) == 1
        assert len(block.lines[0].words) == 1
        assert block.lines[0].words[0] == word1


# ============================================================================
# Тесты для Page
# ============================================================================


class TestPage:
    """Тесты для класса Page"""

    def test_page_creation_single_block(self):
        """Создание Page с одним блоком"""
        word = Word(
            polygon=[(0, 0), (10, 0), (10, 5), (0, 5)], detection_confidence=0.9
        )
        line = Line(words=[word])
        block = Block(lines=[line])

        page = Page(blocks=[block])

        assert len(page.blocks) == 1
        assert page.blocks[0] == block

    def test_page_creation_multiple_blocks(self):
        """Создание Page с несколькими блоками"""
        word1 = Word(
            polygon=[(0, 0), (10, 0), (10, 5), (0, 5)], detection_confidence=0.9
        )
        word2 = Word(
            polygon=[(0, 10), (10, 10), (10, 15), (0, 15)], detection_confidence=0.95
        )
        word3 = Word(
            polygon=[(0, 20), (10, 20), (10, 25), (0, 25)], detection_confidence=0.92
        )

        line1 = Line(words=[word1])
        line2 = Line(words=[word2])
        line3 = Line(words=[word3])

        block1 = Block(lines=[line1])
        block2 = Block(lines=[line2, line3])

        page = Page(blocks=[block1, block2])

        assert len(page.blocks) == 2
        assert page.blocks[0] == block1
        assert page.blocks[1] == block2

    def test_page_empty_blocks(self):
        """Page с пустым списком блоков"""
        page = Page(blocks=[])
        assert len(page.blocks) == 0

    def test_page_complex_structure(self):
        """Сложная структура: Page -> Blocks -> Lines -> Words"""
        # Создаём слова
        words_line1 = [
            Word(
                polygon=[(i * 10, 0), (i * 10 + 8, 0), (i * 10 + 8, 5), (i * 10, 5)],
                detection_confidence=0.9 + i * 0.01,
                text=f"word{i}",
            )
            for i in range(5)
        ]

        words_line2 = [
            Word(
                polygon=[
                    (i * 10, 10),
                    (i * 10 + 8, 10),
                    (i * 10 + 8, 15),
                    (i * 10, 15),
                ],
                detection_confidence=0.85 + i * 0.01,
                text=f"word{i + 5}",
            )
            for i in range(3)
        ]

        # Создаём линии
        line1 = Line(words=words_line1, order=0)
        line2 = Line(words=words_line2, order=1)

        # Создаём блоки
        block1 = Block(lines=[line1, line2], order=0)
        block2 = Block(
            words=[
                Word(
                    polygon=[(0, 20), (10, 20), (10, 25), (0, 25)],
                    detection_confidence=0.88,
                )
            ],
            order=1,
        )

        # Создаём страницу
        page = Page(blocks=[block1, block2])

        # Проверки
        assert len(page.blocks) == 2
        assert len(page.blocks[0].lines) == 2
        assert len(page.blocks[0].lines[0].words) == 5
        assert len(page.blocks[0].lines[1].words) == 3
        assert len(page.blocks[1].lines) == 1  # Legacy words -> line
        assert page.blocks[0].lines[0].words[0].text == "word0"
        assert page.blocks[0].lines[1].words[2].text == "word7"


# ============================================================================
# Тесты для интеграции и крайних случаев
# ============================================================================


class TestIntegration:
    """Интеграционные тесты"""

    def test_full_ocr_pipeline_structure(self):
        """Симуляция полной структуры результата OCR pipeline"""
        # Детектор создаёт Page с Block и Words (без text)
        detected_words = [
            Word(
                polygon=[(i * 20, 0), (i * 20 + 18, 0), (i * 20 + 18, 10), (i * 20, 10)],
                detection_confidence=0.9 + i * 0.01,
            )
            for i in range(3)
        ]

        block = Block(words=detected_words)
        page_after_detection = Page(blocks=[block])

        assert len(page_after_detection.blocks) == 1
        assert len(page_after_detection.blocks[0].lines) == 1
        assert len(page_after_detection.blocks[0].lines[0].words) == 3
        assert all(w.text is None for w in page_after_detection.blocks[0].lines[0].words)

        # Распознаватель добавляет text и recognition_confidence
        for idx, word in enumerate(page_after_detection.blocks[0].lines[0].words):
            word.text = f"Word{idx}"
            word.recognition_confidence = 0.95 + idx * 0.01

        # Проверяем результат
        assert page_after_detection.blocks[0].lines[0].words[0].text == "Word0"
        assert page_after_detection.blocks[0].lines[0].words[1].text == "Word1"
        assert page_after_detection.blocks[0].lines[0].words[2].text == "Word2"
        assert (
            page_after_detection.blocks[0].lines[0].words[0].recognition_confidence
            == 0.95
        )

    def test_backward_compatibility_words_only(self):
        """Проверка обратной совместимости: Block создаётся только с words"""
        words = [
            Word(
                polygon=[(0, 0), (10, 0), (10, 5), (0, 5)],
                detection_confidence=0.9,
                text="Hello",
            ),
            Word(
                polygon=[(15, 0), (30, 0), (30, 5), (15, 5)],
                detection_confidence=0.95,
                text="World",
            ),
        ]

        # Старый API: передаём только words
        block = Block(words=words)

        # Проверяем, что автоматически создалась одна линия
        assert len(block.lines) == 1
        assert len(block.lines[0].words) == 2
        assert block.lines[0].words[0].text == "Hello"
        assert block.lines[0].words[1].text == "World"

    def test_pydantic_serialization(self):
        """Проверка сериализации/десериализации через Pydantic"""
        word = Word(
            polygon=[(0, 0), (10, 0), (10, 5), (0, 5)],
            detection_confidence=0.9,
            text="Test",
            recognition_confidence=0.95,
        )

        # Сериализация в dict
        word_dict = word.model_dump()
        assert word_dict["polygon"] == [(0, 0), (10, 0), (10, 5), (0, 5)]
        assert word_dict["text"] == "Test"

        # Десериализация обратно
        word_restored = Word(**word_dict)
        assert word_restored.text == "Test"
        assert word_restored.detection_confidence == 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
