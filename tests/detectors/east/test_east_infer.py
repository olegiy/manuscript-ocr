"""
Комплексные тесты для EAST детектора (infer.py)
"""

import pytest
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from unittest.mock import Mock, patch, MagicMock
import tempfile

from manuscript.detectors import EAST
from manuscript.data import Page, Block, Word

@pytest.mark.skip(reason="Временно отключено")
class TestEASTInitialization:
    """Тесты инициализации EAST"""

    def test_initialization_default_parameters(self):
        """Тест инициализации с параметрами по умолчанию"""
        detector = EAST()

        # Проверяем базовые атрибуты
        assert detector is not None
        assert hasattr(detector, "predict")
        assert hasattr(detector, "onnx_session")
        assert detector.target_size == 1280
        assert detector.score_thresh == 0.6
        assert detector.expand_ratio_w == 1.4
        assert detector.expand_ratio_h == 1.5

    def test_initialization_custom_parameters(self):
        """Тест инициализации с кастомными параметрами"""
        detector = EAST(
            target_size=640,
            score_thresh=0.8,
            expand_ratio_w=1.0,
            expand_ratio_h=1.0,
            iou_threshold=0.1,
        )

        assert detector.target_size == 640
        assert detector.score_thresh == 0.8
        assert detector.expand_ratio_w == 1.0
        assert detector.expand_ratio_h == 1.0
        assert detector.iou_threshold == 0.1

    def test_initialization_device_auto_selection(self):
        """Тест автоматического выбора устройства"""
        detector = EAST()
        
        # device должен быть cuda или cpu
        assert detector.device in ["cuda", "cpu"]

    def test_initialization_device_explicit_cpu(self):
        """Тест явного выбора CPU"""
        detector = EAST(device="cpu")
        assert detector.device == "cpu"

    def test_initialization_nonexistent_weights(self):
        """Тест ошибки при несуществующих весах"""
        with pytest.raises(FileNotFoundError, match="ONNX model not found"):
            EAST(weights_path="nonexistent_model.onnx")

    def test_initialization_downloads_default_model(self):
        """Тест что модель скачивается если не указан путь"""
        # Модель должна скачаться или уже быть скачанной
        detector = EAST()
        
        # Проверяем что сессия создана
        assert detector.onnx_session is not None
        
        # Проверяем что дефолтная модель существует
        default_path = Path.home() / ".manuscript" / "east" / "east_quad_23_05.onnx"
        assert default_path.exists()

@pytest.mark.skip(reason="Временно отключено")
class TestEASTScalingMethods:
    """Тесты для методов масштабирования и трансформации"""

    def test_scale_boxes_to_original_empty(self):
        """Тест масштабирования пустого массива боксов"""
        detector = EAST(target_size=1280)
        boxes = np.array([])
        
        scaled = detector._scale_boxes_to_original(boxes, (480, 640))
        
        assert len(scaled) == 0

    def test_scale_boxes_to_original_basic(self):
        """Тест базового масштабирования"""
        detector = EAST(target_size=1280)
        
        # Один бокс: 8 координат + 1 score
        boxes = np.array([
            [100, 100, 200, 100, 200, 200, 100, 200, 0.9]
        ])
        
        # Исходное изображение 480x640
        scaled = detector._scale_boxes_to_original(boxes, (480, 640))
        
        # Проверяем что координаты изменились
        assert not np.allclose(scaled[:, :8], boxes[:, :8])
        # Score не должен измениться
        assert scaled[0, 8] == 0.9

    def test_scale_boxes_to_original_correct_scaling(self):
        """Тест правильности масштабирования"""
        detector = EAST(target_size=1000)
        
        # Бокс на изображении 1000x1000 в точке (100, 100)
        boxes = np.array([
            [100, 100, 200, 100, 200, 200, 100, 200, 0.9]
        ])
        
        # Масштабируем на изображение 500x1000
        # scale_x = 1000/1000 = 1.0, scale_y = 500/1000 = 0.5
        scaled = detector._scale_boxes_to_original(boxes, (500, 1000))
        
        # X координаты не изменятся (scale_x=1.0)
        assert scaled[0, 0] == 100
        # Y координаты уменьшатся вдвое (scale_y=0.5)
        assert scaled[0, 1] == 50

    def test_convert_to_axis_aligned_empty(self):
        """Тест конвертации пустого массива"""
        detector = EAST()
        quads = np.array([])
        
        aligned = detector._convert_to_axis_aligned(quads)
        
        assert len(aligned) == 0

    def test_convert_to_axis_aligned_basic(self):
        """Тест конвертации квадрата в axis-aligned прямоугольник"""
        detector = EAST()
        
        # Повернутый квадрат
        quads = np.array([
            [50, 100, 150, 50, 200, 150, 100, 200, 0.9]
        ])
        
        aligned = detector._convert_to_axis_aligned(quads)
        
        # Должен стать axis-aligned (прямоугольник по осям)
        coords = aligned[0, :8].reshape(4, 2)
        
        # Проверяем что это прямоугольник с параллельными сторонами
        # Две точки должны иметь одинаковый x (левая сторона)
        # Две точки должны иметь одинаковый x (правая сторона)
        x_coords = sorted(coords[:, 0])
        assert x_coords[0] == x_coords[1]  # Левая сторона
        assert x_coords[2] == x_coords[3]  # Правая сторона

    def test_convert_to_axis_aligned_multiple_quads(self):
        """Тест конвертации нескольких квадов"""
        detector = EAST()
        
        quads = np.array([
            [0, 0, 100, 0, 100, 100, 0, 100, 0.9],
            [200, 200, 300, 200, 300, 300, 200, 300, 0.8],
        ])
        
        aligned = detector._convert_to_axis_aligned(quads)
        
        assert aligned.shape == quads.shape
        # Scores не изменяются
        assert aligned[0, 8] == 0.9
        assert aligned[1, 8] == 0.8

@pytest.mark.skip(reason="Временно отключено")
class TestEASTGeometricUtils:
    """Тесты для геометрических утилит"""

    def test_polygon_area_batch_empty(self):
        """Тест вычисления площади для пустого массива"""
        areas = EAST._polygon_area_batch(np.array([]))
        assert len(areas) == 0

    def test_polygon_area_batch_square(self):
        """Тест вычисления площади квадрата"""
        # Квадрат 100x100
        polys = np.array([
            [[0, 0], [100, 0], [100, 100], [0, 100]]
        ])
        
        areas = EAST._polygon_area_batch(polys)
        
        assert len(areas) == 1
        assert abs(areas[0] - 10000) < 1  # Площадь ~10000

    def test_polygon_area_batch_multiple(self):
        """Тест вычисления площадей нескольких полигонов"""
        polys = np.array([
            [[0, 0], [10, 0], [10, 10], [0, 10]],  # 100
            [[0, 0], [20, 0], [20, 20], [0, 20]],  # 400
        ])
        
        areas = EAST._polygon_area_batch(polys)
        
        assert len(areas) == 2
        assert abs(areas[0] - 100) < 1
        assert abs(areas[1] - 400) < 1

    def test_is_quad_inside_true(self):
        """Тест что маленький квад внутри большого"""
        detector = EAST()
        
        inner = np.array([[10, 10], [20, 10], [20, 20], [10, 20]])
        outer = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        
        result = detector._is_quad_inside(inner, outer)
        
        assert result is True

    def test_is_quad_inside_false(self):
        """Тест что квад снаружи"""
        detector = EAST()
        
        inner = np.array([[150, 150], [160, 150], [160, 160], [150, 160]])
        outer = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        
        result = detector._is_quad_inside(inner, outer)
        
        assert result is False

    def test_remove_fully_contained_boxes_empty(self):
        """Тест удаления вложенных боксов из пустого массива"""
        detector = EAST()
        quads = np.array([])
        
        result = detector._remove_fully_contained_boxes(quads)
        
        assert len(result) == 0

    def test_remove_fully_contained_boxes_single(self):
        """Тест с одним боксом"""
        detector = EAST()
        quads = np.array([
            [0, 0, 100, 0, 100, 100, 0, 100, 0.9]
        ])
        
        result = detector._remove_fully_contained_boxes(quads)
        
        assert len(result) == 1

    def test_remove_fully_contained_boxes_nested(self):
        """Тест удаления вложенного бокса"""
        detector = EAST()
        
        # Большой бокс и маленький внутри него
        quads = np.array([
            [0, 0, 100, 0, 100, 100, 0, 100, 0.9],      # Большой
            [10, 10, 20, 10, 20, 20, 10, 20, 0.8],      # Маленький внутри
        ])
        
        result = detector._remove_fully_contained_boxes(quads)
        
        # Должен остаться только большой
        assert len(result) == 1
    @pytest.mark.skip(reason="Временно отключено")
    def test_remove_area_anomalies_disabled(self):
        """Тест что аномалии не удаляются если флаг выключен"""
        detector = EAST(remove_area_anomalies=False)
        
        quads = np.array([
            [0, 0, 10, 0, 10, 10, 0, 10, 0.9],
            [0, 0, 1000, 0, 1000, 1000, 0, 1000, 0.9],  # Огромный
        ])
        
        result = detector._remove_area_anomalies(quads)
        
        # Все остаются
        assert len(result) == 2
    @pytest.mark.skip(reason="Временно отключено")
    def test_remove_area_anomalies_too_few_boxes(self):
        """Тест что аномалии не удаляются если боксов мало"""
        detector = EAST(remove_area_anomalies=True, anomaly_min_box_count=10)
        
        quads = np.array([
            [0, 0, 10, 0, 10, 10, 0, 10, 0.9],
            [0, 0, 1000, 0, 1000, 1000, 0, 1000, 0.9],
        ])
        
        result = detector._remove_area_anomalies(quads)
        
        # Все остаются (меньше 10 боксов)
        assert len(result) == 2

@pytest.mark.skip(reason="Временно отключено")
class TestEASTPredict:
    """Тесты для метода predict"""

    @pytest.fixture
    def test_image(self):
        """Создает тестовое изображение"""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    @pytest.fixture
    def test_image_path(self, tmp_path, test_image):
        """Создает временный файл с изображением"""
        img_path = tmp_path / "test_image.jpg"
        cv2.imwrite(str(img_path), cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR))
        return str(img_path)
    @pytest.mark.skip(reason="Временно отключено")
    def test_predict_with_path(self, test_image_path):
        """Тест predict с путем к файлу"""
        detector = EAST()
        
        result = detector.predict(test_image_path, vis=False)
        
        assert isinstance(result, dict)
        assert "page" in result
        assert "vis_image" in result
        assert isinstance(result["page"], Page)
    @pytest.mark.skip(reason="Временно отключено")
    def test_predict_with_numpy_array(self, test_image):
        """Тест predict с numpy array"""
        detector = EAST()
        
        result = detector.predict(test_image, vis=False)
        
        assert isinstance(result, dict)
        assert "page" in result
        assert isinstance(result["page"], Page)
    @pytest.mark.skip(reason="Временно отключено")
    def test_predict_returns_page_structure(self, test_image):
        """Тест что predict возвращает правильную структуру Page"""
        detector = EAST()
        
        result = detector.predict(test_image)
        page = result["page"]
        
        assert hasattr(page, "blocks")
        assert isinstance(page.blocks, list)
        
        if len(page.blocks) > 0:
            block = page.blocks[0]
            assert hasattr(block, "words")
            assert isinstance(block.words, list)
    @pytest.mark.skip(reason="Временно отключено")
    def test_predict_with_visualization(self, test_image):
        """Тест predict с визуализацией"""
        detector = EAST()
        
        result = detector.predict(test_image, vis=True)
        
        assert "vis_image" in result
        vis_img = result["vis_image"]
        
        if vis_img is not None:
            assert isinstance(vis_img, Image.Image)
    @pytest.mark.skip(reason="Временно отключено")
    def test_predict_with_return_maps(self, test_image):
        """Тест predict с возвратом карт"""
        detector = EAST()
        
        result = detector.predict(test_image, return_maps=True)
        
        assert "score_map" in result
        assert "geo_map" in result
        
        if result["score_map"] is not None:
            assert isinstance(result["score_map"], np.ndarray)
        if result["geo_map"] is not None:
            assert isinstance(result["geo_map"], np.ndarray)
    @pytest.mark.skip(reason="Временно отключено")
    def test_predict_without_return_maps(self, test_image):
        """Тест что без return_maps карты None"""
        detector = EAST()
        
        result = detector.predict(test_image, return_maps=False)
        
        assert result["score_map"] is None
        assert result["geo_map"] is None
    @pytest.mark.skip(reason="Временно отключено")
    def test_predict_with_profile(self, test_image, capsys):
        """Тест режима профилирования"""
        detector = EAST()
        
        result = detector.predict(test_image, profile=True)
        
        # Проверяем что были выведены сообщения о времени
        captured = capsys.readouterr()
        assert "Model inference" in captured.out or "inference" in captured.out.lower()
    @pytest.mark.skip(reason="Временно отключено")
    def test_predict_with_sort_reading_order(self, test_image):
        """Тест сортировки в порядке чтения"""
        detector = EAST()
        
        result = detector.predict(test_image, sort_reading_order=True)
        page = result["page"]
        
        # Просто проверяем что не упало
        assert isinstance(page, Page)
    @pytest.mark.skip(reason="Временно отключено")
    def test_predict_different_thresholds(self, test_image):
        """Тест с разными порогами"""
        detector_low = EAST(score_thresh=0.1)
        detector_high = EAST(score_thresh=0.9)
        
        result_low = detector_low.predict(test_image)
        result_high = detector_high.predict(test_image)
        
        # Оба должны отработать без ошибок
        assert isinstance(result_low["page"], Page)
        assert isinstance(result_high["page"], Page)
    @pytest.mark.skip(reason="Временно отключено")
    def test_predict_axis_aligned_vs_original(self, test_image):
        """Тест axis_aligned_output флага"""
        detector_aligned = EAST(axis_aligned_output=True)
        detector_original = EAST(axis_aligned_output=False)
        
        result_aligned = detector_aligned.predict(test_image)
        result_original = detector_original.predict(test_image)
        
        # Оба должны вернуть результаты
        assert isinstance(result_aligned["page"], Page)
        assert isinstance(result_original["page"], Page)


@pytest.mark.skip(reason="Временно отключено")
class TestEASTErrorHandling:
    """Тесты обработки ошибок"""
    @pytest.mark.skip(reason="Временно отключено")
    def test_predict_nonexistent_file(self):
        """Тест ошибки при несуществующем файле"""
        detector = EAST()
        
        with pytest.raises(FileNotFoundError):
            detector.predict("nonexistent_image.jpg")
    @pytest.mark.skip(reason="Временно отключено")
    def test_predict_invalid_type(self):
        """Тест ошибки при неправильном типе входа"""
        detector = EAST()
        
        with pytest.raises((TypeError, AttributeError)):
            detector.predict(12345)
    @pytest.mark.skip(reason="Временно отключено")
    def test_predict_corrupted_image_path(self, tmp_path):
        """Тест с поврежденным файлом изображения"""
        # Создаем текстовый файл вместо изображения
        fake_img = tmp_path / "fake.jpg"
        fake_img.write_text("not an image")
        
        detector = EAST()
        
        # Должна быть ошибка при чтении
        with pytest.raises((cv2.error, ValueError, OSError)):
            detector.predict(str(fake_img))

@pytest.mark.skip(reason="Временно отключено")
class TestEASTEdgeCases:
    """Тесты граничных случаев"""
    @pytest.mark.skip(reason="Временно отключено")
    def test_predict_very_small_image(self):
        """Тест с очень маленьким изображением"""
        detector = EAST()
        
        # Изображение 10x10
        small_img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        
        result = detector.predict(small_img)
        
        # Не должно быть ошибок
        assert isinstance(result["page"], Page)
    @pytest.mark.skip(reason="Временно отключено")
    def test_predict_very_large_image(self):
        """Тест с большим изображением"""
        detector = EAST()
        
        # Изображение 2000x3000
        large_img = np.random.randint(0, 255, (2000, 3000, 3), dtype=np.uint8)
        
        result = detector.predict(large_img)
        
        # Не должно быть ошибок
        assert isinstance(result["page"], Page)
    @pytest.mark.skip(reason="Временно отключено")
    def test_predict_grayscale_image(self):
        """Тест с черно-белым изображением"""
        detector = EAST()
        
        # Grayscale изображение (2D)
        gray_img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        
        # Должна быть ошибка или автоматическая конвертация
        try:
            result = detector.predict(gray_img)
            # Если отработало, проверяем результат
            assert isinstance(result["page"], Page)
        except (ValueError, AttributeError):
            # Ожидаемая ошибка для grayscale
            pass
    @pytest.mark.skip(reason="Временно отключено")
    def test_predict_black_image(self):
        """Тест с полностью черным изображением"""
        detector = EAST()
        
        black_img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result = detector.predict(black_img)
        page = result["page"]
        
        # Скорее всего не будет детекций, но не должно быть ошибок
        assert isinstance(page, Page)
    @pytest.mark.skip(reason="Временно отключено")
    def test_predict_white_image(self):
        """Тест с полностью белым изображением"""
        detector = EAST()
        
        white_img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        result = detector.predict(white_img)
        page = result["page"]
        
        assert isinstance(page, Page)
    @pytest.mark.skip(reason="Временно отключено")
    def test_predict_with_extreme_expand_ratios(self):
        """Тест с экстремальными коэффициентами расширения"""
        detector = EAST(expand_ratio_w=0.0, expand_ratio_h=0.0)
        
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        result = detector.predict(test_img)
        
        # Не должно быть ошибок даже с нулевым расширением
        assert isinstance(result["page"], Page)

@pytest.mark.skip(reason="Временно отключено")
class TestEASTIntegration:
    """Интеграционные тесты с реальным изображением"""

    @pytest.fixture
    def example_image_path(self):
        """Путь к примеру изображения если оно существует"""
        repo_root = Path(__file__).parent.parent.parent.parent
        image_path = repo_root / "example" / "ocr_example_image.jpg"
        
        if not image_path.exists():
            pytest.skip("Тестовое изображение example/ocr_example_image.jpg не найдено")
        
        return str(image_path)
    @pytest.mark.skip(reason="Временно отключено")
    def test_full_pipeline_with_real_image(self, example_image_path):
        """Полный тест pipeline с реальным изображением"""
        detector = EAST(score_thresh=0.3)
        
        result = detector.predict(
            example_image_path,
            vis=True,
            return_maps=True,
            sort_reading_order=True
        )
        
        # Проверяем все возвращаемые значения
        assert "page" in result
        assert "vis_image" in result
        assert "score_map" in result
        assert "geo_map" in result
        
        page = result["page"]
        assert isinstance(page, Page)
        assert len(page.blocks) > 0
        
        # Проверяем что есть детекции
        total_words = sum(len(block.words) for block in page.blocks)
        assert total_words > 0
        
        # Проверяем структуру слов
        first_word = page.blocks[0].words[0]
        assert hasattr(first_word, "polygon")
        assert len(first_word.polygon) == 4
        assert hasattr(first_word, "detection_confidence")
        
        # Проверяем визуализацию
        assert result["vis_image"] is not None
        assert isinstance(result["vis_image"], Image.Image)
        
        # Проверяем карты
        assert result["score_map"] is not None
        assert result["geo_map"] is not None
    @pytest.mark.skip(reason="Временно отключено")
    def test_different_target_sizes(self, example_image_path):
        """Тест с разными размерами target_size"""
        sizes = [640, 1280, 1920]
        
        for size in sizes:
            detector = EAST(target_size=size)
            result = detector.predict(example_image_path)
            
            # Все должны отработать без ошибок
            assert isinstance(result["page"], Page)
    @pytest.mark.skip(reason="Временно отключено")
    def test_consistency_across_runs(self):
        """Тест консистентности результатов"""
        detector = EAST(score_thresh=0.5)
        
        # Создаем фиксированное изображение
        np.random.seed(42)
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Запускаем дважды
        result1 = detector.predict(img)
        result2 = detector.predict(img)
        
        # Количество детекций должно быть одинаковым
        words1 = sum(len(b.words) for b in result1["page"].blocks)
        words2 = sum(len(b.words) for b in result2["page"].blocks)
        
        assert words1 == words2
