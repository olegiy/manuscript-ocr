"""
Тесты для EASTDataset и геометрических функций
"""

import pytest
import json
import numpy as np
import cv2
import torch
from pathlib import Path

from manuscript.detectors._east.dataset import (
    order_vertices_clockwise,
    shrink_poly,
    EASTDataset,
)


# ============================================================================
# Тесты геометрических функций
# ============================================================================


class TestGeometricFunctions:
    """Тесты для вспомогательных геометрических функций"""

    def test_order_vertices_clockwise_square(self):
        """Тест упорядочивания вершин квадрата по часовой стрелке"""
        # Неупорядоченные точки квадрата
        poly = [[100, 100], [100, 0], [0, 0], [0, 100]]
        ordered = order_vertices_clockwise(poly)

        # Проверяем формат
        assert ordered.shape == (4, 2)
        assert ordered.dtype == np.float32

        # Проверяем порядок: TL, TR, BR, BL
        # top-left должен быть левее top-right
        assert ordered[0][0] < ordered[1][0]
        # bottom-right должен быть ниже top-right
        assert ordered[2][1] > ordered[1][1]
        # bottom-left должен быть левее bottom-right
        assert ordered[3][0] < ordered[2][0]

    def test_order_vertices_clockwise_rectangle(self):
        """Тест упорядочивания вершин прямоугольника"""
        # Прямоугольник 200x100
        poly = [[0, 0], [200, 0], [200, 100], [0, 100]]
        ordered = order_vertices_clockwise(poly)

        # Проверяем что top-left в начале
        assert np.allclose(ordered[0], [0, 0])
        # И bottom-right в правильной позиции
        assert np.allclose(ordered[2], [200, 100])

    def test_order_vertices_clockwise_rotated(self):
        """Тест упорядочивания повернутого квадрата"""
        # Квадрат повернутый на 45 градусов
        poly = [[50, 0], [100, 50], [50, 100], [0, 50]]
        ordered = order_vertices_clockwise(poly)

        assert ordered.shape == (4, 2)
        # Проверяем что вершины упорядочены
        assert len(ordered) == 4

    def test_shrink_poly_basic(self):
        """Тест базового сжатия полигона"""
        # Квадрат 100x100
        quad = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)

        shrunk = shrink_poly(quad, shrink_ratio=0.3)

        # Проверяем формат
        assert shrunk.shape == (4, 2)
        assert shrunk.dtype == np.float32

        # Сжатый квадрат должен быть меньше оригинала
        # top-left сдвинут вправо и вниз
        assert shrunk[0][0] > quad[0][0]
        assert shrunk[0][1] > quad[0][1]

        # bottom-right сдвинут влево и вверх
        assert shrunk[2][0] < quad[2][0]
        assert shrunk[2][1] < quad[2][1]

    def test_shrink_poly_different_ratios(self):
        """Тест сжатия с разными коэффициентами"""
        quad = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)

        shrunk_small = shrink_poly(quad, shrink_ratio=0.1)
        shrunk_large = shrink_poly(quad, shrink_ratio=0.5)

        # Больший коэффициент = больше сжатие
        # Расстояние от top-left до центра должно быть больше при большем shrink_ratio
        center = np.array([50, 50])
        dist_small = np.linalg.norm(shrunk_small[0] - center)
        dist_large = np.linalg.norm(shrunk_large[0] - center)

        assert dist_large < dist_small

    def test_shrink_poly_invalid_vertices(self):
        """Тест ошибки при неправильном количестве вершин"""
        # Треугольник (3 вершины)
        triangle = np.array([[0, 0], [100, 0], [50, 100]], dtype=np.float32)

        with pytest.raises(ValueError, match="Expected quadrilateral with 4 vertices"):
            shrink_poly(triangle)

    def test_shrink_poly_clockwise_order(self):
        """Тест что сжатие работает с неупорядоченными вершинами"""
        # Вершины в случайном порядке
        quad = np.array([[100, 0], [100, 100], [0, 100], [0, 0]], dtype=np.float32)

        shrunk = shrink_poly(quad, shrink_ratio=0.2)

        # Не должно быть ошибок
        assert shrunk.shape == (4, 2)


# ============================================================================
# Тесты для EASTDataset
# ============================================================================


class TestEASTDataset:
    """Тесты для класса EASTDataset"""

    @pytest.fixture
    def simple_dataset(self, tmp_path):
        """Создает простой датасет с одним изображением"""
        annotations = {
            "images": [
                {"id": 1, "file_name": "test.jpg", "width": 640, "height": 480}
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "segmentation": [[10, 10, 100, 10, 100, 50, 10, 50]],
                }
            ],
        }

        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(json.dumps(annotations), encoding="utf-8")

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "test.jpg"), img)

        return str(img_dir), str(ann_file)

    def test_east_dataset_initialization(self, simple_dataset):
        """Тест базовой инициализации датасета"""
        img_dir, ann_file = simple_dataset

        dataset = EASTDataset(
            images_folder=img_dir,
            coco_annotation_file=ann_file,
            target_size=512,
            score_geo_scale=0.25,
        )

        assert len(dataset) == 1
        assert dataset.target_size == 512
        assert dataset.score_geo_scale == 0.25
        assert dataset.images_folder == img_dir

    def test_east_dataset_len(self, simple_dataset):
        """Тест метода __len__"""
        img_dir, ann_file = simple_dataset
        dataset = EASTDataset(img_dir, ann_file)

        assert len(dataset) == 1
        assert dataset.__len__() == 1

    def test_east_dataset_getitem(self, simple_dataset):
        """Тест получения элемента из датасета"""
        img_dir, ann_file = simple_dataset
        dataset = EASTDataset(img_dir, ann_file, target_size=512, score_geo_scale=0.25)

        img_tensor, target = dataset[0]

        # Проверяем изображение
        assert isinstance(img_tensor, torch.Tensor)
        assert img_tensor.shape == (3, 512, 512)  # C, H, W

        # Проверяем target
        assert isinstance(target, dict)
        assert "score_map" in target
        assert "geo_map" in target
        assert "rboxes" in target

        # Проверяем размеры карт
        assert target["score_map"].shape[1:] == (128, 128)  # 512 * 0.25
        assert target["geo_map"].shape == (8, 128, 128)

        # Проверяем rboxes
        assert target["rboxes"].shape[1] == 5  # (cx, cy, w, h, angle)

    def test_east_dataset_different_target_size(self, simple_dataset):
        """Тест с разными размерами изображения"""
        img_dir, ann_file = simple_dataset

        dataset_512 = EASTDataset(img_dir, ann_file, target_size=512)
        dataset_1024 = EASTDataset(img_dir, ann_file, target_size=1024)

        img_512, _ = dataset_512[0]
        img_1024, _ = dataset_1024[0]

        assert img_512.shape == (3, 512, 512)
        assert img_1024.shape == (3, 1024, 1024)

    def test_east_dataset_filter_invalid(self, tmp_path):
        """Тест фильтрации невалидных аннотаций"""
        annotations = {
            "images": [
                {"id": 1, "file_name": "valid.jpg", "width": 640, "height": 480},
                {"id": 2, "file_name": "invalid.jpg", "width": 640, "height": 480},
                {"id": 3, "file_name": "no_ann.jpg", "width": 640, "height": 480},
            ],
            "annotations": [
                # Валидная аннотация (4 точки)
                {
                    "id": 1,
                    "image_id": 1,
                    "segmentation": [[10, 10, 100, 10, 100, 50, 10, 50]],
                },
                # Невалидная (меньше 4 точек)
                {"id": 2, "image_id": 2, "segmentation": [[10, 10, 100, 10]]},
                # id 3 вообще без аннотаций
            ],
        }

        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(json.dumps(annotations), encoding="utf-8")

        img_dir = tmp_path / "images"
        img_dir.mkdir()

        for fname in ["valid.jpg", "invalid.jpg", "no_ann.jpg"]:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / fname), img)

        with pytest.warns(
            UserWarning, match="найдено.*изображений без годных квадов"
        ):
            dataset = EASTDataset(str(img_dir), str(ann_file))

        # Должен остаться только 1 валидный
        assert len(dataset) == 1

    def test_east_dataset_missing_image(self, tmp_path):
        """Тест ошибки при отсутствующем изображении"""
        annotations = {
            "images": [
                {"id": 1, "file_name": "missing.jpg", "width": 640, "height": 480}
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "segmentation": [[10, 10, 100, 10, 100, 50, 10, 50]],
                }
            ],
        }

        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(json.dumps(annotations), encoding="utf-8")

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        # НЕ создаем изображение

        dataset = EASTDataset(str(img_dir), str(ann_file))

        with pytest.raises(FileNotFoundError, match="Image not found"):
            _ = dataset[0]

    def test_east_dataset_multiple_quads(self, tmp_path):
        """Тест с несколькими квадратами на одном изображении"""
        annotations = {
            "images": [
                {"id": 1, "file_name": "multi.jpg", "width": 640, "height": 480}
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "segmentation": [[10, 10, 50, 10, 50, 30, 10, 30]],
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "segmentation": [[100, 100, 200, 100, 200, 150, 100, 150]],
                },
                {
                    "id": 3,
                    "image_id": 1,
                    "segmentation": [[300, 200, 400, 200, 400, 300, 300, 300]],
                },
            ],
        }

        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(json.dumps(annotations), encoding="utf-8")

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "multi.jpg"), img)

        dataset = EASTDataset(str(img_dir), str(ann_file))
        _, target = dataset[0]

        # Должно быть 3 rbox
        assert target["rboxes"].shape[0] == 3

    def test_east_dataset_empty_annotations(self, tmp_path):
        """Тест с изображением без аннотаций"""
        annotations = {
            "images": [
                {"id": 1, "file_name": "empty.jpg", "width": 640, "height": 480}
            ],
            "annotations": [],
        }

        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(json.dumps(annotations), encoding="utf-8")

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "empty.jpg"), img)

        with pytest.warns(UserWarning):
            dataset = EASTDataset(str(img_dir), str(ann_file))

        # Должен быть пустым
        assert len(dataset) == 0

    def test_east_dataset_custom_transform(self, simple_dataset):
        """Тест кастомного transform"""
        import torchvision.transforms as transforms

        img_dir, ann_file = simple_dataset

        # Кастомный transform без нормализации
        custom_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )

        dataset = EASTDataset(img_dir, ann_file, transform=custom_transform)

        img_tensor, _ = dataset[0]
        assert img_tensor.shape == (3, 512, 512)
        # Без нормализации значения должны быть в [0, 1]
        assert img_tensor.min() >= 0
        assert img_tensor.max() <= 1

    def test_east_dataset_dataset_name(self, simple_dataset):
        """Тест атрибута dataset_name"""
        img_dir, ann_file = simple_dataset

        # Автоматическое имя из папки
        dataset = EASTDataset(img_dir, ann_file)
        assert dataset.dataset_name == Path(img_dir).stem

        # Кастомное имя
        dataset_custom = EASTDataset(img_dir, ann_file, dataset_name="my_dataset")
        assert dataset_custom.dataset_name == "my_dataset"

    def test_compute_quad_maps(self, simple_dataset):
        """Тест генерации карт score и geo"""
        img_dir, ann_file = simple_dataset
        dataset = EASTDataset(img_dir, ann_file, target_size=512, score_geo_scale=0.25)

        # Создаем квадрат
        quad = np.array(
            [[50, 50], [150, 50], [150, 150], [50, 150]], dtype=np.float32
        )

        score_map, geo_map = dataset.compute_quad_maps([quad])

        # Проверки размеров
        assert score_map.shape == (128, 128)
        assert geo_map.shape == (8, 128, 128)

        # Должна быть хотя бы одна единица в score_map (внутри квадрата)
        assert np.sum(score_map) > 0
        assert score_map.max() == 1.0

    def test_compute_quad_maps_multiple_quads(self, simple_dataset):
        """Тест генерации карт с несколькими квадратами"""
        img_dir, ann_file = simple_dataset
        dataset = EASTDataset(img_dir, ann_file, target_size=512, score_geo_scale=0.25)

        quad1 = np.array(
            [[20, 20], [80, 20], [80, 60], [20, 60]], dtype=np.float32
        )
        quad2 = np.array(
            [[100, 100], [200, 100], [200, 180], [100, 180]], dtype=np.float32
        )

        score_map, geo_map = dataset.compute_quad_maps([quad1, quad2])

        # Обе области должны быть помечены
        assert np.sum(score_map) > 0
        assert score_map.shape == (128, 128)
        assert geo_map.shape == (8, 128, 128)

    def test_compute_quad_maps_empty(self, simple_dataset):
        """Тест генерации карт без квадратов"""
        img_dir, ann_file = simple_dataset
        dataset = EASTDataset(img_dir, ann_file, target_size=512, score_geo_scale=0.25)

        score_map, geo_map = dataset.compute_quad_maps([])

        # Должны быть нулевые карты
        assert score_map.shape == (128, 128)
        assert geo_map.shape == (8, 128, 128)
        assert np.sum(score_map) == 0

    def test_east_dataset_segmentation_variants(self, tmp_path):
        """Тест различных форматов segmentation"""
        annotations = {
            "images": [
                {"id": 1, "file_name": "test.jpg", "width": 640, "height": 480}
            ],
            "annotations": [
                # Вариант 1: простой список
                {
                    "id": 1,
                    "image_id": 1,
                    "segmentation": [[10, 10, 50, 10, 50, 30, 10, 30]],
                },
                # Вариант 2: аннотация без segmentation
                {"id": 2, "image_id": 1},
                # Вариант 3: пустая segmentation
                {"id": 3, "image_id": 1, "segmentation": []},
            ],
        }

        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(json.dumps(annotations), encoding="utf-8")

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "test.jpg"), img)

        dataset = EASTDataset(str(img_dir), str(ann_file))
        _, target = dataset[0]

        # Должен быть только 1 валидный quad
        assert target["rboxes"].shape[0] >= 1

    def test_east_dataset_scaling(self, tmp_path):
        """Тест правильности масштабирования координат"""
        # Изображение 640x480, target_size=512
        annotations = {
            "images": [{"id": 1, "file_name": "test.jpg", "width": 640, "height": 480}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    # Квадрат в исходных координатах
                    "segmentation": [[100, 100, 200, 100, 200, 200, 100, 200]],
                }
            ],
        }

        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(json.dumps(annotations), encoding="utf-8")

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        cv2.imwrite(str(img_dir / "test.jpg"), img)

        dataset = EASTDataset(str(img_dir), str(ann_file), target_size=512)
        _, target = dataset[0]

        # Проверяем что квадраты были масштабированы
        assert target["rboxes"].shape[0] == 1
        # Координаты должны быть в диапазоне [0, 512]
        rbox = target["rboxes"][0]
        assert 0 <= rbox[0] <= 512  # cx
        assert 0 <= rbox[1] <= 512  # cy


class TestEASTDatasetEdgeCases:
    """Тесты граничных случаев для EASTDataset"""

    def test_very_small_quad(self, tmp_path):
        """Тест с очень маленьким квадратом"""
        annotations = {
            "images": [{"id": 1, "file_name": "test.jpg", "width": 640, "height": 480}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "segmentation": [[10, 10, 11, 10, 11, 11, 10, 11]],  # 1x1 px
                }
            ],
        }

        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(json.dumps(annotations), encoding="utf-8")

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "test.jpg"), img)

        dataset = EASTDataset(str(img_dir), str(ann_file))
        # Должен обработаться без ошибок
        img_tensor, target = dataset[0]
        assert img_tensor.shape == (3, 512, 512)

    def test_quad_at_image_boundary(self, tmp_path):
        """Тест с квадратом на границе изображения"""
        annotations = {
            "images": [{"id": 1, "file_name": "test.jpg", "width": 640, "height": 480}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "segmentation": [
                        [0, 0, 100, 0, 100, 100, 0, 100]
                    ],  # В углу изображения
                }
            ],
        }

        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(json.dumps(annotations), encoding="utf-8")

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "test.jpg"), img)

        dataset = EASTDataset(str(img_dir), str(ann_file))
        img_tensor, target = dataset[0]

        # Должен обработаться корректно
        assert target["rboxes"].shape[0] >= 1

    def test_different_score_geo_scales(self, simple_dataset):
        """Тест с разными значениями score_geo_scale"""
        img_dir, ann_file = simple_dataset

        dataset_025 = EASTDataset(img_dir, ann_file, score_geo_scale=0.25)
        dataset_050 = EASTDataset(img_dir, ann_file, score_geo_scale=0.5)

        _, target_025 = dataset_025[0]
        _, target_050 = dataset_050[0]

        # Размеры карт должны различаться
        assert target_025["score_map"].shape[1:] == (128, 128)  # 512 * 0.25
        assert target_050["score_map"].shape[1:] == (256, 256)  # 512 * 0.5

    @pytest.fixture
    def simple_dataset(self, tmp_path):
        """Создает простой датасет с одним изображением"""
        annotations = {
            "images": [
                {"id": 1, "file_name": "test.jpg", "width": 640, "height": 480}
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "segmentation": [[10, 10, 100, 10, 100, 50, 10, 50]],
                }
            ],
        }

        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(json.dumps(annotations), encoding="utf-8")

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "test.jpg"), img)

        return str(img_dir), str(ann_file)
