"""
Тесты корректности установки пакета manuscript-ocr
Проверяет базовые импорты и зависимости
"""

import pytest

@pytest.mark.skip(reason="Временно отключено")
class TestInstallation:
    """Тесты корректности установки"""
    @pytest.mark.skip(reason="Временно отключено")
    def test_basic_imports(self):
        """Тест базовых импортов детекторов"""
        from manuscript.detectors import EAST

        assert EAST is not None
    @pytest.mark.skip(reason="Временно отключено")
    def test_recognizers_imports(self):
        """Тест базовых импортов распознавателей"""
        from manuscript.recognizers import TRBA

        assert TRBA is not None
    @pytest.mark.skip(reason="Временно отключено")
    def test_pytorch_installation(self):
        """Тест установки PyTorch"""
        import torch

        print(f"PyTorch версия: {torch.__version__}")
        print(f"CUDA доступна: {torch.cuda.is_available()}")

        # Проверяем минимальную версию
        version_parts = torch.__version__.split(".")
        major = int(version_parts[0])
        minor = int(version_parts[1])

        assert major >= 1
        if major == 1:
            assert minor >= 11
    @pytest.mark.skip(reason="Временно отключено")
    def test_other_dependencies(self):
        """Тест остальных зависимостей"""
        import numpy
        import cv2
        import torch_optimizer
        import tensorboard

        print("✅ Все зависимости импортированы успешно")
