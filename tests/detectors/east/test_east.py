import pytest
import torch
import torch.nn as nn
from manuscript.detectors._east.east import (
    DecoderBlock,
    ResNetFeatureExtractor,
    FeatureMergingBranchResNet,
    OutputHead,
    EAST,
)


# --- Тесты для DecoderBlock ---


def test_decoder_block_initialization():
    """Тест инициализации DecoderBlock"""
    block = DecoderBlock(in_channels=256, mid_channels=128, out_channels=64)
    
    assert isinstance(block.conv1x1, nn.Sequential)
    assert isinstance(block.conv3x3, nn.Sequential)
    assert len(block.conv1x1) == 3  # Conv2d, BatchNorm2d, ReLU
    assert len(block.conv3x3) == 3


def test_decoder_block_forward():
    """Тест forward pass для DecoderBlock"""
    block = DecoderBlock(in_channels=256, mid_channels=128, out_channels=64)
    x = torch.randn(2, 256, 16, 16)
    
    output = block.forward(x)
    
    assert output.shape == (2, 64, 16, 16)
    assert output.dtype == torch.float32


def test_decoder_block_different_sizes():
    """Тест DecoderBlock с разными размерами входа"""
    block = DecoderBlock(in_channels=512, mid_channels=256, out_channels=128)
    
    # Маленький вход
    x_small = torch.randn(1, 512, 4, 4)
    out_small = block(x_small)
    assert out_small.shape == (1, 128, 4, 4)
    
    # Большой вход
    x_large = torch.randn(1, 512, 32, 32)
    out_large = block(x_large)
    assert out_large.shape == (1, 128, 32, 32)


def test_decoder_block_batch_processing():
    """Тест DecoderBlock с разными размерами batch"""
    block = DecoderBlock(in_channels=128, mid_channels=64, out_channels=32)
    
    for batch_size in [1, 4, 8]:
        x = torch.randn(batch_size, 128, 8, 8)
        output = block(x)
        assert output.shape == (batch_size, 32, 8, 8)


# --- Тесты для ResNetFeatureExtractor ---


def test_resnet_feature_extractor_resnet50():
    """Тест ResNetFeatureExtractor с ResNet50"""
    extractor = ResNetFeatureExtractor(backbone_name="resnet50", pretrained=False)
    
    assert hasattr(extractor, "extractor")
    
    # Тест forward
    x = torch.randn(1, 3, 224, 224)
    features = extractor(x)
    
    assert "res1" in features
    assert "res2" in features
    assert "res3" in features
    assert "res4" in features


def test_resnet_feature_extractor_resnet101():
    """Тест ResNetFeatureExtractor с ResNet101"""
    extractor = ResNetFeatureExtractor(backbone_name="resnet101", pretrained=False)
    
    x = torch.randn(1, 3, 224, 224)
    features = extractor(x)
    
    assert "res1" in features
    assert "res2" in features
    assert "res3" in features
    assert "res4" in features


def test_resnet_feature_extractor_invalid_backbone():
    """Тест ResNetFeatureExtractor с недопустимым backbone"""
    with pytest.raises(ValueError, match="Unsupported backbone"):
        ResNetFeatureExtractor(backbone_name="vgg16", pretrained=False)


def test_resnet_feature_extractor_feature_shapes():
    """Тест правильности размеров feature maps"""
    extractor = ResNetFeatureExtractor(backbone_name="resnet50", pretrained=False)
    x = torch.randn(2, 3, 256, 256)
    
    features = extractor(x)
    
    # ResNet50 channel sizes: 256, 512, 1024, 2048
    assert features["res1"].shape[1] == 256  # layer1
    assert features["res2"].shape[1] == 512  # layer2
    assert features["res3"].shape[1] == 1024  # layer3
    assert features["res4"].shape[1] == 2048  # layer4


def test_resnet_feature_extractor_freeze_first():
    """Тест замораживания первых слоев"""
    extractor = ResNetFeatureExtractor(
        backbone_name="resnet50", pretrained=False, freeze_first=True
    )
    
    # Проверяем что первые слои заморожены
    frozen_params = []
    trainable_params = []
    
    for name, param in extractor.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
        else:
            frozen_params.append(name)
    
    # Должны быть замороженные параметры
    assert len(frozen_params) > 0
    
    # Проверяем что conv1, bn1, layer1 заморожены
    for name in frozen_params:
        assert any(prefix in name for prefix in ["conv1", "bn1", "layer1"])


def test_resnet_feature_extractor_pretrained():
    """Тест загрузки pretrained весов"""
    # Без pretrained
    extractor_no_pretrain = ResNetFeatureExtractor(
        backbone_name="resnet50", pretrained=False
    )
    
    # С pretrained
    extractor_pretrained = ResNetFeatureExtractor(
        backbone_name="resnet50", pretrained=True
    )
    
    # Оба должны работать
    x = torch.randn(1, 3, 224, 224)
    out1 = extractor_no_pretrain(x)
    out2 = extractor_pretrained(x)
    
    assert out1["res1"].shape == out2["res1"].shape


# --- Тесты для FeatureMergingBranchResNet ---


def test_feature_merging_branch_initialization():
    """Тест инициализации FeatureMergingBranchResNet"""
    merger = FeatureMergingBranchResNet()
    
    assert isinstance(merger.block1, DecoderBlock)
    assert isinstance(merger.block2, DecoderBlock)
    assert isinstance(merger.block3, DecoderBlock)
    assert isinstance(merger.block4, DecoderBlock)


def test_feature_merging_branch_forward():
    """Тест forward pass для FeatureMergingBranchResNet"""
    merger = FeatureMergingBranchResNet()
    
    # Создаём фейковые feature maps
    feats = {
        "res1": torch.randn(2, 256, 64, 64),   # stride 4
        "res2": torch.randn(2, 512, 32, 32),   # stride 8
        "res3": torch.randn(2, 1024, 16, 16),  # stride 16
        "res4": torch.randn(2, 2048, 8, 8),    # stride 32
    }
    
    output = merger(feats)
    
    # Выход должен быть того же размера что res1 (stride 4)
    assert output.shape == (2, 32, 64, 64)


def test_feature_merging_branch_upsampling():
    """Тест что upsampling работает корректно"""
    merger = FeatureMergingBranchResNet()
    
    # Разные размеры входа
    feats = {
        "res1": torch.randn(1, 256, 128, 128),
        "res2": torch.randn(1, 512, 64, 64),
        "res3": torch.randn(1, 1024, 32, 32),
        "res4": torch.randn(1, 2048, 16, 16),
    }
    
    output = merger(feats)
    
    # Выход должен соответствовать res1
    assert output.shape == (1, 32, 128, 128)


def test_feature_merging_branch_batch_sizes():
    """Тест FeatureMergingBranchResNet с разными batch sizes"""
    merger = FeatureMergingBranchResNet()
    
    for batch_size in [1, 2, 4]:
        feats = {
            "res1": torch.randn(batch_size, 256, 32, 32),
            "res2": torch.randn(batch_size, 512, 16, 16),
            "res3": torch.randn(batch_size, 1024, 8, 8),
            "res4": torch.randn(batch_size, 2048, 4, 4),
        }
        
        output = merger(feats)
        assert output.shape == (batch_size, 32, 32, 32)


# --- Тесты для OutputHead ---


def test_output_head_initialization():
    """Тест инициализации OutputHead"""
    head = OutputHead()
    
    assert isinstance(head.score_map, nn.Conv2d)
    assert isinstance(head.geo_map, nn.Conv2d)
    assert head.score_map.out_channels == 1
    assert head.geo_map.out_channels == 8


def test_output_head_forward():
    """Тест forward pass для OutputHead"""
    head = OutputHead()
    x = torch.randn(2, 32, 64, 64)
    
    score, geometry = head(x)
    
    assert score.shape == (2, 1, 64, 64)
    assert geometry.shape == (2, 8, 64, 64)


def test_output_head_score_range():
    """Тест что score в диапазоне [0, 1] из-за sigmoid"""
    head = OutputHead()
    x = torch.randn(4, 32, 16, 16)
    
    score, geometry = head(x)
    
    assert score.min() >= 0
    assert score.max() <= 1


def test_output_head_different_sizes():
    """Тест OutputHead с разными размерами входа"""
    head = OutputHead()
    
    sizes = [(8, 8), (16, 16), (32, 32), (64, 64)]
    for h, w in sizes:
        x = torch.randn(1, 32, h, w)
        score, geometry = head(x)
        
        assert score.shape == (1, 1, h, w)
        assert geometry.shape == (1, 8, h, w)


# --- Тесты для полной модели EAST ---


def test_east_initialization_default():
    """Тест инициализации EAST с параметрами по умолчанию"""
    model = EAST(pretrained_backbone=False)
    
    assert isinstance(model.backbone, ResNetFeatureExtractor)
    assert isinstance(model.decoder, FeatureMergingBranchResNet)
    assert isinstance(model.output_head, OutputHead)
    assert model.score_scale == 0.25
    assert model.geo_scale == 0.25


def test_east_initialization_resnet101():
    """Тест инициализации EAST с ResNet101"""
    model = EAST(backbone_name="resnet101", pretrained_backbone=False)
    
    x = torch.randn(1, 3, 256, 256)
    output = model(x)
    
    assert "score" in output
    assert "geometry" in output


def test_east_initialization_freeze_first():
    """Тест инициализации EAST с заморозкой первых слоёв"""
    model = EAST(
        backbone_name="resnet50", pretrained_backbone=False, freeze_first=True
    )
    
    # Проверяем что есть замороженные параметры
    frozen_count = sum(
        1 for param in model.backbone.parameters() if not param.requires_grad
    )
    assert frozen_count > 0


def test_east_forward():
    """Тест forward pass для EAST"""
    model = EAST(pretrained_backbone=False)
    x = torch.randn(2, 3, 512, 512)
    
    output = model(x)
    
    assert "score" in output
    assert "geometry" in output
    assert output["score"].shape[0] == 2  # batch size
    assert output["geometry"].shape[0] == 2
    assert output["score"].shape[1] == 1  # score channel
    assert output["geometry"].shape[1] == 8  # geometry channels


def test_east_output_shapes():
    """Тест правильности размеров выходных тензоров"""
    model = EAST(pretrained_backbone=False)
    
    # Разные размеры входа
    input_sizes = [(256, 256), (512, 512), (640, 640)]
    
    for h, w in input_sizes:
        x = torch.randn(1, 3, h, w)
        output = model(x)
        
        # Выход в 4 раза меньше из-за stride
        expected_h = h // 4
        expected_w = w // 4
        
        assert output["score"].shape == (1, 1, expected_h, expected_w)
        assert output["geometry"].shape == (1, 8, expected_h, expected_w)


def test_east_batch_processing():
    """Тест EAST с разными размерами batch"""
    model = EAST(pretrained_backbone=False)
    
    for batch_size in [1, 2, 4, 8]:
        x = torch.randn(batch_size, 3, 256, 256)
        output = model(x)
        
        assert output["score"].shape[0] == batch_size
        assert output["geometry"].shape[0] == batch_size


def test_east_gradient_flow():
    """Тест что градиенты проходят через модель"""
    model = EAST(pretrained_backbone=False)
    x = torch.randn(1, 3, 256, 256, requires_grad=True)
    
    output = model(x)
    loss = output["score"].sum() + output["geometry"].sum()
    loss.backward()
    
    # Проверяем что градиенты есть
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_east_eval_mode():
    """Тест переключения EAST в eval режим"""
    model = EAST(pretrained_backbone=False)
    
    # Train mode
    model.train()
    x = torch.randn(2, 3, 256, 256)
    out_train = model(x)
    
    # Eval mode
    model.eval()
    with torch.no_grad():
        out_eval = model(x)
    
    # Выходы должны иметь одинаковую форму
    assert out_train["score"].shape == out_eval["score"].shape
    assert out_train["geometry"].shape == out_eval["geometry"].shape


def test_east_no_pretrained_backbone():
    """Тест EAST без pretrained backbone"""
    model = EAST(backbone_name="resnet50", pretrained_backbone=False)
    
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    
    assert output["score"].shape[1] == 1
    assert output["geometry"].shape[1] == 8


def test_east_pretrained_backbone():
    """Тест EAST с pretrained backbone"""
    model = EAST(backbone_name="resnet50", pretrained_backbone=True)
    
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    
    assert output["score"].shape[1] == 1
    assert output["geometry"].shape[1] == 8


def test_east_pretrained_model_path_nonexistent(tmp_path):
    """Тест загрузки несуществующей модели"""
    fake_path = tmp_path / "nonexistent_model.pth"
    
    with pytest.raises(FileNotFoundError):
        EAST(pretrained_backbone=False, pretrained_model_path=str(fake_path))


def test_east_pretrained_model_path_valid(tmp_path):
    """Тест загрузки существующей модели"""
    # Создаём модель и сохраняем её
    model1 = EAST(pretrained_backbone=False)
    model_path = tmp_path / "test_model.pth"
    torch.save(model1.state_dict(), model_path)
    
    # Загружаем в новую модель
    model2 = EAST(pretrained_backbone=False, pretrained_model_path=str(model_path))
    
    # Проверяем что модель работает
    x = torch.randn(1, 3, 256, 256)
    output = model2(x)
    
    assert "score" in output
    assert "geometry" in output


def test_east_parameters_count():
    """Тест что у модели есть обучаемые параметры"""
    model = EAST(pretrained_backbone=False)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    assert total_params > 0
    assert trainable_params > 0


def test_east_different_backbones_compatibility():
    """Тест что разные backbones дают совместимые выходы"""
    model50 = EAST(backbone_name="resnet50", pretrained_backbone=False)
    model101 = EAST(backbone_name="resnet101", pretrained_backbone=False)
    
    x = torch.randn(1, 3, 256, 256)
    
    out50 = model50(x)
    out101 = model101(x)
    
    # Выходы должны иметь одинаковую форму
    assert out50["score"].shape == out101["score"].shape
    assert out50["geometry"].shape == out101["geometry"].shape


# --- Граничные случаи ---


def test_decoder_block_single_pixel():
    """Тест DecoderBlock с минимальным входом 1x1"""
    block = DecoderBlock(in_channels=256, mid_channels=128, out_channels=64)
    block.eval()  # BatchNorm не работает с 1x1 в train mode
    x = torch.randn(1, 256, 1, 1)
    
    with torch.no_grad():
        output = block(x)
    assert output.shape == (1, 64, 1, 1)


def test_feature_merging_small_features():
    """Тест FeatureMergingBranchResNet с маленькими feature maps"""
    merger = FeatureMergingBranchResNet()
    merger.eval()  # BatchNorm не работает с 1x1 в train mode
    
    feats = {
        "res1": torch.randn(1, 256, 8, 8),
        "res2": torch.randn(1, 512, 4, 4),
        "res3": torch.randn(1, 1024, 2, 2),
        "res4": torch.randn(1, 2048, 1, 1),
    }
    
    with torch.no_grad():
        output = merger(feats)
    assert output.shape == (1, 32, 8, 8)


def test_east_minimum_input_size():
    """Тест EAST с минимальным разумным размером входа"""
    model = EAST(pretrained_backbone=False)
    
    # Минимальный размер для ResNet (должен быть кратен 32)
    x = torch.randn(1, 3, 128, 128)
    output = model(x)
    
    assert output["score"].shape == (1, 1, 32, 32)
    assert output["geometry"].shape == (1, 8, 32, 32)


def test_east_non_square_input():
    """Тест EAST с прямоугольным входом"""
    model = EAST(pretrained_backbone=False)
    
    x = torch.randn(1, 3, 256, 512)  # height != width
    output = model(x)
    
    assert output["score"].shape == (1, 1, 64, 128)
    assert output["geometry"].shape == (1, 8, 64, 128)


def test_east_cpu_device():
    """Тест что EAST работает на CPU"""
    model = EAST(pretrained_backbone=False)
    model = model.cpu()
    
    x = torch.randn(1, 3, 256, 256)
    output = model(x)
    
    assert output["score"].device.type == "cpu"
    assert output["geometry"].device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_east_cuda_device():
    """Тест что EAST работает на CUDA"""
    model = EAST(pretrained_backbone=False)
    model = model.cuda()
    
    x = torch.randn(1, 3, 256, 256).cuda()
    output = model(x)
    
    assert output["score"].device.type == "cuda"
    assert output["geometry"].device.type == "cuda"
