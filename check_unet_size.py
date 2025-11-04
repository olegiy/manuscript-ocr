"""Проверка размера U-Net до и после увеличения."""

import torch
from src.manuscript.recognizers._trba.model.unet import CompactUNet

# Старая версия
unet_old = CompactUNet(
    in_channels=3,
    features=[16, 32, 64],
    hard_binarize=False,
)

# Новая версия
unet_new = CompactUNet(
    in_channels=3,
    features=[32, 64, 128],
    hard_binarize=True,
)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


params_old = count_parameters(unet_old)
params_new = count_parameters(unet_new)

print("=" * 60)
print("U-Net Architecture Comparison")
print("=" * 60)
print(f"OLD [16, 32, 64]:  {params_old:,} parameters")
print(f"NEW [32, 64, 128]: {params_new:,} parameters")
print(f"Increase: {params_new / params_old:.2f}x ({params_new - params_old:,} params)")
print("=" * 60)

# Тест forward pass
x = torch.randn(1, 3, 32, 128)
print("\nTest forward pass:")
with torch.no_grad():
    mask_old = unet_old(x)
    mask_new = unet_new(x)

print(
    f"OLD output - min: {mask_old.min():.4f}, max: {mask_old.max():.4f}, unique values: {mask_old.unique().numel()}"
)
print(
    f"NEW output - min: {mask_new.min():.4f}, max: {mask_new.max():.4f}, unique values: {mask_new.unique().numel()}"
)
print(f"NEW is binary: {set(mask_new.unique().tolist()) == {0.0, 1.0}}")
