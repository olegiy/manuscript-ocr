"""
Диагностика предобработки изображений: сравнение PTH и ONNX.
"""

import cv2
import numpy as np
from manuscript.recognizers import TRBA
from manuscript.recognizers._trba.data.transforms import get_val_transform
from pathlib import Path

# Тестовое изображение
TEST_IMAGE = r"C:\shared\orig_cyrillic\test\images_group_8_127_6750.png"
IMG_H = 64
IMG_W = 256

print("=" * 80)
print("🔬 ДИАГНОСТИКА ПРЕДОБРАБОТКИ: PTH vs ONNX")
print("=" * 80)
print(f"Test image: {TEST_IMAGE}")
print(f"Target size: {IMG_H}x{IMG_W}")
print()

# ============================================
# 1. PTH PREPROCESSING (через albumentations)
# ============================================

print("1️⃣ PTH PREPROCESSING (albumentations)")
print("-" * 80)

transform = get_val_transform(img_h=IMG_H, img_w=IMG_W)

img = cv2.imread(TEST_IMAGE)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(f"   Original image shape: {img.shape}, dtype: {img.dtype}")
print(f"   Original value range: [{img.min()}, {img.max()}]")

transformed = transform(image=img)
img_pth = transformed["image"].numpy()  # [3, H, W]

print(f"   Transformed shape: {img_pth.shape}, dtype: {img_pth.dtype}")
print(f"   Transformed value range: [{img_pth.min():.4f}, {img_pth.max():.4f}]")
print(f"   Mean per channel: {img_pth.mean(axis=(1, 2))}")
print(f"   Std per channel: {img_pth.std(axis=(1, 2))}")
print()

# ============================================
# 2. ONNX PREPROCESSING (через TRBA._preprocess_image)
# ============================================

print("2️⃣ ONNX PREPROCESSING (TRBA._preprocess_image)")
print("-" * 80)

# Создаем TRBA объект (любой, нам нужен только метод препроцессинга)
recognizer = TRBA(
    weights_path=r"C:\Users\USER\Desktop\trba_exp_lite\trba_exp_lite_FIXED.onnx",
    config_path=r"C:\Users\USER\Desktop\trba_exp_lite\config.json",
    charset_path=r"C:\Users\USER\Desktop\trba_exp_lite\charset.txt"
)

img_onnx_batch = recognizer._preprocess_image(TEST_IMAGE)  # [1, 3, H, W]
img_onnx = img_onnx_batch[0]  # [3, H, W]

print(f"   Preprocessed shape: {img_onnx.shape}, dtype: {img_onnx.dtype}")
print(f"   Preprocessed value range: [{img_onnx.min():.4f}, {img_onnx.max():.4f}]")
print(f"   Mean per channel: {img_onnx.mean(axis=(1, 2))}")
print(f"   Std per channel: {img_onnx.std(axis=(1, 2))}")
print()

# ============================================
# 3. СРАВНЕНИЕ
# ============================================

print("3️⃣ COMPARISON")
print("-" * 80)

# Проверяем идентичность
diff = np.abs(img_pth - img_onnx)
max_diff = diff.max()
mean_diff = diff.mean()

print(f"   Max absolute difference: {max_diff:.8f}")
print(f"   Mean absolute difference: {mean_diff:.8f}")
print(f"   Are they identical? {np.allclose(img_pth, img_onnx, atol=1e-6)}")

if max_diff > 1e-6:
    print(f"\n   ⚠️  DIFFERENCE DETECTED!")
    print(f"   Location of max difference:")
    max_idx = np.unravel_index(diff.argmax(), diff.shape)
    print(f"      Channel={max_idx[0]}, Y={max_idx[1]}, X={max_idx[2]}")
    print(f"      PTH value:  {img_pth[max_idx]:.8f}")
    print(f"      ONNX value: {img_onnx[max_idx]:.8f}")
    print(f"      Difference: {diff[max_idx]:.8f}")
    
    # Проверяем где больше всего различий
    large_diffs = diff > 1e-4
    if large_diffs.any():
        print(f"\n   Pixels with diff > 1e-4: {large_diffs.sum()} / {diff.size} ({large_diffs.sum()/diff.size*100:.2f}%)")
else:
    print(f"\n   ✅ Preprocessing is IDENTICAL!")

# ============================================
# 4. ДЕТАЛЬНОЕ СРАВНЕНИЕ ШАГОВ
# ============================================

print("\n4️⃣ STEP-BY-STEP COMPARISON")
print("-" * 80)

# Загружаем изображение снова
img_test = cv2.imread(TEST_IMAGE)
img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)
h, w = img_test.shape[:2]
print(f"   Original size: {w}x{h}")

# Шаг 1: Resize
scale = min(IMG_H / max(h, 1), IMG_W / max(w, 1))
new_w = max(1, int(round(w * scale)))
new_h = max(1, int(round(h * scale)))
print(f"   Scale: {scale:.4f}")
print(f"   Resized size: {new_w}x{new_h}")

# Шаг 2: Canvas placement
y_offset = (IMG_H - new_h) // 2
x_offset = 0
print(f"   Canvas placement: x_offset={x_offset}, y_offset={y_offset}")

# Шаг 3: Проверяем canvas dtype
canvas_uint8 = np.full((IMG_H, IMG_W, 3), 255, dtype=np.uint8)
canvas_orig = np.full((IMG_H, IMG_W, 3), 255, dtype=img_test.dtype)
print(f"   Canvas dtype (ONNX): {canvas_uint8.dtype}")
print(f"   Canvas dtype (should be): {canvas_orig.dtype}")
print(f"   Input image dtype: {img_test.dtype}")

# Шаг 4: Нормализация
test_value = 255
norm_method1 = (test_value - 127.5) / 127.5  # ONNX way
norm_method2 = (test_value / 255.0 - 0.5) / 0.5  # Albumentations way
print(f"\n   Normalization of value {test_value}:")
print(f"      ONNX way: (x - 127.5) / 127.5 = {norm_method1:.8f}")
print(f"      Albumentations way: (x/255 - 0.5) / 0.5 = {norm_method2:.8f}")
print(f"      Difference: {abs(norm_method1 - norm_method2):.8f}")

print("\n" + "=" * 80)
