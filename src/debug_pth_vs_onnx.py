"""
Диагностический скрипт для сравнения PTH и ONNX инференса TRBA.
Цель: найти различия в выходе моделей.
"""

import os
import json
import torch
import cv2
import numpy as np
from pathlib import Path

from manuscript.recognizers import TRBA
from manuscript.recognizers._trba.model.model import TRBAModel
from manuscript.recognizers._trba.data.transforms import load_charset, get_val_transform

# ============================================
# НАСТРОЙКИ
# ============================================

# Пути
WEIGHTS_PTH = r"C:\Users\USER\Desktop\trba_exp_lite\best_acc_weights.pth"
WEIGHTS_ONNX = r"C:\Users\USER\Desktop\trba_exp_lite\trba_exp_lite.onnx"
CONFIG_PATH = r"C:\Users\USER\Desktop\trba_exp_lite\config.json"
CHARSET_PATH = r"C:\Users\USER\Desktop\trba_exp_lite\charset.txt"

# Тестовое изображение
TEST_IMAGE = r"C:\shared\orig_cyrillic\test\test4.png"

# Функция для чтения с кириллицей в пути
def imread_unicode(path):
    """Читает изображение с Unicode путём (кириллица)"""
    with open(path, 'rb') as f:
        arr = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 80)
print("🔬 ДИАГНОСТИКА: Сравнение PTH vs ONNX инференса")
print("=" * 80)
print(f"PTH weights:  {WEIGHTS_PTH}")
print(f"ONNX weights: {WEIGHTS_ONNX}")
print(f"Test image:   {TEST_IMAGE}")
print(f"Device:       {DEVICE}")
print("=" * 80)
print()

# ============================================
# ЗАГРУЗКА КОНФИГУРАЦИИ
# ============================================

print("📋 Loading config...")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

print(f"   Config keys: {list(config.keys())}")
print(f"   Full config:")
for k, v in config.items():
    print(f"      {k}: {v}")

img_h = config.get("img_h", 64)
img_w = config.get("img_w", 256)
max_len = config.get("max_len", 40)
hidden_size = config.get("hidden_size", 256)
num_encoder_layers = config.get("num_encoder_layers", 2)
cnn_in_channels = config.get("cnn_in_channels", 3)
cnn_out_channels = config.get("cnn_out_channels", 512)
cnn_backbone = config.get("cnn_backbone", "seresnet31")

print(f"📄 Config: img_size={img_h}x{img_w}, max_len={max_len}, hidden={hidden_size}")
print()

# ============================================
# ЗАГРУЗКА CHARSET
# ============================================

itos, stoi = load_charset(CHARSET_PATH)
num_classes = len(itos)
print(f"📚 Charset: {num_classes} classes")
print()

# ============================================
# 1. PTH ИНФЕРЕНС
# ============================================

print("=" * 80)
print("1️⃣ PTH MODEL INFERENCE")
print("=" * 80)

# Создание модели
model_pth = TRBAModel(
    num_classes=num_classes,
    hidden_size=hidden_size,
    num_encoder_layers=num_encoder_layers,
    img_h=img_h,
    img_w=img_w,
    cnn_in_channels=cnn_in_channels,
    cnn_out_channels=cnn_out_channels,
    cnn_backbone=cnn_backbone,
    sos_id=stoi["<SOS>"],
    eos_id=stoi["<EOS>"],
    pad_id=stoi["<PAD>"],
    blank_id=stoi.get("<BLANK>", None),
    use_ctc_head=False,
    use_attention_head=True,
)

# Загрузка весов
state_dict = torch.load(WEIGHTS_PTH, map_location=DEVICE)
model_pth.load_state_dict(state_dict, strict=False)
model_pth.to(DEVICE)
model_pth.eval()
print("✅ PTH model loaded")

# Предобработка
transform = get_val_transform(img_h=img_h, img_w=img_w)
img = imread_unicode(TEST_IMAGE)
if img is None:
    raise ValueError(f"Cannot read image: {TEST_IMAGE}")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
transformed = transform(image=img)
image_tensor = transformed["image"].unsqueeze(0).to(DEVICE)  # [1, 3, H, W]

# Инференс
with torch.no_grad():
    output_pth = model_pth(
        image_tensor,
        is_train=False,
        mode="attention",
        batch_max_length=max_len
    )
    preds_pth = output_pth["attention_preds"]  # [1, T]
    logits_pth = output_pth["attention_logits"]  # [1, T, num_classes]

preds_pth_np = preds_pth[0].cpu().numpy()
logits_pth_np = logits_pth[0].cpu().numpy()
probs_pth = torch.softmax(logits_pth, dim=-1)[0].cpu().numpy()

# Декодирование
text_pth = []
for token_id in preds_pth_np:
    if token_id == stoi["<EOS>"]:
        break
    if token_id not in [stoi["<PAD>"], stoi["<SOS>"]]:
        if token_id < len(itos):
            text_pth.append(itos[token_id])

text_pth_str = "".join(text_pth)

print(f"📊 Output shape: preds={preds_pth.shape}, logits={logits_pth.shape}")
print(f"📝 Decoded text: '{text_pth_str}'")
print(f"🔢 Token IDs (first 20): {preds_pth_np[:20].tolist()}")
print()

# ============================================
# 2. ONNX ИНФЕРЕНС
# ============================================

print("=" * 80)
print("2️⃣ ONNX MODEL INFERENCE")
print("=" * 80)

recognizer_onnx = TRBA(
    weights_path=WEIGHTS_ONNX,
    config_path=CONFIG_PATH,
    charset_path=CHARSET_PATH,
    device=DEVICE
)
print("✅ ONNX model loaded")
print(f"   ONNX config in recognizer:")
print(f"      img_h: {recognizer_onnx.img_h}")
print(f"      img_w: {recognizer_onnx.img_w}")
print(f"      max_length: {recognizer_onnx.max_length}")
print(f"      hidden_size: {recognizer_onnx.hidden_size}")
print(f"      num_classes: {len(recognizer_onnx.itos)}")
print(f"      cnn_backbone: {recognizer_onnx.cnn_backbone}")
print(f"      cnn_out_channels: {recognizer_onnx.cnn_out_channels}")
print(f"      num_encoder_layers: {recognizer_onnx.num_encoder_layers}")
print()

# Инференс через ONNX
result_onnx = recognizer_onnx.predict(TEST_IMAGE)[0]
text_onnx_str = result_onnx["text"]
confidence_onnx = result_onnx["confidence"]

print(f"📝 Decoded text: '{text_onnx_str}'")
print(f"📊 Confidence: {confidence_onnx:.4f}")
print()

# Получаем сырые выходы ONNX для детального сравнения
# Предобработка такая же как в TRBA
img_preprocessed = recognizer_onnx._preprocess_image(TEST_IMAGE)  # [1, 3, H, W]

import onnxruntime as ort
input_name = recognizer_onnx.onnx_session.get_inputs()[0].name
output_name = recognizer_onnx.onnx_session.get_outputs()[0].name

ort_outputs = recognizer_onnx.onnx_session.run(
    [output_name],
    {input_name: img_preprocessed}
)
logits_onnx_np = ort_outputs[0][0]  # [max_length, num_classes]
preds_onnx_np = np.argmax(logits_onnx_np, axis=-1)  # [max_length]
probs_onnx = np.exp(logits_onnx_np - np.max(logits_onnx_np, axis=-1, keepdims=True))
probs_onnx = probs_onnx / np.sum(probs_onnx, axis=-1, keepdims=True)

print(f"📊 Output shape: preds={preds_onnx_np.shape}, logits={logits_onnx_np.shape}")
print(f"🔢 Token IDs (first 20): {preds_onnx_np[:20].tolist()}")
print()

# ============================================
# 3. СРАВНЕНИЕ ВХОДОВ И ВЫХОДОВ
# ============================================

print("=" * 80)
print("🔍 INPUT/OUTPUT COMPARISON")
print("=" * 80)

# Сравнение входных тензоров
print(f"\n📥 INPUT TENSORS:")
print(f"   PTH input shape:  {image_tensor.shape}")
print(f"   ONNX input shape: {img_preprocessed.shape}")
print(f"   Shapes match: {image_tensor.shape == tuple(img_preprocessed.shape)}")

# Конвертируем PTH тензор в numpy для сравнения
img_tensor_pth_np = image_tensor.cpu().numpy()

# Статистика входных данных
print(f"\n   PTH input stats:")
print(f"      min={img_tensor_pth_np.min():.6f}, max={img_tensor_pth_np.max():.6f}")
print(f"      mean={img_tensor_pth_np.mean():.6f}, std={img_tensor_pth_np.std():.6f}")
print(f"   ONNX input stats:")
print(f"      min={img_preprocessed.min():.6f}, max={img_preprocessed.max():.6f}")
print(f"      mean={img_preprocessed.mean():.6f}, std={img_preprocessed.std():.6f}")

# Pixel-level comparison
input_diff = np.abs(img_tensor_pth_np - img_preprocessed)
print(f"\n   Input difference (abs):")
print(f"      min={input_diff.min():.10f}, max={input_diff.max():.10f}")
print(f"      mean={input_diff.mean():.10f}, std={input_diff.std():.10f}")
print(f"   Inputs identical: {np.allclose(img_tensor_pth_np, img_preprocessed, atol=1e-6)}")

# Сравнение выходных логитов
print(f"\n📤 OUTPUT LOGITS:")
print(f"   PTH output shape:  {logits_pth.shape}")
print(f"   ONNX output shape: {logits_onnx_np.shape}")

# Берем только общую длину для сравнения
min_len_logits = min(logits_pth.shape[1], logits_onnx_np.shape[0])
logits_pth_np = logits_pth[0, :min_len_logits, :].cpu().numpy()
logits_onnx_trimmed = logits_onnx_np[:min_len_logits, :]

print(f"\n   PTH logits stats (first {min_len_logits} steps):")
print(f"      min={logits_pth_np.min():.6f}, max={logits_pth_np.max():.6f}")
print(f"      mean={logits_pth_np.mean():.6f}, std={logits_pth_np.std():.6f}")
print(f"   ONNX logits stats (first {min_len_logits} steps):")
print(f"      min={logits_onnx_trimmed.min():.6f}, max={logits_onnx_trimmed.max():.6f}")
print(f"      mean={logits_onnx_trimmed.mean():.6f}, std={logits_onnx_trimmed.std():.6f}")

# Logits difference
logits_diff = np.abs(logits_pth_np - logits_onnx_trimmed)
print(f"\n   Logits difference (abs, first {min_len_logits} steps):")
print(f"      min={logits_diff.min():.10f}, max={logits_diff.max():.10f}")
print(f"      mean={logits_diff.mean():.10f}, std={logits_diff.std():.10f}")
print(f"   Logits close: {np.allclose(logits_pth_np, logits_onnx_trimmed, atol=1e-5)}")

# Показываем где максимальные различия
if logits_diff.max() > 1e-5:
    max_diff_pos = np.unravel_index(logits_diff.argmax(), logits_diff.shape)
    print(f"\n   ⚠️  Max difference at position {max_diff_pos}:")
    print(f"      PTH:  {logits_pth_np[max_diff_pos]:.10f}")
    print(f"      ONNX: {logits_onnx_trimmed[max_diff_pos]:.10f}")
    print(f"      Diff: {logits_diff[max_diff_pos]:.10f}")

# ============================================
# 4. СРАВНЕНИЕ ДЕКОДИРОВАННЫХ РЕЗУЛЬТАТОВ
# ============================================

print("\n" + "=" * 80)
print("🔍 DECODED TEXT COMPARISON")
print("=" * 80)

print(f"\n📝 Text comparison:")
print(f"   PTH:  '{text_pth_str}'")
print(f"   ONNX: '{text_onnx_str}'")
print(f"   Match: {text_pth_str == text_onnx_str}")

print(f"\n🔢 Token sequence comparison (first 20):")
print(f"   PTH:  {preds_pth_np[:20].tolist()}")
print(f"   ONNX: {preds_onnx_np[:20].tolist()}")

# Найдем первое различие
min_len = min(len(preds_pth_np), len(preds_onnx_np))
first_diff = None
for i in range(min_len):
    if preds_pth_np[i] != preds_onnx_np[i]:
        first_diff = i
        break

if first_diff is not None:
    print(f"\n⚠️  First difference at position {first_diff}:")
    print(f"   PTH:  token_id={preds_pth_np[first_diff]} → '{itos[preds_pth_np[first_diff]] if preds_pth_np[first_diff] < len(itos) else '?'}'")
    print(f"   ONNX: token_id={preds_onnx_np[first_diff]} → '{itos[preds_onnx_np[first_diff]] if preds_onnx_np[first_diff] < len(itos) else '?'}'")
elif len(preds_pth_np) != len(preds_onnx_np):
    print(f"\n⚠️  Sequences have different lengths:")
    print(f"   PTH:  {len(preds_pth_np)} tokens")
    print(f"   ONNX: {len(preds_onnx_np)} tokens")
else:
    print(f"\n✅ Token sequences are identical!")

print(f"\n📏 Length comparison:")
print(f"   PTH output length:  {preds_pth_np.shape[0]} tokens")
print(f"   ONNX output length: {preds_onnx_np.shape[0]} tokens")
print(f"   Expected max_len:   {max_len} tokens")

# Сравнение вероятностей на первых токенах
print(f"\n📊 Probability comparison (all tokens until EOS or mismatch):")

# Определяем EOS ID
eos_token = '[EOS]'
eos_id = stoi.get(eos_token, -1)
print(f"   EOS token: '{eos_token}' (id={eos_id})")
print()

for i in range(min(20, min_len)):
    token_pth = preds_pth_np[i]
    token_onnx = preds_onnx_np[i]
    prob_pth = probs_pth[i, token_pth]
    prob_onnx = probs_onnx[i, token_onnx]
    
    char_pth = itos[token_pth] if token_pth < len(itos) else '?'
    char_onnx = itos[token_onnx] if token_onnx < len(itos) else '?'
    
    match = "✓" if token_pth == token_onnx else "✗"
    
    # Показываем топ-3 вероятности для каждой позиции при несовпадении
    if token_pth != token_onnx:
        print(f"   [{i}] {match} PTH: '{char_pth}' (id={token_pth}, p={prob_pth:.6f}) | ONNX: '{char_onnx}' (id={token_onnx}, p={prob_onnx:.6f})")
        
        # Топ-3 для PTH
        top3_pth = np.argsort(probs_pth[i])[-3:][::-1]
        print(f"       PTH top-3: ", end="")
        for tid in top3_pth:
            c = itos[tid] if tid < len(itos) else '?'
            print(f"'{c}'({tid})={probs_pth[i,tid]:.6f} ", end="")
        print()
        
        # Топ-3 для ONNX
        top3_onnx = np.argsort(probs_onnx[i])[-3:][::-1]
        print(f"       ONNX top-3: ", end="")
        for tid in top3_onnx:
            c = itos[tid] if tid < len(itos) else '?'
            print(f"'{c}'({tid})={probs_onnx[i,tid]:.6f} ", end="")
        print()
    else:
        print(f"   [{i}] {match} PTH: '{char_pth}' (id={token_pth}, p={prob_pth:.6f}) | ONNX: '{char_onnx}' (id={token_onnx}, p={prob_onnx:.6f})")
    
    # Останавливаемся на EOS или при несовпадении
    if token_pth == eos_id or token_onnx == eos_id:
        print(f"   ... (reached EOS)")
        break

# ============================================
# 4. ДЕТАЛЬНЫЙ АНАЛИЗ РАЗЛИЧИЙ
# ============================================

print("\n" + "=" * 80)
print("🔬 DETAILED ANALYSIS")
print("=" * 80)

# Проверим настройки декодирования
print(f"\n⚙️ Decoding settings:")
print(f"   Max length config: {max_len}")
print(f"   PTH uses: batch_max_length + 1 = {max_len + 1} steps (with early stopping)")
print(f"   ONNX uses: batch_max_length = {max_len} steps (no early stopping)")
print(f"   → This is a KEY DIFFERENCE!")

# Найдем позицию EOS токена
eos_id = stoi["<EOS>"]
eos_pos_pth = None
eos_pos_onnx = None

for i, token in enumerate(preds_pth_np):
    if token == eos_id:
        eos_pos_pth = i
        break

for i, token in enumerate(preds_onnx_np):
    if token == eos_id:
        eos_pos_onnx = i
        break

print(f"\n🛑 EOS token positions:")
print(f"   PTH:  EOS at position {eos_pos_pth if eos_pos_pth is not None else 'NOT FOUND'}")
print(f"   ONNX: EOS at position {eos_pos_onnx if eos_pos_onnx is not None else 'NOT FOUND'}")

if eos_pos_pth is not None and eos_pos_onnx is not None:
    if eos_pos_pth != eos_pos_onnx:
        print(f"   ⚠️  EOS positions differ by {abs(eos_pos_pth - eos_pos_onnx)} steps")

print("\n" + "=" * 80)
print("💡 CONCLUSIONS")
print("=" * 80)

if text_pth_str == text_onnx_str:
    print("✅ Texts are IDENTICAL - no issues detected")
else:
    print("❌ Texts are DIFFERENT - investigating reasons:")
    print(f"\n   1. Different decoding lengths:")
    print(f"      PTH:  {max_len + 1} steps (max_length + 1)")
    print(f"      ONNX: {max_len} steps (max_length)")
    print(f"      → ONNX may miss last character or cut early")
    
    print(f"\n   2. Early stopping:")
    print(f"      PTH:  Uses early stopping on EOS")
    print(f"      ONNX: No early stopping (always runs full length)")
    print(f"      → May affect autoregressive behavior")
    
    print(f"\n   3. Possible causes:")
    print(f"      - ONNX export with onnx_mode=True reduces steps")
    print(f"      - This affects how many characters can be predicted")
    print(f"      - Autoregressive nature means each step affects next")

print("\n✨ DONE!")
