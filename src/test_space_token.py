"""
Проверка декодирования пробела в PTH и ONNX.
"""

import os
import json
import torch
import cv2
import numpy as np

from manuscript.recognizers import TRBA
from manuscript.recognizers._trba.model.model import TRBAModel
from manuscript.recognizers._trba.data.transforms import load_charset, get_val_transform

WEIGHTS_PTH = r"C:\Users\USER\Desktop\trba_exp_lite\best_acc_weights.pth"
WEIGHTS_ONNX = r"C:\Users\USER\Desktop\trba_exp_lite\trba_exp_lite.onnx"
CONFIG_PATH = r"C:\Users\USER\Desktop\trba_exp_lite\config.json"
CHARSET_PATH = r"C:\Users\USER\Desktop\trba_exp_lite\charset.txt"

# Тестовое изображение с пробелом
TEST_IMAGE = r"C:\shared\orig_cyrillic\test\test1000.png"  # '1 класса'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 80)
print("🔍 ПРОВЕРКА ДЕКОДИРОВАНИЯ ПРОБЕЛА")
print("=" * 80)
print(f"Test image: {TEST_IMAGE}")
print(f"Expected:   '1 класса' (с пробелом)")
print("=" * 80)
print()

# ============================================
# ЗАГРУЗКА CHARSET
# ============================================

itos, stoi = load_charset(CHARSET_PATH)

print(f"📚 Charset: {len(itos)} classes")
print(f"   Special tokens:")
print(f"      PAD:   id={stoi.get('<PAD>', -1)}, char='{itos[stoi.get('<PAD>', 0)] if '<PAD>' in stoi else 'N/A'}'")
print(f"      SOS:   id={stoi.get('<SOS>', -1)}, char='{itos[stoi.get('<SOS>', 0)] if '<SOS>' in stoi else 'N/A'}'")
print(f"      EOS:   id={stoi.get('<EOS>', -1)}, char='{itos[stoi.get('<EOS>', 0)] if '<EOS>' in stoi else 'N/A'}'")
print(f"      BLANK: id={stoi.get('<BLANK>', -1)}, char='{itos[stoi.get('<BLANK>', 0)] if '<BLANK>' in stoi else 'N/A'}'")
print(f"      SPACE: id={stoi.get(' ', -1)}, char=' '")
print()

space_id = stoi.get(' ', -1)
if space_id == -1:
    print("⚠️  WARNING: Space character not found in charset!")
    print()

# ============================================
# ЗАГРУЗКА КОНФИГУРАЦИИ
# ============================================

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

img_h = config.get("img_h", 64)
img_w = config.get("img_w", 256)
max_len = config.get("max_len", 40)
hidden_size = config.get("hidden_size", 256)
num_encoder_layers = config.get("num_encoder_layers", 2)
cnn_in_channels = config.get("cnn_in_channels", 3)
cnn_out_channels = config.get("cnn_out_channels", 512)
cnn_backbone = config.get("cnn_backbone", "seresnet31")

# ============================================
# ФУНКЦИЯ ЧТЕНИЯ
# ============================================

def imread_unicode(path):
    with open(path, 'rb') as f:
        arr = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

# ============================================
# PTH INFERENCE
# ============================================

print("=" * 80)
print("1️⃣ PTH MODEL")
print("=" * 80)

model_pth = TRBAModel(
    num_classes=len(itos),
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

state_dict = torch.load(WEIGHTS_PTH, map_location=DEVICE)
model_pth.load_state_dict(state_dict, strict=False)
model_pth.to(DEVICE)
model_pth.eval()

transform = get_val_transform(img_h=img_h, img_w=img_w)

img = imread_unicode(TEST_IMAGE)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
transformed = transform(image=img)
image_tensor = transformed["image"].unsqueeze(0).to(DEVICE)

with torch.no_grad():
    output_pth = model_pth(
        image_tensor,
        is_train=False,
        mode="attention",
        batch_max_length=max_len
    )
    preds_pth = output_pth["attention_preds"][0].cpu().numpy()
    logits_pth = output_pth["attention_logits"][0].cpu().numpy()
    probs_pth = torch.softmax(output_pth["attention_logits"], dim=-1)[0].cpu().numpy()

# Декодирование
decoded_chars = []
token_details = []

for i, token_id in enumerate(preds_pth):
    if token_id == stoi["<EOS>"]:
        token_details.append(f"[{i}] <EOS> (id={token_id})")
        break
    
    if token_id == stoi["<PAD>"]:
        token_details.append(f"[{i}] <PAD> (id={token_id})")
        continue
    
    if token_id == stoi["<SOS>"]:
        token_details.append(f"[{i}] <SOS> (id={token_id})")
        continue
    
    char = itos[token_id] if token_id < len(itos) else '?'
    decoded_chars.append(char)
    
    if char == ' ':
        token_details.append(f"[{i}] SPACE (id={token_id})")
    else:
        token_details.append(f"[{i}] '{char}' (id={token_id})")

text_pth = "".join(decoded_chars)

print(f"📝 Decoded text: '{text_pth}'")
print(f"🔢 Token sequence:")
for detail in token_details[:20]:
    print(f"   {detail}")
print()

# ============================================
# ONNX INFERENCE
# ============================================

print("=" * 80)
print("2️⃣ ONNX MODEL")
print("=" * 80)

recognizer_onnx = TRBA(
    weights_path=WEIGHTS_ONNX,
    config_path=CONFIG_PATH,
    charset_path=CHARSET_PATH,
    device=DEVICE
)

result_onnx = recognizer_onnx.predict(TEST_IMAGE)[0]
text_onnx = result_onnx["text"]

# Получаем сырые токены
img_preprocessed = recognizer_onnx._preprocess_image(TEST_IMAGE)

import onnxruntime as ort
input_name = recognizer_onnx.onnx_session.get_inputs()[0].name
output_name = recognizer_onnx.onnx_session.get_outputs()[0].name

ort_outputs = recognizer_onnx.onnx_session.run(
    [output_name],
    {input_name: img_preprocessed}
)
logits_onnx = ort_outputs[0][0]
preds_onnx = np.argmax(logits_onnx, axis=-1)

# Вычисляем вероятности для ONNX
probs_onnx = np.exp(logits_onnx - np.max(logits_onnx, axis=-1, keepdims=True))
probs_onnx = probs_onnx / np.sum(probs_onnx, axis=-1, keepdims=True)

# Декодирование
decoded_chars_onnx = []
token_details_onnx = []

for i, token_id in enumerate(preds_onnx):
    if token_id == recognizer_onnx.eos_id:
        token_details_onnx.append(f"[{i}] <EOS> (id={token_id})")
        break
    
    if token_id == recognizer_onnx.pad_id:
        token_details_onnx.append(f"[{i}] <PAD> (id={token_id})")
        continue
    
    if token_id == recognizer_onnx.sos_id:
        token_details_onnx.append(f"[{i}] <SOS> (id={token_id})")
        continue
    
    char = recognizer_onnx.itos[token_id] if token_id < len(recognizer_onnx.itos) else '?'
    decoded_chars_onnx.append(char)
    
    if char == ' ':
        token_details_onnx.append(f"[{i}] SPACE (id={token_id})")
    else:
        token_details_onnx.append(f"[{i}] '{char}' (id={token_id})")

print(f"📝 Decoded text: '{text_onnx}'")
print(f"🔢 Token sequence:")
for detail in token_details_onnx[:20]:
    print(f"   {detail}")
print()

# ============================================
# СРАВНЕНИЕ
# ============================================

print("=" * 80)
print("🔍 COMPARISON")
print("=" * 80)

print(f"\n📝 Texts:")
print(f"   Expected: '1 класса'")
print(f"   PTH:      '{text_pth}'")
print(f"   ONNX:     '{text_onnx}'")
print()

print(f"🔢 Token comparison:")
min_len = min(len(token_details), len(token_details_onnx))
for i in range(min_len):
    match = "✓" if preds_pth[i] == preds_onnx[i] else "✗"
    print(f"   {match} {token_details[i]:30s} | {token_details_onnx[i]}")

if len(token_details) != len(token_details_onnx):
    print(f"\n⚠️  Length mismatch:")
    print(f"   PTH:  {len(token_details)} tokens")
    print(f"   ONNX: {len(token_details_onnx)} tokens")

# ============================================
# АНАЛИЗ ЛОГИТОВ НА ПОЗИЦИИ [1]
# ============================================

print()
print("=" * 80)
print("🔬 LOGITS ANALYSIS AT POSITION [1]")
print("=" * 80)

pos = 1
space_id_val = stoi.get(' ', 3)
one_id = 57

print(f"\nЛогиты и вероятности на позиции [{pos}]:")
print(f"   Token ID для пробела ' ': {space_id_val}")
print(f"   Token ID для '1': {one_id}")
print()

# PTH
print(f"PTH:")
print(f"   Logit[{space_id_val}] (space): {logits_pth[pos, space_id_val]:.6f}")
print(f"   Logit[{one_id}] ('1'):    {logits_pth[pos, one_id]:.6f}")
print(f"   Prob[{space_id_val}] (space):  {probs_pth[pos, space_id_val]:.6f}")
print(f"   Prob[{one_id}] ('1'):     {probs_pth[pos, one_id]:.6f}")
print(f"   Predicted: id={preds_pth[pos]}, char='{itos[preds_pth[pos]]}'")
print()

# ONNX
print(f"ONNX:")
print(f"   Logit[{space_id_val}] (space): {logits_onnx[pos, space_id_val]:.6f}")
print(f"   Logit[{one_id}] ('1'):    {logits_onnx[pos, one_id]:.6f}")
print(f"   Prob[{space_id_val}] (space):  {probs_onnx[pos, space_id_val]:.6f}")
print(f"   Prob[{one_id}] ('1'):     {probs_onnx[pos, one_id]:.6f}")
print(f"   Predicted: id={preds_onnx[pos]}, char='{recognizer_onnx.itos[preds_onnx[pos]]}'")
print()

# Разница
print(f"Difference:")
print(f"   ΔLogit[{space_id_val}] (space): {abs(logits_pth[pos, space_id_val] - logits_onnx[pos, space_id_val]):.6f}")
print(f"   ΔLogit[{one_id}] ('1'):    {abs(logits_pth[pos, one_id] - logits_onnx[pos, one_id]):.6f}")
print()

# Топ-5 для PTH и ONNX
print(f"Top-5 predictions:")
print(f"   PTH:")
top5_pth = np.argsort(probs_pth[pos])[-5:][::-1]
for tid in top5_pth:
    char = itos[tid] if tid < len(itos) else '?'
    if char == ' ':
        char = 'SPACE'
    print(f"      [{tid:3d}] '{char}': logit={logits_pth[pos, tid]:8.4f}, prob={probs_pth[pos, tid]:.6f}")

print(f"   ONNX:")
top5_onnx = np.argsort(probs_onnx[pos])[-5:][::-1]
for tid in top5_onnx:
    char = recognizer_onnx.itos[tid] if tid < len(recognizer_onnx.itos) else '?'
    if char == ' ':
        char = 'SPACE'
    print(f"      [{tid:3d}] '{char}': logit={logits_onnx[pos, tid]:8.4f}, prob={probs_onnx[pos, tid]:.6f}")

print()
print("=" * 80)
