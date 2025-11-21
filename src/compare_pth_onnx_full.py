"""
Полное сравнение PTH и ONNX на одинаковом наборе изображений.
Цель: убедиться что метрики идентичны.
"""

import os
import csv
import json
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from jiwer import cer, wer

from manuscript.recognizers import TRBA
from manuscript.recognizers._trba.model.model import TRBAModel
from manuscript.recognizers._trba.data.transforms import load_charset, get_val_transform

# ============================================
# НАСТРОЙКИ
# ============================================

WEIGHTS_PTH = r"C:\Users\USER\Desktop\trba_exp_lite\best_acc_weights.pth"
WEIGHTS_ONNX = r"C:\Users\USER\Desktop\trba_exp_lite\trba_exp_lite.onnx"
CONFIG_PATH = r"C:\Users\USER\Desktop\trba_exp_lite\config.json"
CHARSET_PATH = r"C:\Users\USER\Desktop\trba_exp_lite\charset.txt"

# Тестовый датасет (выберите ОДИН маленький датасет для теста)
DATASETS = [
    {
        "image_dir": r"C:\shared\orig_cyrillic\test",
        "gt_path": r"C:\shared\orig_cyrillic\test.csv",
    },
]

BATCH_SIZE = 32
MAX_IMAGES = 100  # Для быстрого теста
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 80)
print("🔬 ПОЛНОЕ СРАВНЕНИЕ PTH vs ONNX")
print("=" * 80)
print(f"PTH weights:  {WEIGHTS_PTH}")
print(f"ONNX weights: {WEIGHTS_ONNX}")
print(f"Device:       {DEVICE}")
print(f"Max images:   {MAX_IMAGES}")
print("=" * 80)
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

itos, stoi = load_charset(CHARSET_PATH)

print(f"📄 Config: img_size={img_h}x{img_w}, max_len={max_len}, hidden={hidden_size}")
print(f"📚 Charset: {len(itos)} classes")
print()

# ============================================
# ЗАГРУЗКА GT
# ============================================

print("📂 Loading ground truth...")
gt_data = {}
for dataset in DATASETS:
    with open(dataset["gt_path"], "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                fname = row[0].strip()
                text = ",".join(row[1:]).strip()
                gt_data[fname] = text

print(f"   Loaded {len(gt_data)} GT entries")
print()

# ============================================
# СКАНИРОВАНИЕ ИЗОБРАЖЕНИЙ
# ============================================

print("📁 Scanning images...")
valid_ext = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
images = []

for dataset in DATASETS:
    image_dir = dataset["image_dir"]
    dataset_images = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in valid_ext
    ]
    images.extend(dataset_images)

images = images[:MAX_IMAGES]
print(f"   Total images: {len(images)}")
print()

# ============================================
# ФУНКЦИЯ ЧТЕНИЯ С UNICODE
# ============================================

def imread_unicode(path):
    """Читает изображение с Unicode путём (кириллица)"""
    with open(path, 'rb') as f:
        arr = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

# ============================================
# PTH INFERENCE
# ============================================

print("=" * 80)
print("1️⃣ PTH MODEL INFERENCE")
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
print("✅ PTH model loaded")

transform = get_val_transform(img_h=img_h, img_w=img_w)

def preprocess_image_pth(image_path):
    img = imread_unicode(image_path)
    if img is None:
        raise ValueError(f"Failed to load: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transformed = transform(image=img)
    return transformed["image"]

# Инференс
results_pth = []
print("   Running PTH inference...")
for i in tqdm(range(0, len(images), BATCH_SIZE)):
    batch_paths = images[i : i + BATCH_SIZE]
    batch_tensors = []
    
    for img_path in batch_paths:
        try:
            tensor = preprocess_image_pth(img_path)
            batch_tensors.append(tensor)
        except Exception as e:
            print(f"   ⚠️  Failed to load {img_path}: {e}")
            results_pth.append({"path": img_path, "text": "", "confidence": 0.0})
            continue
    
    if not batch_tensors:
        continue
    
    batch = torch.stack(batch_tensors).to(DEVICE)
    
    with torch.no_grad():
        output = model_pth(batch, is_train=False, mode="attention", batch_max_length=max_len)
        preds = output["attention_preds"].cpu().numpy()
        logits = output["attention_logits"]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    
    for j in range(len(batch_tensors)):
        pred_row = preds[j]
        
        decoded_chars = []
        for token_id in pred_row:
            if token_id == stoi["<EOS>"]:
                break
            if token_id not in [stoi["<PAD>"], stoi["<SOS>"]]:
                if token_id < len(itos):
                    decoded_chars.append(itos[token_id])
        
        text = "".join(decoded_chars)
        
        seq_probs = []
        for t, token_id in enumerate(pred_row):
            if token_id == stoi["<EOS>"]:
                break
            if token_id not in [stoi["<PAD>"], stoi["<SOS>"]]:
                seq_probs.append(probs[j, t, token_id])
        
        confidence = float(np.mean(seq_probs)) if seq_probs else 0.0
        
        results_pth.append({
            "path": batch_paths[j],
            "text": text,
            "confidence": confidence
        })

print(f"✅ PTH inference completed: {len(results_pth)} results")
print()

# ============================================
# ONNX INFERENCE
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

# Инференс
results_onnx = []
print("   Running ONNX inference...")
for i in tqdm(range(0, len(images), BATCH_SIZE)):
    batch_paths = images[i : i + BATCH_SIZE]
    
    try:
        batch_results = recognizer_onnx.predict(batch_paths, batch_size=BATCH_SIZE)
        for j, result in enumerate(batch_results):
            results_onnx.append({
                "path": batch_paths[j],
                "text": result["text"],
                "confidence": result["confidence"]
            })
    except Exception as e:
        print(f"   ⚠️  Batch {i} failed: {e}")
        for path in batch_paths:
            results_onnx.append({"path": path, "text": "", "confidence": 0.0})

print(f"✅ ONNX inference completed: {len(results_onnx)} results")
print()

# ============================================
# СРАВНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================

print("=" * 80)
print("🔍 COMPARISON")
print("=" * 80)

# Подготовка данных для метрик
refs = []
hyps_pth = []
hyps_onnx = []
mismatches = []

for i in range(len(results_pth)):
    img_path = results_pth[i]["path"]
    fname = os.path.basename(img_path)
    
    if fname not in gt_data:
        continue
    
    ref_text = gt_data[fname]
    pth_text = results_pth[i]["text"]
    onnx_text = results_onnx[i]["text"]
    
    refs.append(ref_text)
    hyps_pth.append(pth_text)
    hyps_onnx.append(onnx_text)
    
    if pth_text != onnx_text:
        mismatches.append({
            "fname": fname,
            "ref": ref_text,
            "pth": pth_text,
            "onnx": onnx_text
        })

print(f"\n📊 Processed {len(refs)} images with GT")
print()

# Метрики PTH
if refs and hyps_pth:
    cer_pth = cer(refs, hyps_pth) * 100
    wer_pth = wer(refs, hyps_pth) * 100
    acc_pth = sum(r == h for r, h in zip(refs, hyps_pth)) / len(refs) * 100
    print(f"📈 PTH Metrics:")
    print(f"   CER: {cer_pth:.2f}%")
    print(f"   WER: {wer_pth:.2f}%")
    print(f"   Accuracy: {acc_pth:.2f}%")
    print()

# Метрики ONNX
if refs and hyps_onnx:
    cer_onnx = cer(refs, hyps_onnx) * 100
    wer_onnx = wer(refs, hyps_onnx) * 100
    acc_onnx = sum(r == h for r, h in zip(refs, hyps_onnx)) / len(refs) * 100
    print(f"📈 ONNX Metrics:")
    print(f"   CER: {cer_onnx:.2f}%")
    print(f"   WER: {wer_onnx:.2f}%")
    print(f"   Accuracy: {acc_onnx:.2f}%")
    print()

# Разница
print(f"📊 Difference:")
print(f"   ΔCER: {abs(cer_pth - cer_onnx):.4f}%")
print(f"   ΔWER: {abs(wer_pth - wer_onnx):.4f}%")
print(f"   ΔAccuracy: {abs(acc_pth - acc_onnx):.4f}%")
print()

# Несовпадения
print(f"⚠️  Mismatches: {len(mismatches)} / {len(refs)} ({len(mismatches)/len(refs)*100:.2f}%)")
if mismatches:
    print(f"\n   First 10 mismatches:")
    for i, mm in enumerate(mismatches[:10]):
        print(f"   [{i+1}] {mm['fname']}")
        print(f"       GT:   '{mm['ref']}'")
        print(f"       PTH:  '{mm['pth']}'")
        print(f"       ONNX: '{mm['onnx']}'")
        print()

print("=" * 80)
print("✅ COMPARISON COMPLETE")
print("=" * 80)
