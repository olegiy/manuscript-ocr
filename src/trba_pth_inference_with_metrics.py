"""
Инференс TRBA модели с использованием PTH весов и детальными метриками.
Аналогично src/trba_metrics.py, но с прямой загрузкой PyTorch модели.
"""

import os
import time
import csv
import json
from collections import Counter
from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image
import Levenshtein
from tqdm import tqdm

from manuscript.recognizers._trba.model.model import TRBAModel
from manuscript.recognizers._trba.data.transforms import load_charset, get_val_transform
from manuscript.recognizers._trba.training.metrics import (
    character_error_rate,
    word_error_rate,
    compute_accuracy,
)

# ============================================
# КОНФИГУРАЦИЯ
# ============================================

# Пути к модели и данным
WEIGHTS_PATH = r"C:\Users\USER\Desktop\trba_exp_lite\best_acc_weights.pth"
CONFIG_PATH = r"C:\Users\USER\Desktop\trba_exp_lite\config.json"
CHARSET_PATH = r"C:\Users\USER\Desktop\trba_exp_lite\charset.txt"

# Датасеты для валидации
datasets = [
    {
        "image_dir": r"C:\shared\orig_cyrillic\test",
        "gt_path": r"C:\shared\orig_cyrillic\test.csv",
    },
    {
        "image_dir": r"C:\shared\school_notebooks_RU\school_notebooks_RU\val",
        "gt_path": r"C:\shared\school_notebooks_RU\school_notebooks_RU\val_converted.csv",
    },
    {
        "image_dir": r"C:\Users\USER\Desktop\archive_25_09\dataset\printed\val\img",
        "gt_path": r"C:\Users\USER\Desktop\archive_25_09\dataset\printed\val\labels.csv",
    },
    {
        "image_dir": r"C:\Users\USER\Desktop\archive_25_09\dataset\handwritten\val\img",
        "gt_path": r"C:\Users\USER\Desktop\archive_25_09\dataset\handwritten\val\labels.csv",
    },
]

# Параметры инференса
BATCH_SIZE = 64
MAX_IMAGES = 10000000000000000  # Ограничение на количество изображений
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Директория для сохранения отчетов
OUTPUT_DIR = Path(WEIGHTS_PATH).parent

print("=" * 80)
print("🚀 TRBA INFERENCE WITH PTH WEIGHTS + METRICS")
print("=" * 80)
print(f"Weights: {WEIGHTS_PATH}")
print(f"Config:  {CONFIG_PATH}")
print(f"Charset: {CHARSET_PATH}")
print(f"Device:  {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print("=" * 80)
print()

# ============================================
# ЗАГРУЗКА КОНФИГУРАЦИИ
# ============================================

print("📄 Loading configuration...")
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

print(f"   Image size: {img_h}×{img_w}")
print(f"   Max length: {max_len}")
print(f"   Hidden size: {hidden_size}")
print(f"   Encoder layers: {num_encoder_layers}")
print(f"   CNN backbone: {cnn_backbone}")
print(f"   CNN out channels: {cnn_out_channels}")
print()

# ============================================
# ЗАГРУЗКА CHARSET
# ============================================

print("📚 Loading charset...")
itos, stoi = load_charset(CHARSET_PATH)
num_classes = len(itos)
print(f"   Total classes: {num_classes}")
print(f"   Special tokens: PAD={stoi['<PAD>']}, SOS={stoi['<SOS>']}, EOS={stoi['<EOS>']}")
print()

# ============================================
# СОЗДАНИЕ И ЗАГРУЗКА МОДЕЛИ
# ============================================

print("🏗️  Building model...")
model = TRBAModel(
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
    use_ctc_head=False,  # Только attention для инференса
    use_attention_head=True,
)

print(f"   Loading weights from {WEIGHTS_PATH}...")
state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
model.load_state_dict(state_dict, strict=False)
model.to(DEVICE)
model.eval()

# Подсчет параметров
total_params = sum(p.numel() for p in model.parameters())
print(f"   Total parameters: {total_params:,} ({total_params*4/(1024*1024):.2f} MB)")
print()

# ============================================
# ЗАГРУЗКА GROUND TRUTH
# ============================================

print("📂 Loading ground truth data...")
gt_data = {}
total_gt_lines = 0

for idx, dataset in enumerate(datasets, 1):
    image_dir = dataset["image_dir"]
    gt_path = dataset["gt_path"]
    
    print(f"   Dataset {idx}: {os.path.basename(image_dir)}")
    
    dataset_gt = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                fname = row[0].strip()
                text = ",".join(row[1:]).strip()
                dataset_gt[fname] = text
    
    print(f"      Loaded {len(dataset_gt)} entries from {os.path.basename(gt_path)}")
    total_gt_lines += len(dataset_gt)
    
    for fname, text in dataset_gt.items():
        if fname in gt_data:
            print(f"      ⚠️  Duplicate file: {fname} (using last version)")
        gt_data[fname] = text

print(f"\n   Total GT entries: {total_gt_lines}")
print(f"   Unique files: {len(gt_data)}")
print()

# ============================================
# СКАНИРОВАНИЕ ИЗОБРАЖЕНИЙ
# ============================================

print("📁 Scanning images...")
valid_ext = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
images = []

for idx, dataset in enumerate(datasets, 1):
    image_dir = dataset["image_dir"]
    
    dataset_images = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in valid_ext
    ]
    
    if not dataset_images:
        print(f"   ⚠️  Dataset {idx}: No images found in {image_dir}!")
    else:
        print(f"   Dataset {idx}: Found {len(dataset_images)} images")
        images.extend(dataset_images)

if len(images) > MAX_IMAGES:
    print(f"   ⚠️  Taking only first {MAX_IMAGES} images from {len(images)}")
    images = images[:MAX_IMAGES]

if not images:
    raise RuntimeError(f"❌ No images found in any dataset!")

print(f"\n   TOTAL: {len(images)} images for recognition")
print()

# ============================================
# ПОДГОТОВКА ТРАНСФОРМАЦИЙ
# ============================================

transform = get_val_transform(img_h=img_h, img_w=img_w)

def imread_unicode(path):
    """Читает изображение с Unicode путём (поддержка кириллицы)"""
    with open(path, 'rb') as f:
        arr = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def preprocess_image(image_path):
    """Загрузка и предобработка изображения"""
    img = imread_unicode(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transformed = transform(image=img)
    return transformed["image"]  # [3, H, W]

# ============================================
# INFERENCE
# ============================================

print("🔮 Running inference...")
print(f"   Processing {len(images)} images in batches of {BATCH_SIZE}...")
print()

results = []
start_time = time.perf_counter()

with torch.no_grad():
    for i in tqdm(range(0, len(images), BATCH_SIZE), desc="Processing"):
        batch_images = images[i : i + BATCH_SIZE]
        
        # Загрузка и предобработка батча
        batch_tensors = []
        for img_path in batch_images:
            try:
                tensor = preprocess_image(img_path)
                batch_tensors.append(tensor)
            except Exception as e:
                print(f"\n⚠️  Error loading {img_path}: {e}")
                # Добавляем пустой результат
                results.append({"text": "", "confidence": 0.0})
                continue
        
        if not batch_tensors:
            continue
        
        # Создаем батч [B, 3, H, W]
        batch = torch.stack(batch_tensors).to(DEVICE)
        
        # Инференс
        output = model(batch, is_train=False, mode="attention", batch_max_length=max_len)
        preds = output["attention_preds"]  # [B, T]
        logits = output["attention_logits"]  # [B, T, num_classes]
        
        # Декодирование
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = preds.cpu().numpy()
        
        for j in range(len(batch_tensors)):
            pred_row = preds[j]  # [max_length]
            
            # Декодирование текста
            decoded_chars = []
            for token_id in pred_row:
                if token_id == stoi["<EOS>"]:
                    break
                if token_id not in [stoi["<PAD>"], stoi["<SOS>"]]:
                    if token_id < len(itos):
                        decoded_chars.append(itos[token_id])
            
            text = "".join(decoded_chars)
            
            # Расчет уверенности
            seq_probs = []
            for t, token_id in enumerate(pred_row):
                if token_id == stoi["<EOS>"]:
                    break
                if token_id not in [stoi["<PAD>"], stoi["<SOS>"]]:
                    seq_probs.append(probs[j, t, token_id])
            
            confidence = float(np.mean(seq_probs)) if seq_probs else 0.0
            
            results.append({"text": text, "confidence": confidence})

end_time = time.perf_counter()
total_time = end_time - start_time
avg_time = total_time / len(images)
fps = 1.0 / avg_time if avg_time > 0 else float("inf")

print(f"\n✅ Inference completed!")
print(f"   Total time: {total_time:.3f}s")
print(f"   Average per image: {avg_time:.3f}s ({fps:.1f} FPS)")
print()

# ============================================
# ФУНКЦИЯ ДЛЯ ФИЛЬТРАЦИИ ТОЛЬКО БУКВ И ЦИФР
# ============================================

def filter_chars_only(text):
    """
    Оставляет только буквы (включая кириллицу и дореформенные) и цифры.
    Убирает пробелы, пунктуацию и спецсимволы.
    """
    # Определяем допустимые символы: буквы латиницы, кириллицы (включая дореформенные) и цифры
    allowed_chars = set(
        'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
        'ѣѢіІѳѲѵѴѫѪѭѬѯѮѱѰѡѠѕѕѧѦѩѨ'  # Дореформенные символы
        '0123456789'
    )
    return ''.join(c for c in text if c in allowed_chars)

# ============================================
# СОПОСТАВЛЕНИЕ С GROUND TRUTH И РАСЧЕТ МЕТРИК
# ============================================

print("=" * 80)
print("📊 CALCULATING METRICS")
print("=" * 80)
print()

refs, hyps = [], []
total_cer, total_wer = 0.0, 0.0
cer_count, wer_count = 0, 0
error_details = []

# Создаем уникальные имена для датасетов и нормализуем пути
dataset_mapping = {}  # путь к изображению -> уникальное имя датасета
dataset_results = {}  # уникальное имя -> метрики

for idx, dataset in enumerate(datasets, 1):
    # Создаем уникальное имя: если базовое имя не уникально, добавляем часть родительского пути
    base_name = os.path.basename(dataset["image_dir"])
    
    # Проверяем уникальность базового имени
    existing_names = [name for name in dataset_results.keys()]
    if base_name in existing_names:
        # Добавляем часть пути для уникальности
        parent = os.path.basename(os.path.dirname(dataset["image_dir"]))
        dataset_name = f"{parent}_{base_name}"
    else:
        dataset_name = base_name
    
    # Нормализуем путь к директории для точного сопоставления
    normalized_dir = os.path.normpath(dataset["image_dir"])
    dataset_mapping[normalized_dir] = dataset_name
    
    dataset_results[dataset_name] = {
        'refs': [],
        'hyps': [],
        'total_cer': 0.0,
        'total_wer': 0.0,
        'count': 0
    }

print("Results:")
print("-" * 80)
for path, result in zip(images, results):
    pred_text = result["text"]
    score = result["confidence"]
    fname = os.path.basename(path)
    ref_text = gt_data.get(fname)
    
    if ref_text is None:
        print(f"{fname:40s} → {pred_text:20s} (no GT)")
        continue

    refs.append(ref_text)
    hyps.append(pred_text)

    cer = character_error_rate(ref_text, pred_text)
    wer = word_error_rate(ref_text, pred_text)

    total_cer += cer
    total_wer += wer
    cer_count += 1
    wer_count += 1
    
    # Определяем к какому датасету относится изображение
    # Нормализуем путь изображения и ищем точное совпадение директории
    normalized_path = os.path.normpath(path)
    path_dir = os.path.dirname(normalized_path)
    
    dataset_name = dataset_mapping.get(path_dir)
    if dataset_name:
        dataset_results[dataset_name]['refs'].append(ref_text)
        dataset_results[dataset_name]['hyps'].append(pred_text)
        dataset_results[dataset_name]['total_cer'] += cer
        dataset_results[dataset_name]['total_wer'] += wer
        dataset_results[dataset_name]['count'] += 1
    
    # Сохраняем детали ошибок
    if ref_text != pred_text:
        error_details.append({
            'fname': fname,
            'ref': ref_text,
            'hyp': pred_text,
            'cer': cer,
            'wer': wer,
            'confidence': score
        })

    print(f"{fname:40s} → {pred_text:20s} | GT: {ref_text:20s} | CER={cer:.3f} | WER={wer:.3f}")

print("-" * 80)
print()

# ============================================
# ОСНОВНЫЕ МЕТРИКИ
# ============================================

acc = compute_accuracy(refs, hyps)
acc_case_insensitive = sum(1 for r, h in zip(refs, hyps) if r.lower() == h.lower()) / max(len(refs), 1)

# Точность только по буквам и цифрам (chars only)
refs_chars_only = [filter_chars_only(r) for r in refs]
hyps_chars_only = [filter_chars_only(h) for h in hyps]
acc_chars_only = sum(1 for r, h in zip(refs_chars_only, hyps_chars_only) if r.lower() == h.lower()) / max(len(refs), 1)

case_only_errors = sum(1 for r, h in zip(refs, hyps) if r.lower() == h.lower() and r != h)
avg_cer = total_cer / max(cer_count, 1)
avg_wer = total_wer / max(wer_count, 1)

# CER/WER для case-insensitive
total_cer_ci = sum(character_error_rate(r.lower(), h.lower()) for r, h in zip(refs, hyps)) / max(len(refs), 1)
total_wer_ci = sum(word_error_rate(r.lower(), h.lower()) for r, h in zip(refs, hyps)) / max(len(refs), 1)

# CER/WER для chars only (пропускаем пары с пустыми строками)
total_cer_chars_only = 0.0
total_wer_chars_only = 0.0
chars_only_count = 0

for r, h in zip(refs_chars_only, hyps_chars_only):
    # Пропускаем пары где хотя бы одна строка пустая
    if not r and not h:
        # Обе пустые - считаем как совпадение (CER=0, WER=0)
        chars_only_count += 1
        continue
    elif not r or not h:
        # Одна пустая, другая нет - считаем как полную ошибку
        total_cer_chars_only += 1.0
        total_wer_chars_only += 1.0
        chars_only_count += 1
        continue
    
    # Обе непустые - считаем нормально
    total_cer_chars_only += character_error_rate(r, h)
    total_wer_chars_only += word_error_rate(r, h)
    chars_only_count += 1

total_cer_chars_only = total_cer_chars_only / max(chars_only_count, 1)
total_wer_chars_only = total_wer_chars_only / max(chars_only_count, 1)

print("=" * 80)
print("📈 SUMMARY METRICS")
print("=" * 80)
print(f"Accuracy (case-sensitive):     {acc*100:.2f}%")
print(f"Accuracy (case-insensitive):   {acc_case_insensitive*100:.2f}%")
print(f"Accuracy (chars only):         {acc_chars_only*100:.2f}%")
print(f"Case-only errors:              {case_only_errors} ({case_only_errors/max(len(refs), 1)*100:.2f}%)")
print(f"Avg CER:  {avg_cer:.4f}")
print(f"Avg WER:  {avg_wer:.4f}")
print(f"Processed {len(images)} images in {total_time:.3f} sec")
print(f"Average per image: {avg_time:.3f} sec ({fps:.1f} FPS)")
print("=" * 80)
print()

# ============================================
# ТАБЛИЦА МЕТРИК ПО ДАТАСЕТАМ
# ============================================

print("=" * 100)
print("📊 МЕТРИКИ ПО ДАТАСЕТАМ")
print("=" * 100)

# Подготовка данных для таблицы
metrics_table = []

for dataset_name, data in dataset_results.items():
    if data['count'] == 0:
        continue
    
    # Accuracy (case-sensitive)
    acc_ds = compute_accuracy(data['refs'], data['hyps'])
    
    # Accuracy (case-insensitive)
    acc_ci_ds = sum(1 for r, h in zip(data['refs'], data['hyps']) if r.lower() == h.lower()) / max(data['count'], 1)
    
    # Accuracy (chars only)
    refs_co_ds = [filter_chars_only(r) for r in data['refs']]
    hyps_co_ds = [filter_chars_only(h) for h in data['hyps']]
    acc_co_ds = sum(1 for r, h in zip(refs_co_ds, hyps_co_ds) if r.lower() == h.lower()) / max(data['count'], 1)
    
    # CER/WER
    avg_cer_ds = data['total_cer'] / data['count']
    avg_wer_ds = data['total_wer'] / data['count']
    
    # CER/WER (case-insensitive)
    cer_ci_ds = sum(character_error_rate(r.lower(), h.lower()) for r, h in zip(data['refs'], data['hyps'])) / data['count']
    wer_ci_ds = sum(word_error_rate(r.lower(), h.lower()) for r, h in zip(data['refs'], data['hyps'])) / data['count']
    
    # CER/WER (chars only) - с обработкой пустых строк
    total_cer_co_ds = 0.0
    total_wer_co_ds = 0.0
    co_count_ds = 0
    
    for r, h in zip(refs_co_ds, hyps_co_ds):
        if not r and not h:
            co_count_ds += 1
            continue
        elif not r or not h:
            total_cer_co_ds += 1.0
            total_wer_co_ds += 1.0
            co_count_ds += 1
            continue
        total_cer_co_ds += character_error_rate(r, h)
        total_wer_co_ds += word_error_rate(r, h)
        co_count_ds += 1
    
    cer_co_ds = total_cer_co_ds / max(co_count_ds, 1)
    wer_co_ds = total_wer_co_ds / max(co_count_ds, 1)
    
    metrics_table.append({
        'Dataset': dataset_name,
        'Count': data['count'],
        'Acc (CS)': acc_ds,
        'Acc (CI)': acc_ci_ds,
        'Acc (CO)': acc_co_ds,
        'CER (CS)': avg_cer_ds,
        'CER (CI)': cer_ci_ds,
        'CER (CO)': cer_co_ds,
        'WER (CS)': avg_wer_ds,
        'WER (CI)': wer_ci_ds,
        'WER (CO)': wer_co_ds,
    })

# Добавляем общую строку (TOTAL)
metrics_table.append({
    'Dataset': 'TOTAL',
    'Count': len(refs),
    'Acc (CS)': acc,
    'Acc (CI)': acc_case_insensitive,
    'Acc (CO)': acc_chars_only,
    'CER (CS)': avg_cer,
    'CER (CI)': total_cer_ci,
    'CER (CO)': total_cer_chars_only,
    'WER (CS)': avg_wer,
    'WER (CI)': total_wer_ci,
    'WER (CO)': total_wer_chars_only,
})

# Вывод таблицы
print(f"\n{'Dataset':<30} {'Count':>6} {'Acc(CS)':>8} {'Acc(CI)':>8} {'Acc(CO)':>8} {'CER(CS)':>8} {'CER(CI)':>8} {'CER(CO)':>8} {'WER(CS)':>8} {'WER(CI)':>8} {'WER(CO)':>8}")
print("-" * 100)
for row in metrics_table:
    is_total = row['Dataset'] == 'TOTAL'
    sep = "=" if is_total else "-"
    if is_total:
        print(sep * 100)
    print(f"{row['Dataset']:<30} {row['Count']:>6} {row['Acc (CS)']*100:>7.2f}% {row['Acc (CI)']*100:>7.2f}% {row['Acc (CO)']*100:>7.2f}% "
          f"{row['CER (CS)']:>8.4f} {row['CER (CI)']:>8.4f} {row['CER (CO)']:>8.4f} "
          f"{row['WER (CS)']:>8.4f} {row['WER (CI)']:>8.4f} {row['WER (CO)']:>8.4f}")

# Сохранение таблицы в CSV
csv_output_path = OUTPUT_DIR / "metrics_by_dataset.csv"
with open(csv_output_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=[
        'Dataset', 'Count', 
        'Acc (CS)', 'Acc (CI)', 'Acc (CO)',
        'CER (CS)', 'CER (CI)', 'CER (CO)',
        'WER (CS)', 'WER (CI)', 'WER (CO)'
    ])
    writer.writeheader()
    for row in metrics_table:
        writer.writerow(row)

print(f"\n💾 Таблица метрик сохранена: {csv_output_path}")
print("=" * 100)
print()

# ============================================
# ДЕТАЛЬНЫЙ АНАЛИЗ ОШИБОК
# ============================================

def analyze_character_errors(refs, hyps):
    """Анализ ошибок на уровне символов"""
    substitutions = Counter()
    insertions = Counter()
    deletions = Counter()
    error_positions = {'start': 0, 'middle': 0, 'end': 0}
    
    for ref, hyp in zip(refs, hyps):
        if ref == hyp:
            continue
            
        ops = Levenshtein.editops(ref, hyp)
        
        for op_type, ref_pos, hyp_pos in ops:
            word_len = len(ref)
            if ref_pos < word_len * 0.2:
                error_positions['start'] += 1
            elif ref_pos > word_len * 0.8:
                error_positions['end'] += 1
            else:
                error_positions['middle'] += 1
            
            if op_type == 'replace':
                ref_char = ref[ref_pos] if ref_pos < len(ref) else ''
                hyp_char = hyp[hyp_pos] if hyp_pos < len(hyp) else ''
                substitutions[(ref_char, hyp_char)] += 1
            elif op_type == 'insert':
                hyp_char = hyp[hyp_pos] if hyp_pos < len(hyp) else ''
                insertions[hyp_char] += 1
            elif op_type == 'delete':
                ref_char = ref[ref_pos] if ref_pos < len(ref) else ''
                deletions[ref_char] += 1
    
    return substitutions, insertions, deletions, error_positions


def analyze_word_lengths(error_details):
    """Анализ длины слов с ошибками"""
    return [len(detail['ref']) for detail in error_details]


def analyze_error_types(error_details):
    """Анализ типов ошибок"""
    total_errors = len(error_details)
    length_mismatch = 0
    case_errors = 0
    similar_chars = 0
    completely_wrong = 0
    
    for detail in error_details:
        ref = detail['ref']
        hyp = detail['hyp']
        
        if len(ref) != len(hyp):
            length_mismatch += 1
        elif ref.lower() == hyp.lower():
            case_errors += 1
        else:
            distance = Levenshtein.distance(ref, hyp)
            if distance <= 2:
                similar_chars += 1
            else:
                completely_wrong += 1
    
    return {
        'total': total_errors,
        'length_mismatch': length_mismatch,
        'case_errors': case_errors,
        'similar_chars': similar_chars,
        'completely_wrong': completely_wrong
    }


if error_details:
    print("=" * 80)
    print("📊 DETAILED ERROR ANALYSIS")
    print("=" * 80)
    
    # 1. Общая статистика
    print(f"\n1️⃣ GENERAL STATISTICS:")
    print(f"   Total examples: {len(refs)}")
    print(f"   Correct: {len(refs) - len(error_details)}")
    print(f"   With errors: {len(error_details)} ({len(error_details)/len(refs)*100:.1f}%)")
    
    print(f"\n   📏 Metrics (case-sensitive):")
    print(f"      Accuracy: {acc*100:.2f}%")
    print(f"      CER: {avg_cer:.4f}")
    print(f"      WER: {avg_wer:.4f}")
    
    print(f"\n   📏 Metrics (case-insensitive):")
    print(f"      Accuracy: {acc_case_insensitive*100:.2f}%")
    print(f"      CER: {total_cer_ci:.4f}")
    print(f"      WER: {total_wer_ci:.4f}")
    if avg_cer > 0:
        print(f"      CER improvement: {(avg_cer - total_cer_ci)/avg_cer*100:.1f}%")
    if avg_wer > 0:
        print(f"      WER improvement: {(avg_wer - total_wer_ci)/avg_wer*100:.1f}%")
    
    print(f"\n   📏 Metrics (chars only - no special chars/spaces):")
    print(f"      Accuracy: {acc_chars_only*100:.2f}%")
    print(f"      CER: {total_cer_chars_only:.4f}")
    print(f"      WER: {total_wer_chars_only:.4f}")
    if avg_cer > 0:
        print(f"      CER improvement: {(avg_cer - total_cer_chars_only)/avg_cer*100:.1f}%")
    if avg_wer > 0:
        print(f"      WER improvement: {(avg_wer - total_wer_chars_only)/avg_wer*100:.1f}%")
    
    # 2. Типы ошибок
    print(f"\n2️⃣ ERROR TYPES:")
    error_types = analyze_error_types(error_details)
    print(f"   Different length: {error_types['length_mismatch']} ({error_types['length_mismatch']/error_types['total']*100:.1f}%)")
    print(f"   Case only: {error_types['case_errors']} ({error_types['case_errors']/error_types['total']*100:.1f}%)")
    print(f"   Similar (1-2 chars): {error_types['similar_chars']} ({error_types['similar_chars']/error_types['total']*100:.1f}%)")
    print(f"   Completely wrong: {error_types['completely_wrong']} ({error_types['completely_wrong']/error_types['total']*100:.1f}%)")
    
    # 3. Длина слов с ошибками
    print(f"\n3️⃣ ERROR WORD LENGTHS:")
    error_lengths = analyze_word_lengths(error_details)
    if error_lengths:
        avg_error_len = sum(error_lengths) / len(error_lengths)
        print(f"   Average length: {avg_error_len:.1f} characters")
        print(f"   Min: {min(error_lengths)}, Max: {max(error_lengths)}")
        
        length_dist = Counter(error_lengths)
        print(f"   Distribution (top-10):")
        for length in sorted(length_dist.keys())[:10]:
            print(f"      {length} characters: {length_dist[length]} words")
    
    # 4. Анализ ошибок по символам
    print(f"\n4️⃣ CHARACTER-LEVEL ERROR ANALYSIS:")
    substitutions, insertions, deletions, error_positions = analyze_character_errors(refs, hyps)
    
    total_pos = sum(error_positions.values())
    if total_pos > 0:
        print(f"   Error position in word:")
        print(f"      Start (0-20%): {error_positions['start']} ({error_positions['start']/total_pos*100:.1f}%)")
        print(f"      Middle (20-80%): {error_positions['middle']} ({error_positions['middle']/total_pos*100:.1f}%)")
        print(f"      End (80-100%): {error_positions['end']} ({error_positions['end']/total_pos*100:.1f}%)")
    
    print(f"\n   🔄 Top-20 character substitutions (correct → wrong):")
    
    case_substitutions = []
    non_case_substitutions = []
    
    for (correct, wrong), count in substitutions.items():
        if correct.lower() == wrong.lower() and correct != wrong:
            case_substitutions.append(((correct, wrong), count))
        else:
            non_case_substitutions.append(((correct, wrong), count))
    
    case_substitutions.sort(key=lambda x: x[1], reverse=True)
    non_case_substitutions.sort(key=lambda x: x[1], reverse=True)
    
    if case_substitutions:
        print(f"\n      Case substitutions (top-10):")
        for (correct, wrong), count in case_substitutions[:10]:
            print(f"         '{correct}' → '{wrong}': {count} times")
    
    if non_case_substitutions:
        print(f"\n      Other substitutions (top-20):")
        for (correct, wrong), count in non_case_substitutions[:20]:
            print(f"         '{correct}' → '{wrong}': {count} times")
    
    if insertions:
        print(f"\n   ➕ Top-10 inserted characters:")
        for char, count in insertions.most_common(10):
            print(f"      '{char}': {count} times")
    
    if deletions:
        print(f"\n   ➖ Top-10 deleted characters:")
        for char, count in deletions.most_common(10):
            print(f"      '{char}': {count} times")
    
    # 5. Худшие примеры
    print(f"\n5️⃣ WORST EXAMPLES (top-10 by CER):")
    worst_examples = sorted(error_details, key=lambda x: x['cer'], reverse=True)[:10]
    for i, ex in enumerate(worst_examples, 1):
        print(f"   {i}. [{ex['fname']}]")
        print(f"      GT:   '{ex['ref']}'")
        print(f"      Pred: '{ex['hyp']}'")
        print(f"      CER: {ex['cer']:.3f}, Conf: {ex['confidence']:.3f}")
    
    # 6. Связь уверенности и ошибок
    print(f"\n6️⃣ CONFIDENCE VS ERRORS:")
    low_conf_errors = [e for e in error_details if e['confidence'] < 0.8]
    high_conf_errors = [e for e in error_details if e['confidence'] >= 0.8]
    print(f"   Errors with low confidence (<0.8): {len(low_conf_errors)} ({len(low_conf_errors)/len(error_details)*100:.1f}%)")
    print(f"   Errors with high confidence (≥0.8): {len(high_conf_errors)} ({len(high_conf_errors)/len(error_details)*100:.1f}%)")
    
    if error_details:
        avg_conf_errors = sum(e['confidence'] for e in error_details) / len(error_details)
        print(f"   Average confidence on errors: {avg_conf_errors:.3f}")
    
    # 7. Сохранение ошибок в CSV
    print(f"\n7️⃣ SAVING ERROR DETAILS TO CSV...")
    sorted_errors = sorted(error_details, key=lambda x: x['confidence'], reverse=True)
    
    output_csv = OUTPUT_DIR / "ocr_errors_by_confidence.csv"
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "confidence", "CER", "WER", "GT", "Prediction"])
        for err in sorted_errors:
            writer.writerow([
                err['fname'],
                f"{err['confidence']:.4f}",
                f"{err['cer']:.4f}",
                f"{err['wer']:.4f}",
                err['ref'],
                err['hyp'],
            ])
    
    print(f"   💾 Errors saved to: {output_csv}")
    
    print()
    print("=" * 80)
else:
    print("✅ NO ERRORS! All words recognized perfectly!")
    print("=" * 80)

print("\n✨ DONE!")
