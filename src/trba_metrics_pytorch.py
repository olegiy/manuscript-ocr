import os
import time
import csv
import re
import json
import torch
from collections import Counter, defaultdict
from manuscript.recognizers._trba.training.metrics import (
    character_error_rate,
    word_error_rate,
    compute_accuracy,
)
from manuscript.recognizers._trba.training.utils import load_checkpoint
from manuscript.recognizers._trba.data.dataset import OCRDatasetAttn
from manuscript.recognizers._trba.data.transforms import (
    load_charset,
    decode_tokens,
    get_val_transform,
)
from manuscript.recognizers._trba.model.model import TRBAModel
from torch.utils.data import DataLoader
import Levenshtein
from tqdm import tqdm


def normalize_text_letters_only(text: str) -> str:
    """
    Нормализует текст: оставляет только буквы, приводит к нижнему регистру.
    Удаляет пробелы, пунктуацию и цифры.
    """
    letters_only = re.sub(r'[^a-zA-Zа-яА-ЯёЁ\u0080-\uFFFF]', '', text)
    return letters_only.lower()


# === Пути ===
datasets = [
    {
        "image_dir": r"C:\shared\orig_cyrillic\test",
        "gt_path": r"C:\shared\orig_cyrillic\test.csv",
    },
]

# PyTorch модель
weights_path = r"C:\Users\USER\Desktop\trba_exp_lite\best_acc_weights.pth"
config_path = r"C:\Users\USER\Desktop\trba_exp_lite\config.json"
charset_path = r"C:\Users\USER\Desktop\trba_exp_lite\charset.txt"

batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🖥️  Device: {device}")

# === Загружаем конфиг ===
print("\n📋 Загрузка конфигурации...")
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

max_len = config.get('max_len')
img_h = config.get('img_h')
img_w = config.get('img_w')
hidden_size = config.get('hidden_size')
num_encoder_layers = config.get('num_encoder_layers')
cnn_in_channels = config.get('cnn_in_channels')
cnn_out_channels = config.get('cnn_out_channels')
cnn_backbone = config.get('cnn_backbone')

print(f"   max_len: {max_len}")
print(f"   img_size: {img_h}x{img_w}")
print(f"   backbone: {cnn_backbone}")

# === Загружаем charset ===
print("\n📚 Загрузка charset...")
itos, stoi = load_charset(charset_path)
num_classes = len(itos)
PAD = stoi["<PAD>"]
SOS = stoi["<SOS>"]
EOS = stoi["<EOS>"]
BLANK = stoi.get("<BLANK>", None)

print(f"   Размер алфавита: {num_classes} символов")
print(f"   Специальные токены: PAD={PAD}, SOS={SOS}, EOS={EOS}, BLANK={BLANK}")

# === Создаём PyTorch модель ===
print("\n🔧 Создание PyTorch модели...")
model = TRBAModel(
    num_classes=num_classes,
    hidden_size=hidden_size,
    num_encoder_layers=num_encoder_layers,
    img_h=img_h,
    img_w=img_w,
    cnn_in_channels=cnn_in_channels,
    cnn_out_channels=cnn_out_channels,
    cnn_backbone=cnn_backbone,
    use_ctc_head=False,
    use_attention_head=True,
)
model.to(device)

# Загружаем веса используя ту же функцию что и при обучении
print(f"📦 Загрузка весов из {os.path.basename(weights_path)}...")
checkpoint_data = load_checkpoint(
    path=weights_path,
    model=model,
    map_location=device,
    strict=False
)

model.eval()
print("✅ Модель загружена")

# === Читаем данные используя OCRDatasetAttn ===
print("\n📂 Загрузка датасета...")

val_transform = get_val_transform(img_h, img_w)

image_dir = datasets[0]["image_dir"]
gt_path = datasets[0]["gt_path"]

print(f"📂 Датасет: {os.path.basename(image_dir)}")
print(f"   CSV: {os.path.basename(gt_path)}")

ocr_ds = OCRDatasetAttn(
    csv_path=gt_path,
    images_dir=image_dir,
    stoi=stoi,
    img_height=img_h,
    img_max_width=img_w,
    transform=val_transform,
    has_header=None,
    encoding="utf-8",
    delimiter=None,
    strict_charset=False,
    validate_image=False,
    max_len=max_len,
    strict_max_len=True,
    num_workers=0
)

print(f"   ✅ Загружено {len(ocr_ds)} валидных примеров")

if hasattr(ocr_ds, '_reasons'):
    total_filtered = sum(ocr_ds._reasons.values())
    if total_filtered > 0:
        print(f"   ⚠️  Отфильтровано {total_filtered} примеров:")
        for reason, count in ocr_ds._reasons.items():
            if count > 0:
                print(f"      - {reason}: {count}")

# Создаём collate_fn как при обучении
collate_val = OCRDatasetAttn.make_collate_attn(
    stoi, max_len=max_len, drop_blank=True
)

dataloader = DataLoader(
    ocr_ds, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=0,
    collate_fn=collate_val,
    pin_memory=False
)

# === Распознавание ===
print(f"\n🚀 Начало распознавания (PyTorch)...")
start_time = time.perf_counter()

refs, hyps = [], []
error_details = []

# Счетчик для отслеживания индекса примера
sample_idx = 0

with torch.no_grad():
    for batch_imgs, text_in, target_y, lengths in tqdm(dataloader, desc="Распознавание"):
        batch_imgs = batch_imgs.to(device)
        
        # Инференс (как при валидации в обучении)
        result = model(
            batch_imgs,
            is_train=False,
            batch_max_length=max_len,
            mode="attention"
        )
        
        pred_ids = result["attention_preds"].cpu()
        tgt_ids = target_y.cpu()
        
        # Декодирование (как при обучении)
        for pred_row, tgt_row in zip(pred_ids, tgt_ids):
            # Декодируем референс
            ref_text = decode_tokens(tgt_row, itos, pad_id=PAD, eos_id=EOS, blank_id=BLANK)
            # Декодируем предсказание
            hyp_text = decode_tokens(pred_row, itos, pad_id=PAD, eos_id=EOS, blank_id=BLANK)
            
            refs.append(ref_text)
            hyps.append(hyp_text)
            
            # Получаем путь к файлу из датасета
            img_path, _ = ocr_ds.samples[sample_idx]
            sample_idx += 1
            
            # Вычисляем ошибки
            if ref_text != hyp_text:
                cer = character_error_rate(ref_text, hyp_text)
                wer = word_error_rate(ref_text, hyp_text)
                
                error_details.append({
                    'fname': os.path.basename(img_path),
                    'ref': ref_text,
                    'hyp': hyp_text,
                    'cer': cer,
                    'wer': wer,
                    'confidence': 0.0,  # Нет confidence в PyTorch режиме
                    'dataset_id': 1  # У нас один датасет
                })

end_time = time.perf_counter()
total_time = end_time - start_time
avg_time = total_time / len(refs)
fps = 1.0 / avg_time if avg_time > 0 else float("inf")

print(f"\n✅ Распознавание завершено")
print(f"   Время: {total_time:.2f} сек")
print(f"   Скорость: {fps:.1f} FPS ({avg_time*1000:.1f} мс/изображение)")

# === Метрики ===
acc = compute_accuracy(refs, hyps)
acc_case_insensitive = sum(1 for r, h in zip(refs, hyps) if r.lower() == h.lower()) / max(len(refs), 1)
acc_letters_only = sum(
    1 for r, h in zip(refs, hyps) 
    if normalize_text_letters_only(r) == normalize_text_letters_only(h)
) / max(len(refs), 1)

case_only_errors = sum(1 for r, h in zip(refs, hyps) if r.lower() == h.lower() and r != h)

total_cer = sum(character_error_rate(r, h) for r, h in zip(refs, hyps))
total_wer = sum(word_error_rate(r, h) for r, h in zip(refs, hyps))
avg_cer = total_cer / max(len(refs), 1)
avg_wer = total_wer / max(len(refs), 1)

print("\n" + "="*120)
print("📊 МЕТРИКИ (PyTorch)")
print("="*120)

# Вычисляем CER case-insensitive и letters-only для overall
total_cer_ci = sum(character_error_rate(r.lower(), h.lower()) for r, h in zip(refs, hyps))
avg_cer_ci = total_cer_ci / max(len(refs), 1)

total_cer_letters = 0.0
for r, h in zip(refs, hyps):
    r_letters = normalize_text_letters_only(r)
    h_letters = normalize_text_letters_only(h)
    if r_letters:
        total_cer_letters += character_error_rate(r_letters, h_letters)
avg_cer_letters = total_cer_letters / max(len(refs), 1)

print(f"\n{'Метрика':<30} {'Значение':>15}")
print("-" * 50)
print(f"{'Всего примеров':<30} {len(refs):>15}")
print(f"{'Accuracy (case-sensitive)':<30} {acc*100:>14.2f}%")
print(f"{'Accuracy (case-insensitive)':<30} {acc_case_insensitive*100:>14.2f}%")
print(f"{'Accuracy (letters only)':<30} {acc_letters_only*100:>14.2f}%")
print(f"{'CER':<30} {avg_cer:>15.4f}")
print(f"{'CER (case-insensitive)':<30} {avg_cer_ci:>15.4f}")
print(f"{'CER (letters only)':<30} {avg_cer_letters:>15.4f}")
print(f"{'WER':<30} {avg_wer:>15.4f}")
print(f"{'Case-only errors':<30} {case_only_errors:>15}")
print("-" * 50)

print("\nЛегенда:")
print("  Acc      - Accuracy (case-sensitive)")
print("  Acc-CI   - Accuracy (case-insensitive)")
print("  Acc-L    - Accuracy (letters only, case-insensitive)")
print("  CER      - Character Error Rate")
print("  CER-CI   - CER (case-insensitive)")
print("  CER-L    - CER (letters only)")
print("  WER      - Word Error Rate")

print("\n=== Дополнительная информация ===")
print(f"Case-only errors: {case_only_errors} ({case_only_errors/max(len(refs), 1)*100:.2f}%)")

# === Худшие примеры ===
if error_details:
    print(f"\n🔴 ХУДШИЕ ПРИМЕРЫ (топ-20 по CER):")
    worst_examples = sorted(error_details, key=lambda x: x['cer'], reverse=True)[:20]
    for i, ex in enumerate(worst_examples, 1):
        print(f"   {i}. [{ex['fname']}]")
        print(f"      GT:   '{ex['ref']}'")
        print(f"      Pred: '{ex['hyp']}'")
        print(f"      CER: {ex['cer']:.3f}")
    
    print(f"\n📊 Статистика ошибок:")
    print(f"   Всего примеров: {len(refs)}")
    print(f"   Правильных: {len(refs) - len(error_details)}")
    print(f"   С ошибками: {len(error_details)} ({len(error_details)/len(refs)*100:.1f}%)")
else:
    print("\n✅ Нет ошибок! Все слова распознаны идеально!")

print("\n" + "="*80)
print("✅ PyTorch валидация завершена")
print("="*80)
