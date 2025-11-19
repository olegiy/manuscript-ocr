"""
Запуск валидации используя ТУ ЖЕ логику что и при обучении.
Позволяет воспроизвести точные метрики обучения.
"""
import os
import json
import torch
import logging
from pathlib import Path

from manuscript.recognizers._trba.model.model import TRBAModel
from manuscript.recognizers._trba.data.dataset import OCRDatasetAttn
from manuscript.recognizers._trba.data.transforms import (
    load_charset,
    decode_tokens,
    get_train_transform,
    get_val_transform,
)
from manuscript.recognizers._trba.training.utils import load_checkpoint
from manuscript.recognizers._trba.training.metrics import (
    character_error_rate,
    word_error_rate,
    compute_accuracy,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

# === НАСТРОЙКИ ===
# Путь к чекпоинту (можно указать best_acc_ckpt.pth или last_ckpt.pth)
checkpoint_path = r"C:\Users\USER\Desktop\trba_exp_lite\best_acc_ckpt.pth"

# Путь к датасету для валидации
val_csv = r"C:\Users\USER\Desktop\archive_25_09\dataset\handwritten\val\labels.csv"
val_images_dir = r"C:\Users\USER\Desktop\archive_25_09\dataset\handwritten\val\img"

# Параметры
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*80)
print("🔄 ВАЛИДАЦИЯ С ИСПОЛЬЗОВАНИЕМ ЛОГИКИ ОБУЧЕНИЯ")
print("="*80)
print(f"🖥️  Device: {device}")
print(f"📦 Checkpoint: {os.path.basename(checkpoint_path)}")
print(f"📂 Validation CSV: {os.path.basename(val_csv)}")
print(f"📁 Images dir: {os.path.basename(val_images_dir)}")

# === Загружаем чекпоинт ===
print("\n📦 Загрузка чекпоинта...")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

# Извлекаем конфиг из чекпоинта
if 'config' in checkpoint:
    config = checkpoint['config']
    print("✅ Конфиг загружен из чекпоинта")
else:
    # Пробуем загрузить из config.json рядом с чекпоинтом
    config_path = Path(checkpoint_path).parent / "config.json"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✅ Конфиг загружен из {config_path.name}")
    else:
        raise FileNotFoundError(f"Не найден config в чекпоинте и файл {config_path}")

# Извлекаем параметры
max_len = config.get('max_len')
img_h = config.get('img_h')
img_w = config.get('img_w')
hidden_size = config.get('hidden_size')
num_encoder_layers = config.get('num_encoder_layers')
cnn_in_channels = config.get('cnn_in_channels', 1)
cnn_out_channels = config.get('cnn_out_channels', 512)
cnn_backbone = config.get('cnn_backbone', 'seresnet31lite')

print(f"\n📋 Параметры модели:")
print(f"   max_len: {max_len}")
print(f"   img_size: {img_h}x{img_w}")
print(f"   hidden_size: {hidden_size}")
print(f"   num_encoder_layers: {num_encoder_layers}")
print(f"   backbone: {cnn_backbone}")

# Загружаем charset
charset_path = Path(checkpoint_path).parent / "charset.txt"
if not charset_path.exists():
    raise FileNotFoundError(f"Не найден файл charset: {charset_path}")

print(f"\n📚 Загрузка charset из {charset_path.name}...")
itos, stoi = load_charset(str(charset_path))
num_classes = len(itos)
PAD = stoi["<PAD>"]
SOS = stoi["<SOS>"]
EOS = stoi["<EOS>"]
BLANK = stoi.get("<BLANK>", None)

print(f"   Размер алфавита: {num_classes} символов")
print(f"   Специальные токены: PAD={PAD}, SOS={SOS}, EOS={EOS}, BLANK={BLANK}")

# Информация о чекпоинте
if 'epoch' in checkpoint:
    print(f"\n📊 Информация о чекпоинте:")
    print(f"   Эпоха: {checkpoint['epoch']}")
    if 'val_acc' in checkpoint:
        print(f"   Val Accuracy: {checkpoint['val_acc']*100:.2f}%")
    if 'val_cer' in checkpoint:
        print(f"   Val CER: {checkpoint['val_cer']:.4f}")
    if 'val_loss' in checkpoint:
        print(f"   Val Loss: {checkpoint['val_loss']:.4f}")

# === Создаём модель ===
print("\n🔧 Создание модели...")
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

# Загружаем веса используя функцию из обучения
print(f"📦 Загрузка весов из чекпоинта...")
load_checkpoint(
    path=checkpoint_path,
    model=model,
    map_location=device,
    strict=False
)

model.eval()
print("✅ Модель загружена и готова к валидации")

# === Создаём датасет для валидации (КАК ПРИ ОБУЧЕНИИ) ===
print("\n📂 Создание валидационного датасета...")
val_transform = get_val_transform(img_h, img_w)

val_dataset = OCRDatasetAttn(
    csv_path=val_csv,
    images_dir=val_images_dir,
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

print(f"   ✅ Загружено {len(val_dataset)} примеров")

# Создаём collate_fn как при обучении
collate_val = OCRDatasetAttn.make_collate_attn(
    stoi, max_len=max_len, drop_blank=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_val,
    pin_memory=False
)

print(f"   DataLoader создан: {len(val_loader)} батчей")

# === ВАЛИДАЦИЯ (ТОЧНО КАК ПРИ ОБУЧЕНИИ) ===
print("\n🚀 Запуск валидации...")
print("="*80)

model.eval()
refs = []
hyps = []
total_loss = 0.0

with torch.no_grad():
    for imgs, text_in, target_y, lengths in tqdm(val_loader, desc="Валидация"):
        imgs = imgs.to(device)
        
        # Инференс (КАК В ОБУЧЕНИИ)
        result = model(
            imgs,
            is_train=False,
            batch_max_length=max_len,
            mode="attention"
        )
        
        # Получаем предсказания
        pred_ids = result["attention_preds"].cpu()
        tgt_ids = target_y.cpu()
        
        # Декодирование (КАК В ОБУЧЕНИИ)
        for pred_row, tgt_row in zip(pred_ids, tgt_ids):
            ref_text = decode_tokens(
                tgt_row, itos, pad_id=PAD, eos_id=EOS, blank_id=BLANK
            )
            hyp_text = decode_tokens(
                pred_row, itos, pad_id=PAD, eos_id=EOS, blank_id=BLANK
            )
            refs.append(ref_text)
            hyps.append(hyp_text)

# === Вычисление метрик (КАК В ОБУЧЕНИИ) ===
print("\n📊 Вычисление метрик...")

val_acc = compute_accuracy(refs, hyps)
val_cer = sum(
    character_error_rate(r, h) for r, h in zip(refs, hyps)
) / max(1, len(refs))
val_wer = sum(
    word_error_rate(r, h) for r, h in zip(refs, hyps)
) / max(1, len(refs))

# Дополнительные метрики
acc_case_insensitive = sum(
    1 for r, h in zip(refs, hyps) if r.lower() == h.lower()
) / max(len(refs), 1)

correct_count = sum(1 for r, h in zip(refs, hyps) if r == h)

print("\n" + "="*80)
print("📊 РЕЗУЛЬТАТЫ ВАЛИДАЦИИ (как при обучении)")
print("="*80)
print(f"\n{'Метрика':<40} {'Значение':>15}")
print("-" * 60)
print(f"{'Всего примеров':<40} {len(refs):>15}")
print(f"{'Правильно распознано':<40} {correct_count:>15}")
print(f"{'С ошибками':<40} {len(refs) - correct_count:>15}")
print("-" * 60)
print(f"{'Accuracy (case-sensitive)':<40} {val_acc*100:>14.2f}%")
print(f"{'Accuracy (case-insensitive)':<40} {acc_case_insensitive*100:>14.2f}%")
print(f"{'Character Error Rate (CER)':<40} {val_cer:>15.4f}")
print(f"{'Word Error Rate (WER)':<40} {val_wer:>15.4f}")
print("="*80)

# Сравнение с метриками из чекпоинта
if 'val_acc' in checkpoint:
    print("\n📈 СРАВНЕНИЕ С ЧЕКПОИНТОМ:")
    print("-" * 60)
    ckpt_acc = checkpoint.get('val_acc', 0.0)
    ckpt_cer = checkpoint.get('val_cer', 0.0)
    ckpt_wer = checkpoint.get('val_wer', 0.0)
    
    print(f"{'Метрика':<30} {'Чекпоинт':>15} {'Текущая':>15} {'Δ':>10}")
    print("-" * 60)
    print(f"{'Accuracy':<30} {ckpt_acc*100:>14.2f}% {val_acc*100:>14.2f}% {(val_acc-ckpt_acc)*100:>9.2f}%")
    if ckpt_cer > 0:
        print(f"{'CER':<30} {ckpt_cer:>15.4f} {val_cer:>15.4f} {val_cer-ckpt_cer:>10.4f}")
    if ckpt_wer > 0:
        print(f"{'WER':<30} {ckpt_wer:>15.4f} {val_wer:>15.4f} {val_wer-ckpt_wer:>10.4f}")
    print("="*80)

# === Худшие примеры ===
print("\n🔴 ХУДШИЕ ПРИМЕРЫ (топ-20 по CER):")
errors = []
for i, (r, h) in enumerate(zip(refs, hyps)):
    if r != h:
        cer = character_error_rate(r, h)
        img_path, _ = val_dataset.samples[i]
        errors.append({
            'fname': os.path.basename(img_path),
            'ref': r,
            'hyp': h,
            'cer': cer
        })

worst_errors = sorted(errors, key=lambda x: x['cer'], reverse=True)[:20]
for i, err in enumerate(worst_errors, 1):
    print(f"   {i}. [{err['fname']}]")
    print(f"      GT:   '{err['ref']}'")
    print(f"      Pred: '{err['hyp']}'")
    print(f"      CER: {err['cer']:.3f}")

print("\n" + "="*80)
print("✅ Валидация завершена")
print("="*80)
