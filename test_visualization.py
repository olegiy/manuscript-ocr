"""Тест функции визуализации случайных предсказаний."""

import torch
from torch.utils.data import DataLoader
from src.manuscript.recognizers._trba.training.train import visualize_random_predictions
from src.manuscript.recognizers._trba.model.model import TRBAModel
from src.manuscript.recognizers._trba.data.dataset import OCRDatasetAttn
from src.manuscript.recognizers._trba.data.transforms import (
    load_charset,
    get_val_transform,
)
import os

print("=" * 80)
print("ТЕСТ ФУНКЦИИ ВИЗУАЛИЗАЦИИ СЛУЧАЙНЫХ ПРЕДСКАЗАНИЙ")
print("=" * 80)

# Параметры
charset_path = r"src\manuscript\recognizers\_trba\configs\charset.txt"
img_h = 32
img_w = 128
max_len = 30

# Загружаем charset
print("\n1. Загрузка charset...")
itos, stoi = load_charset(charset_path)
PAD = stoi["<PAD>"]
SOS = stoi["<SOS>"]
EOS = stoi["<EOS>"]
BLANK = stoi.get("<BLANK>", None)
num_classes = len(itos)
print(f"   Загружено {num_classes} символов")

# Проверяем наличие валидационных данных
val_csv = r"C:\shared\orig_cyrillic\test.tsv"
val_root = r"C:\shared\orig_cyrillic\test"

if not os.path.exists(val_csv):
    print(f"\n✗ Валидационный CSV не найден: {val_csv}")
    print("Пропускаем тест...")
    exit(0)

print(f"\n2. Создание валидационного датасета...")
print(f"   CSV: {val_csv}")
print(f"   Root: {val_root}")

try:
    val_dataset = OCRDatasetAttn(
        csv_path=val_csv,
        root_dir=val_root,
        stoi=stoi,
        sos_id=SOS,
        eos_id=EOS,
        pad_id=PAD,
        max_len=max_len,
        img_h=img_h,
        img_w=img_w,
        transform=get_val_transform(img_h=img_h, img_w=img_w),
        encoding="utf-8",
    )
    print(f"   ✓ Датасет создан, размер: {len(val_dataset)}")
except Exception as e:
    print(f"   ✗ Ошибка создания датасета: {e}")
    exit(1)

# Создаем DataLoader
print("\n3. Создание DataLoader...")
val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)
print(f"   ✓ DataLoader создан, {len(val_loader)} батчей")

# Создаем модель
print("\n4. Создание модели...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   Устройство: {device}")

model = TRBAModel(
    num_classes=num_classes,
    hidden_size=128,
    num_encoder_layers=1,
    encoder_type="GRU",
    img_h=img_h,
    img_w=img_w,
    sos_id=SOS,
    eos_id=EOS,
    pad_id=PAD,
    blank_id=BLANK,
).to(device)
print("   ✓ Модель создана")

# Тест визуализации
print("\n5. Тестирование функции visualize_random_predictions...")
print("\n" + "-" * 80)

try:
    # Тест с greedy декодированием
    visualize_random_predictions(
        model=model,
        val_loader=val_loader,
        itos=itos,
        pad_id=PAD,
        eos_id=EOS,
        blank_id=BLANK,
        device=device,
        num_samples=5,
        max_len=max_len,
        mode="greedy",
        epoch=1,
        logger=None,  # Используем print вместо logger
    )
    print("\n✓ Greedy декодирование работает!")

    print("\n" + "-" * 80)

    # Тест с beam search
    visualize_random_predictions(
        model=model,
        val_loader=val_loader,
        itos=itos,
        pad_id=PAD,
        eos_id=EOS,
        blank_id=BLANK,
        device=device,
        num_samples=5,
        max_len=max_len,
        mode="beam",
        epoch=1,
        logger=None,
        beam_size=8,
        alpha=0.9,
        temperature=1.7,
    )
    print("\n✓ Beam search декодирование работает!")

except Exception as e:
    print(f"\n✗ Ошибка визуализации: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

print("\n" + "=" * 80)
print("ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
print("=" * 80)
