"""
Тестирование разных архитектур TRBA на CPU.

Сравниваем:
1. Разное количество encoder layers (1, 2, 3, 4)
2. Разный hidden_size (128, 256, 512)
3. Режимы декодирования (greedy vs beam)

Цель: найти оптимальный баланс скорости и capacity для CPU.
"""

import time
import json
import torch
import cv2

# Импорты для создания моделей напрямую
from manuscript.recognizers._trba.model.model import TRBAModel
from manuscript.recognizers._trba.data.transforms import load_charset, get_val_transform

# === Конфигурация ===
TEST_IMAGE = r"C:\Users\USER\Desktop\t2.png"
NUM_IMAGES = 25  # Количество копий для теста
CHARSET_PATH = r"C:\Users\USER\manuscript-ocr\src\manuscript\recognizers\_trba\configs\charset.txt"

# Загрузка charset
itos, stoi = load_charset(CHARSET_PATH)
num_classes_full = len(itos)

print("=" * 80)
print("🔬 ТЕСТИРОВАНИЕ РАЗНЫХ АРХИТЕКТУР TRBA НА CPU")
print("=" * 80)
print(f"Тестовое изображение: {TEST_IMAGE}")
print(f"Количество тестов: {NUM_IMAGES} изображений")
print(f"Полный словарь: {num_classes_full} символов")
print()

# === Подготовка данных ===
transform = get_val_transform(img_h=64, img_w=256)

# Загрузка изображения
img = cv2.imread(TEST_IMAGE)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
transformed = transform(image=img)
image_tensor = transformed["image"].unsqueeze(0)  # [1, 3, 64, 256]

# Создаём батч
batch_tensor = image_tensor.repeat(NUM_IMAGES, 1, 1, 1)  # [N, 3, 64, 256]
print(f"✅ Батч подготовлен: {batch_tensor.shape}")
print()

# === Конфигурации для тестирования ===
configs = [
    # Название, num_encoder_layers, hidden_size, img_h, img_w, cnn_in_channels, cnn_out_channels, описание
    ("Микро (1×128, 32×128, CNN256)", 1, 128, 32, 128, 3, 256, "Минимальная: маленькое изображение + легкий CNN"),
    ("Легкая CNN (2×256, CNN256)", 2, 256, 64, 256, 3, 256, "Легкий CNN backbone"),
    ("Средняя CNN (2×256, CNN384)", 2, 256, 64, 256, 3, 384, "Средний CNN backbone"),
    ("Стандарт (2×256, CNN512)", 2, 256, 64, 256, 3, 512, "Default конфигурация"),
    ("Тяжелая CNN (2×256, CNN768)", 2, 256, 64, 256, 3, 768, "Тяжелый CNN backbone"),
    ("Легкая + 1 encoder (1×256, CNN384)", 1, 256, 64, 256, 3, 384, "1 encoder + средний CNN"),
    ("Глубокая + легкая CNN (3×256, CNN384)", 3, 256, 64, 256, 3, 384, "3 encoder + средний CNN"),
    ("Тяжелая (2×512, CNN512)", 2, 512, 64, 256, 3, 512, "Большой hidden_size"),
]

results = []

# === Тестирование каждой конфигурации ===
for config_name, num_enc_layers, hidden_size, img_h, img_w, cnn_in, cnn_out, description in configs:
    print("=" * 80)
    print(f"📊 Тест: {config_name}")
    print(f"   Описание: {description}")
    print(f"   Параметры: enc_layers={num_enc_layers}, hidden={hidden_size}, "
          f"img={img_h}×{img_w}, CNN_in={cnn_in}, CNN_out={cnn_out}")
    print("-" * 80)
    
    # Ресайз изображения под нужные размеры
    resized_img = cv2.resize(img, (img_w, img_h))
    transform_temp = get_val_transform(img_h=img_h, img_w=img_w)
    transformed_temp = transform_temp(image=resized_img)
    image_tensor_temp = transformed_temp["image"].unsqueeze(0)
    batch_tensor_temp = image_tensor_temp.repeat(NUM_IMAGES, 1, 1, 1)
    
    # Создание модели
    model = TRBAModel(
        num_classes=num_classes_full,
        hidden_size=hidden_size,
        num_encoder_layers=num_enc_layers,
        img_h=img_h,
        img_w=img_w,
        cnn_in_channels=cnn_in,
        cnn_out_channels=cnn_out,
        sos_id=stoi["<SOS>"],
        eos_id=stoi["<EOS>"],
        pad_id=stoi["<PAD>"],
        blank_id=stoi.get("<BLANK>", None),
    )
    model.eval()
    
    # Подсчёт параметров
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.enc_rnn.parameters())
    cnn_params = sum(p.numel() for p in model.cnn.parameters())
    attn_params = sum(p.numel() for p in model.attn.parameters())
    
    print(f"   Параметры:")
    print(f"      Total:    {total_params:>10,} ({total_params*4/(1024*1024):>6.2f} MB)")
    print(f"      CNN:      {cnn_params:>10,} ({cnn_params/total_params*100:>5.1f}%)")
    print(f"      Encoder:  {encoder_params:>10,} ({encoder_params/total_params*100:>5.1f}%)")
    print(f"      Attention: {attn_params:>10,} ({attn_params/total_params*100:>5.1f}%)")
    
    # === Тест 1: Greedy mode ===
    print("\n   🏃 Greedy mode:")
    
    # Прогрев
    with torch.no_grad():
        _ = model(batch_tensor_temp[:2], is_train=False, mode="greedy", batch_max_length=25)
    
    # Замер
    start = time.perf_counter()
    with torch.no_grad():
        probs_greedy, preds_greedy = model(
            batch_tensor_temp,
            is_train=False,
            mode="greedy",
            batch_max_length=25
        )
    greedy_time = time.perf_counter() - start
    greedy_fps = NUM_IMAGES / greedy_time
    
    print(f"      Время: {greedy_time:.3f}s ({greedy_fps:.1f} img/s)")
    print(f"      На изображение: {greedy_time/NUM_IMAGES*1000:.1f}ms")
    
    # === Тест 2: Beam search mode ===
    print("\n   🔍 Beam search mode (beam_size=8):")
    
    # Прогрев
    with torch.no_grad():
        _ = model(batch_tensor_temp[:2], is_train=False, mode="beam", 
                 beam_size=8, batch_max_length=25)
    
    # Замер
    start = time.perf_counter()
    with torch.no_grad():
        probs_beam, preds_beam = model(
            batch_tensor_temp,
            is_train=False,
            mode="beam",
            beam_size=8,
            batch_max_length=25
        )
    beam_time = time.perf_counter() - start
    beam_fps = NUM_IMAGES / beam_time
    
    print(f"      Время: {beam_time:.3f}s ({beam_fps:.1f} img/s)")
    print(f"      На изображение: {beam_time/NUM_IMAGES*1000:.1f}ms")
    print(f"      Медленнее greedy в: {beam_time/greedy_time:.1f}×")
    
    # Сохранение результатов
    results.append({
        "name": config_name,
        "description": description,
        "num_encoder_layers": num_enc_layers,
        "hidden_size": hidden_size,
        "img_h": img_h,
        "img_w": img_w,
        "cnn_in_channels": cnn_in,
        "cnn_out_channels": cnn_out,
        "total_params": total_params,
        "encoder_params": encoder_params,
        "cnn_params": cnn_params,
        "attn_params": attn_params,
        "greedy_time": greedy_time,
        "greedy_fps": greedy_fps,
        "greedy_ms_per_img": greedy_time / NUM_IMAGES * 1000,
        "beam_time": beam_time,
        "beam_fps": beam_fps,
        "beam_ms_per_img": beam_time / NUM_IMAGES * 1000,
        "beam_slowdown": beam_time / greedy_time,
    })
    
    print()

# === Итоговое сравнение ===
print("=" * 80)
print("📈 ИТОГОВОЕ СРАВНЕНИЕ")
print("=" * 80)
print()

# Таблица greedy mode
print("🏃 GREEDY MODE:")
print("-" * 80)
print(f"{'Конфигурация':<35} {'Params':>10} {'CNN%':>6} {'Time':>8} {'FPS':>8} {'ms/img':>8}")
print("-" * 80)

baseline_greedy = results[0]["greedy_time"]
for r in results:
    speedup = baseline_greedy / r["greedy_time"]
    marker = "⭐" if speedup > 1.2 else "  "
    cnn_pct = r["cnn_params"] / r["total_params"] * 100
    print(f"{marker}{r['name']:<33} {r['total_params']:>10,} {cnn_pct:>5.1f}% "
          f"{r['greedy_time']:>7.2f}s {r['greedy_fps']:>7.1f} {r['greedy_ms_per_img']:>7.1f}")

print()

# Таблица beam mode
print("🔍 BEAM SEARCH MODE (beam_size=8):")
print("-" * 80)
print(f"{'Конфигурация':<35} {'Params':>10} {'CNN%':>6} {'Time':>8} {'FPS':>8} {'ms/img':>8}")
print("-" * 80)

baseline_beam = results[0]["beam_time"]
for r in results:
    speedup = baseline_beam / r["beam_time"]
    marker = "⭐" if speedup > 1.2 else "  "
    cnn_pct = r["cnn_params"] / r["total_params"] * 100
    print(f"{marker}{r['name']:<33} {r['total_params']:>10,} {cnn_pct:>5.1f}% "
          f"{r['beam_time']:>7.2f}s {r['beam_fps']:>7.1f} {r['beam_ms_per_img']:>7.1f}")

print()

# Детальная таблица по компонентам
print("📦 ДЕТАЛЬНАЯ РАЗБИВКА ПО КОМПОНЕНТАМ:")
print("-" * 100)
print(f"{'Конфигурация':<35} {'Total MB':>9} {'CNN MB':>9} {'Enc MB':>9} {'Attn MB':>9}")
print("-" * 100)

for r in results:
    total_mb = r['total_params'] * 4 / (1024 * 1024)
    cnn_mb = r['cnn_params'] * 4 / (1024 * 1024)
    enc_mb = r['encoder_params'] * 4 / (1024 * 1024)
    attn_mb = r['attn_params'] * 4 / (1024 * 1024)
    
    print(f"{r['name']:<35} {total_mb:>8.1f}  {cnn_mb:>8.1f}  {enc_mb:>8.1f}  {attn_mb:>8.1f}")

print()

# Сравнение с baseline
print("📊 СРАВНЕНИЕ С BASELINE (Микро):")
print("-" * 80)
print(f"{'Конфигурация':<35} {'Greedy':>12} {'Beam':>12} {'Params':>12}")
print("-" * 80)

for r in results:
    greedy_ratio = r["greedy_time"] / baseline_greedy
    beam_ratio = r["beam_time"] / baseline_beam
    params_ratio = r["total_params"] / results[0]["total_params"]
    
    print(f"{r['name']:<35} "
          f"{greedy_ratio:>11.2f}× {beam_ratio:>11.2f}× {params_ratio:>11.2f}×")

print()

# Рекомендации
print("=" * 80)
print("💡 РЕКОМЕНДАЦИИ")
print("=" * 80)

# Находим самую быструю для greedy
fastest_greedy = min(results, key=lambda x: x["greedy_time"])
print(f"\n🏆 Самая быстрая (greedy): {fastest_greedy['name']}")
print(f"   {fastest_greedy['greedy_fps']:.1f} img/s ({fastest_greedy['greedy_ms_per_img']:.1f}ms/img)")
print(f"   Размер: {fastest_greedy['total_params']*4/(1024*1024):.1f} MB")

# Находим лучший баланс (скорость/качество)
# Ищем модель ~20-30 MB, с хорошей скоростью
best_balance = None
for r in results:
    mb = r['total_params'] * 4 / (1024 * 1024)
    if 15 < mb < 35 and r['greedy_fps'] > fastest_greedy['greedy_fps'] * 0.7:
        if best_balance is None or r['greedy_fps'] > best_balance['greedy_fps']:
            best_balance = r

if best_balance:
    print(f"\n⚖️  Лучший баланс: {best_balance['name']}")
    print(f"   {best_balance['greedy_fps']:.1f} img/s (greedy), {best_balance['beam_fps']:.1f} img/s (beam)")
    print(f"   Размер: {best_balance['total_params']*4/(1024*1024):.1f} MB")

# Находим самую компактную
smallest = min(results, key=lambda x: x["total_params"])
print(f"\n📦 Самая компактная: {smallest['name']}")
print(f"   {smallest['total_params']:,} параметров ({smallest['total_params']*4/(1024*1024):.1f} MB)")
print(f"   {smallest['greedy_fps']:.1f} img/s (greedy)")

# Анализ влияния CNN
print(f"\n🔬 ВЛИЯНИЕ CNN OUT_CHANNELS НА РАЗМЕР:")
cnn_configs = [r for r in results if r['num_encoder_layers'] == 2 and r['hidden_size'] == 256]
if len(cnn_configs) > 1:
    for r in cnn_configs:
        cnn_mb = r['cnn_params'] * 4 / (1024 * 1024)
        total_mb = r['total_params'] * 4 / (1024 * 1024)
        print(f"   CNN={r['cnn_out_channels']}: {cnn_mb:.1f} MB ({r['cnn_params']/r['total_params']*100:.1f}% от {total_mb:.1f} MB)")

print()
print("=" * 80)

# Сохранение результатов
output_file = "architecture_benchmark_results.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"✅ Результаты сохранены в: {output_file}")
print("=" * 80)
