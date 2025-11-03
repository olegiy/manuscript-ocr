"""
Детальный анализ размера модели TRBA по компонентам.
Показывает, сколько параметров и мегабайт занимает каждая часть.
"""

import sys
import os

# Добавляем путь к src
sys.path.insert(0, r"C:\Users\USER\manuscript-ocr\src")

import torch

# Прямой импорт без пакета manuscript
from manuscript.recognizers._trba.model.model import TRBAModel

# Простая загрузка charset без импорта transforms
def load_charset(charset_path):
    with open(charset_path, 'r', encoding='utf-8') as f:
        chars = [line.strip() for line in f if line.strip()]
    itos = chars
    stoi = {char: idx for idx, char in enumerate(chars)}
    return itos, stoi

CHARSET_PATH = r"C:\Users\USER\manuscript-ocr\src\manuscript\recognizers\_trba\configs\charset.txt"

# Загрузка charset
itos, stoi = load_charset(CHARSET_PATH)
num_classes = len(itos)

print("=" * 80)
print("📊 ДЕТАЛЬНЫЙ АНАЛИЗ РАЗМЕРА МОДЕЛИ TRBA")
print("=" * 80)
print(f"Количество классов: {num_classes}")
print()

# Создаём стандартную модель (2×256)
model = TRBAModel(
    num_classes=num_classes,
    hidden_size=256,
    num_encoder_layers=2,
    img_h=64,
    img_w=256,
    cnn_in_channels=3,
    cnn_out_channels=512,
    sos_id=stoi["<SOS>"],
    eos_id=stoi["<EOS>"],
    pad_id=stoi["<PAD>"],
    blank_id=stoi.get("<BLANK>", None),
)

def count_parameters(module):
    """Подсчёт параметров модуля"""
    return sum(p.numel() for p in module.parameters())

def format_size(num_params):
    """Форматирование размера в MB (float32)"""
    mb = num_params * 4 / (1024 * 1024)  # 4 байта на параметр (float32)
    return mb

def analyze_module(name, module, indent=0):
    """Рекурсивный анализ модуля"""
    params = count_parameters(module)
    mb = format_size(params)
    prefix = "  " * indent
    print(f"{prefix}{name:<40} {params:>12,} params  ({mb:>7.2f} MB)")
    return params, mb

print("=" * 80)
print("ОБЩАЯ СТРУКТУРА")
print("=" * 80)

total_params = count_parameters(model)
total_mb = format_size(total_params)

# Анализируем основные компоненты
cnn_params, cnn_mb = analyze_module("1. CNN (SEResNet31)", model.cnn)
pool_params = 0  # AdaptiveAvgPool не имеет параметров
enc_rnn_params, enc_rnn_mb = analyze_module("2. Encoder RNN (BiLSTM)", model.enc_rnn)
attn_params, attn_mb = analyze_module("3. Attention Decoder", model.attn)

print("-" * 80)
print(f"{'ИТОГО':<40} {total_params:>12,} params  ({total_mb:>7.2f} MB)")
print()

# Процентное соотношение
print("=" * 80)
print("ПРОЦЕНТНОЕ СООТНОШЕНИЕ")
print("=" * 80)
print(f"CNN:              {cnn_params/total_params*100:>6.2f}%  ({cnn_mb:>7.2f} MB)")
print(f"Encoder RNN:      {enc_rnn_params/total_params*100:>6.2f}%  ({enc_rnn_mb:>7.2f} MB)")
print(f"Attention Decoder: {attn_params/total_params*100:>6.2f}%  ({attn_mb:>7.2f} MB)")
print()

# Детальный анализ CNN
print("=" * 80)
print("ДЕТАЛИ CNN (SEResNet31)")
print("=" * 80)

# Анализ слоёв CNN
for name, module in model.cnn.named_children():
    params = count_parameters(module)
    mb = format_size(params)
    print(f"  {name:<38} {params:>12,} params  ({mb:>7.2f} MB)")

print()

# Детальный анализ Encoder RNN
print("=" * 80)
print("ДЕТАЛИ ENCODER RNN (BiLSTM)")
print("=" * 80)

for idx, layer in enumerate(model.enc_rnn):
    params = count_parameters(layer)
    mb = format_size(params)
    print(f"  BiLSTM Layer {idx+1:<28} {params:>12,} params  ({mb:>7.2f} MB)")
    
    # Детали BiLSTM слоя
    rnn_params = count_parameters(layer.rnn)
    linear_params = count_parameters(layer.linear)
    print(f"    ├─ LSTM (bidirectional)        {rnn_params:>12,} params  ({format_size(rnn_params):>7.2f} MB)")
    print(f"    └─ Linear projection           {linear_params:>12,} params  ({format_size(linear_params):>7.2f} MB)")

print()

# Детальный анализ Attention Decoder
print("=" * 80)
print("ДЕТАЛИ ATTENTION DECODER")
print("=" * 80)

attn_cell_params = count_parameters(model.attn.attention_cell)
generator_params = count_parameters(model.attn.generator)

print(f"  Attention Cell                     {attn_cell_params:>12,} params  ({format_size(attn_cell_params):>7.2f} MB)")
print(f"  Generator (Linear)                 {generator_params:>12,} params  ({format_size(generator_params):>7.2f} MB)")

# Детали Attention Cell
print("\n  Детали Attention Cell:")
for name, module in model.attn.attention_cell.named_children():
    params = count_parameters(module)
    mb = format_size(params)
    print(f"    {name:<36} {params:>12,} params  ({mb:>7.2f} MB)")

print()

# Анализ влияния параметров
print("=" * 80)
print("ВЛИЯНИЕ АРХИТЕКТУРНЫХ ПАРАМЕТРОВ НА РАЗМЕР")
print("=" * 80)

configs = [
    ("Микро (1×128, CNN=256)", 1, 128, 256),
    ("Легкая (1×256, CNN=512)", 1, 256, 512),
    ("Стандарт (2×256, CNN=512)", 2, 256, 512),
    ("Тяжелая (2×512, CNN=512)", 2, 512, 512),
    ("Мощная CNN (2×256, CNN=768)", 2, 256, 768),
    ("Глубокая (4×256, CNN=512)", 4, 256, 512),
]

print(f"{'Конфигурация':<30} {'Params':>12} {'Size (MB)':>12} {'vs Baseline':>12}")
print("-" * 80)

baseline_params = None
for name, num_enc, hidden, cnn_out in configs:
    m = TRBAModel(
        num_classes=num_classes,
        hidden_size=hidden,
        num_encoder_layers=num_enc,
        img_h=64,
        img_w=256,
        cnn_in_channels=3,
        cnn_out_channels=cnn_out,
        sos_id=stoi["<SOS>"],
        eos_id=stoi["<EOS>"],
        pad_id=stoi["<PAD>"],
        blank_id=stoi.get("<BLANK>", None),
    )
    
    params = count_parameters(m)
    mb = format_size(params)
    
    if baseline_params is None:
        baseline_params = params
        ratio = "baseline"
    else:
        ratio = f"{params/baseline_params:.2f}×"
    
    print(f"{name:<30} {params:>12,} {mb:>11.2f} MB {ratio:>12}")

print()

# Что можно сократить
print("=" * 80)
print("🔧 РЕКОМЕНДАЦИИ ПО УМЕНЬШЕНИЮ РАЗМЕРА")
print("=" * 80)

print("""
1. CNN (SEResNet31) — самая тяжелая часть (~60-70% модели)
   - Уменьшить cnn_out_channels: 512 → 384 или 256 (-25-50%)
   - Использовать более легкий backbone (MobileNet, EfficientNet)
   - Применить квантизацию (INT8 вместо FP32) → ÷4 размер

2. Encoder RNN (BiLSTM) — 15-25% модели
   - Уменьшить hidden_size: 256 → 128 (-50% параметров)
   - Уменьшить num_encoder_layers: 2 → 1 (-50% параметров)
   - BiLSTM заменить на обычный LSTM (bidirectional=False) → ÷2

3. Attention Decoder — 10-20% модели
   - Уменьшить hidden_size (влияет на attention_cell и generator)
   - Упростить attention mechanism (например, dot-product вместо MLP)

4. Общие методы
   - Квантизация (FP32 → INT8): ÷4 размер, небольшая потеря точности
   - Pruning: удаление неважных весов (5-20% сокращение)
   - Knowledge Distillation: обучение маленькой модели на большой

5. Оптимальная конфигурация для CPU (легкая, но эффективная):
   - num_encoder_layers=1
   - hidden_size=128
   - cnn_out_channels=256
   - Квантизация INT8
   Итого: ~5-10 MB вместо 40 MB
""")

print("=" * 80)
