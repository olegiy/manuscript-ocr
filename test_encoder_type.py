"""Тест для проверки параметра encoder_type в TRBA."""

import torch
from src.manuscript.recognizers._trba.model.model import TRBAModel

# Параметры для теста
num_classes = 100
hidden_size = 256
num_encoder_layers = 2
img_h = 64
img_w = 256
batch_size = 2

print("=" * 60)
print("Тестирование encoder_type в TRBA")
print("=" * 60)

# Тест 1: LSTM encoder (по умолчанию)
print("\n1. Тестирование LSTM encoder...")
model_lstm = TRBAModel(
    num_classes=num_classes,
    hidden_size=hidden_size,
    num_encoder_layers=num_encoder_layers,
    encoder_type="LSTM",
    img_h=img_h,
    img_w=img_w,
)

# Подсчет параметров LSTM
lstm_params = sum(p.numel() for p in model_lstm.parameters())
print(f"   Количество параметров (LSTM): {lstm_params:,}")

# Тест forward pass
x = torch.randn(batch_size, 3, img_h, img_w)
tgt = torch.randint(0, num_classes, (batch_size, 20))
try:
    output = model_lstm(x, tgt)
    print(f"   Output shape: {output.shape}")
    print("   ✓ LSTM forward pass успешен")
except Exception as e:
    print(f"   ✗ Ошибка LSTM: {e}")

# Тест 2: GRU encoder
print("\n2. Тестирование GRU encoder...")
model_gru = TRBAModel(
    num_classes=num_classes,
    hidden_size=hidden_size,
    num_encoder_layers=num_encoder_layers,
    encoder_type="GRU",
    img_h=img_h,
    img_w=img_w,
)

# Подсчет параметров GRU
gru_params = sum(p.numel() for p in model_gru.parameters())
print(f"   Количество параметров (GRU): {gru_params:,}")

# Тест forward pass
try:
    output = model_gru(x, tgt)
    print(f"   Output shape: {output.shape}")
    print("   ✓ GRU forward pass успешен")
except Exception as e:
    print(f"   ✗ Ошибка GRU: {e}")

# Сравнение
print("\n3. Сравнение моделей:")
print(f"   Параметры LSTM: {lstm_params:,}")
print(f"   Параметры GRU:  {gru_params:,}")
diff = lstm_params - gru_params
diff_pct = (diff / lstm_params) * 100
print(f"   Разница: {diff:,} параметров ({diff_pct:.1f}%)")
print(f"   GRU экономит ~{diff_pct:.0f}% параметров по сравнению с LSTM")

# Тест 3: Проверка загрузки из конфига
print("\n4. Тестирование инициализации через TRBA класс...")
from src.manuscript.recognizers import TRBA

try:
    # Тест с LSTM
    trba_lstm = TRBA(encoder_type="LSTM")
    print("   ✓ TRBA с encoder_type='LSTM' инициализирован")

    # Тест с GRU
    trba_gru = TRBA(encoder_type="GRU")
    print("   ✓ TRBA с encoder_type='GRU' инициализирован")

except Exception as e:
    print(f"   ✗ Ошибка инициализации TRBA: {e}")

print("\n" + "=" * 60)
print("Все тесты завершены!")
print("=" * 60)
