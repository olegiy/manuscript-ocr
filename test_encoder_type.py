"""Тест для проверки параметров encoder_type и decoder_type в TRBA."""

import torch
from src.manuscript.recognizers._trba.model.model import TRBAModel

# Параметры для теста
num_classes = 100
hidden_size = 256
num_encoder_layers = 2
img_h = 64
img_w = 256
batch_size = 2

print("=" * 80)
print("Тестирование encoder_type и decoder_type в TRBA")
print("=" * 80)

# Матрица тестов: все комбинации encoder/decoder
test_configs = [
    ("LSTM", "LSTM", "Encoder: LSTM, Decoder: LSTM"),
    ("LSTM", "GRU", "Encoder: LSTM, Decoder: GRU"),
    ("GRU", "LSTM", "Encoder: GRU, Decoder: LSTM"),
    ("GRU", "GRU", "Encoder: GRU, Decoder: GRU"),
]

results = []

for encoder_type, decoder_type, desc in test_configs:
    print(f"\n{'=' * 80}")
    print(f"Тест: {desc}")
    print("=" * 80)

    try:
        # Создаем модель
        model = TRBAModel(
            num_classes=num_classes,
            hidden_size=hidden_size,
            num_encoder_layers=num_encoder_layers,
            encoder_type=encoder_type,
            decoder_type=decoder_type,
            img_h=img_h,
            img_w=img_w,
        )

        # Подсчет параметров
        params = sum(p.numel() for p in model.parameters())
        print(f"   Количество параметров: {params:,}")

        # Тест forward pass (training)
        x = torch.randn(batch_size, 3, img_h, img_w)
        tgt = torch.randint(0, num_classes, (batch_size, 20))

        model.train()
        output_train = model(model.encode(x), text=tgt, is_train=True)
        print(f"   Training output shape: {output_train.shape}")

        # Тест forward pass (inference greedy)
        model.eval()
        with torch.no_grad():
            probs, preds = model(model.encode(x), is_train=False, mode="greedy")
        print(f"   Greedy inference preds shape: {preds.shape}")

        # Тест forward pass (inference beam)
        with torch.no_grad():
            probs, preds = model(
                model.encode(x), is_train=False, mode="beam", beam_size=3
            )
        print(f"   Beam inference preds shape: {preds.shape}")

        print(f"   ✓ Все тесты пройдены!")
        results.append((desc, params, "✓ OK"))

    except Exception as e:
        print(f"   ✗ Ошибка: {e}")
        import traceback

        traceback.print_exc()
        results.append((desc, 0, f"✗ FAILED: {str(e)[:50]}"))

# Итоговая таблица
print("\n" + "=" * 80)
print("ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("=" * 80)
print(f"{'Конфигурация':<40} {'Параметры':>15} {'Статус':<20}")
print("-" * 80)

for desc, params, status in results:
    params_str = f"{params:,}" if params > 0 else "N/A"
    print(f"{desc:<40} {params_str:>15} {status:<20}")

print("=" * 80)

# Сравнение размеров моделей
if len(results) == 4:
    lstm_lstm = results[0][1]
    gru_gru = results[3][1]

    if lstm_lstm > 0 and gru_gru > 0:
        diff = lstm_lstm - gru_gru
        diff_pct = (diff / lstm_lstm) * 100

        print(f"\n📊 Сравнение LSTM+LSTM vs GRU+GRU:")
        print(f"   LSTM+LSTM: {lstm_lstm:,} параметров")
        print(f"   GRU+GRU:   {gru_gru:,} параметров")
        print(f"   Экономия:  {diff:,} параметров ({diff_pct:.1f}%)")
        print(f"   GRU+GRU экономит ~{diff_pct:.0f}% параметров!")

print("\n" + "=" * 80)
print("Тестирование завершено!")
print("=" * 80)
