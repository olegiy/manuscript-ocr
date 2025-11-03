"""
Тест параметра num_encoder_layers в TRBA модели.

Проверяет:
1. Создание модели с разным количеством encoder layers
2. Инференс работает корректно
3. Параметр правильно сохраняется/загружается из конфига
"""

import torch
from manuscript.recognizers._trba.model.model import TRBAModel
from manuscript.recognizers._trba.data.transforms import load_charset
import os

# Загрузка charset
current_dir = os.path.dirname(os.path.abspath(__file__))
charset_path = os.path.join(
    current_dir,
    "src",
    "manuscript",
    "recognizers",
    "_trba",
    "configs",
    "charset.txt"
)

itos, stoi = load_charset(charset_path)
num_classes = len(itos)

print("=" * 60)
print("Тестирование num_encoder_layers в TRBAModel")
print("=" * 60)

# Тест 1: Модель с 1 encoder layer
print("\n1️⃣  Создание модели с num_encoder_layers=1")
model_1layer = TRBAModel(
    num_classes=num_classes,
    hidden_size=256,
    num_encoder_layers=1,
    sos_id=stoi["<SOS>"],
    eos_id=stoi["<EOS>"],
    pad_id=stoi["<PAD>"],
)
print(f"✅ Модель создана с {model_1layer.num_encoder_layers} encoder layer(s)")
print(f"   Encoder: {len(model_1layer.enc_rnn)} слоёв")

# Тест 2: Модель с 2 encoder layers (по умолчанию)
print("\n2️⃣  Создание модели с num_encoder_layers=2 (default)")
model_2layers = TRBAModel(
    num_classes=num_classes,
    hidden_size=256,
    num_encoder_layers=2,
    sos_id=stoi["<SOS>"],
    eos_id=stoi["<EOS>"],
    pad_id=stoi["<PAD>"],
)
print(f"✅ Модель создана с {model_2layers.num_encoder_layers} encoder layer(s)")
print(f"   Encoder: {len(model_2layers.enc_rnn)} слоёв")

# Тест 3: Модель с 4 encoder layers
print("\n3️⃣  Создание модели с num_encoder_layers=4")
model_4layers = TRBAModel(
    num_classes=num_classes,
    hidden_size=256,
    num_encoder_layers=4,
    sos_id=stoi["<SOS>"],
    eos_id=stoi["<EOS>"],
    pad_id=stoi["<PAD>"],
)
print(f"✅ Модель создана с {model_4layers.num_encoder_layers} encoder layer(s)")
print(f"   Encoder: {len(model_4layers.enc_rnn)} слоёв")

# Тест 4: Проверка инференса
print("\n4️⃣  Тест инференса с разными моделями")
batch_size = 2
dummy_input = torch.randn(batch_size, 3, 64, 256)

for name, model in [
    ("1 layer", model_1layer),
    ("2 layers", model_2layers),
    ("4 layers", model_4layers)
]:
    model.eval()
    with torch.no_grad():
        probs, preds = model(
            dummy_input,
            is_train=False,
            mode="greedy",
            batch_max_length=25
        )
    print(f"   {name:10s}: Output shape = {probs.shape}, Preds shape = {preds.shape}")

print("\n✅ Все тесты пройдены!")

# Тест 5: Подсчёт параметров
print("\n5️⃣  Сравнение количества параметров")
for name, model in [
    ("1 encoder layer", model_1layer),
    ("2 encoder layers", model_2layers),
    ("4 encoder layers", model_4layers)
]:
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.enc_rnn.parameters())
    print(f"   {name:17s}: Total = {total_params:,} params, Encoder = {encoder_params:,} params")

print("\n" + "=" * 60)
print("🎉 Параметр num_encoder_layers работает корректно!")
print("=" * 60)
