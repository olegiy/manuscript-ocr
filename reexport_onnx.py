"""
Скрипт для переэкспорта ONNX модели с исправлением max_length.
Использует исправленный код с max_length + 1 шагами.
"""

from manuscript.recognizers import TRBA
from pathlib import Path

# ============================================
# НАСТРОЙКИ - УКАЖИТЕ ВАШИ ПУТИ
# ============================================

# Модель для переэкспорта
MODEL_NAME = "trba_exp_lite"

WEIGHTS_PATH = rf"C:\Users\USER\Desktop\{MODEL_NAME}\best_acc_weights.pth"
CONFIG_PATH = rf"C:\Users\USER\Desktop\{MODEL_NAME}\config.json"
CHARSET_PATH = rf"C:\Users\USER\Desktop\{MODEL_NAME}\charset.txt"
OUTPUT_PATH = rf"C:\Users\USER\Desktop\{MODEL_NAME}\{MODEL_NAME}_FIXED.onnx"

# ============================================
# ПРОВЕРКА ФАЙЛОВ
# ============================================

print("=" * 80)
print("🔧 ПЕРЕЭКСПОРТ ONNX МОДЕЛИ С ИСПРАВЛЕНИЕМ")
print("=" * 80)
print()

# Проверяем что все файлы существуют
for path, name in [(WEIGHTS_PATH, "PTH weights"), (CONFIG_PATH, "Config"), (CHARSET_PATH, "Charset")]:
    if not Path(path).exists():
        print(f"❌ Ошибка: {name} не найден: {path}")
        exit(1)
    else:
        print(f"✅ {name}: {path}")

print()

# ============================================
# ЭКСПОРТ
# ============================================

print("🚀 Начинаем экспорт...")
print(f"   Выходной файл: {OUTPUT_PATH}")
print()

try:
    TRBA.export_to_onnx(
        weights_path=WEIGHTS_PATH,
        config_path=CONFIG_PATH,
        charset_path=CHARSET_PATH,
        output_path=OUTPUT_PATH,
        opset_version=14,
        simplify=True
    )
    
    print()
    print("=" * 80)
    print("✅ ЭКСПОРТ ЗАВЕРШЕН УСПЕШНО!")
    print("=" * 80)
    print(f"📄 Новая модель сохранена: {OUTPUT_PATH}")
    print()
    print("⚠️  ВАЖНО:")
    print("   1. Старая ONNX модель теперь несовместима")
    print("   2. Используйте новую модель для инференса")
    print("   3. Проверьте результаты с помощью debug_pth_vs_onnx.py")
    print()
    
except Exception as e:
    print()
    print("=" * 80)
    print("❌ ОШИБКА ПРИ ЭКСПОРТЕ!")
    print("=" * 80)
    print(f"Сообщение: {e}")
    print()
    import traceback
    traceback.print_exc()
