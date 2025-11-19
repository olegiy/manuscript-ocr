"""
Скрипт для сравнения предсказаний PyTorch и ONNX моделей TRBA.
Помогает определить, есть ли проблемы при конвертации в ONNX.
"""
import os
import subprocess
import sys

print("="*80)
print("🔍 СРАВНЕНИЕ PyTorch и ONNX моделей TRBA")
print("="*80)

# Запускаем PyTorch валидацию
print("\n" + "="*80)
print("1️⃣ Запуск PyTorch валидации...")
print("="*80)
pytorch_result = subprocess.run(
    [sys.executable, "src/trba_metrics_pytorch.py"],
    capture_output=False,
    text=True
)

if pytorch_result.returncode != 0:
    print("❌ Ошибка при запуске PyTorch валидации!")
    sys.exit(1)

print("\n" + "="*80)
print("2️⃣ Запуск ONNX валидации...")
print("="*80)
onnx_result = subprocess.run(
    [sys.executable, "src/trba_metrics.py"],
    capture_output=False,
    text=True
)

if onnx_result.returncode != 0:
    print("❌ Ошибка при запуске ONNX валидации!")
    sys.exit(1)

print("\n" + "="*80)
print("✅ Сравнение завершено!")
print("="*80)
print("\n📋 Для анализа различий:")
print("   1. Сравните метрики OVERALL в обоих выводах")
print("   2. Сравните худшие примеры - если они разные, проблема в ONNX")
print("   3. Если метрики PyTorch совпадают с обучением, а ONNX - нет,")
print("      то проблема в конвертации → нужно переэкспортировать модель")
