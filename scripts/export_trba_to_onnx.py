"""
Скрипт для экспорта модели TRBA в формат ONNX.

Usage:
    python scripts/export_trba_to_onnx.py
"""

from manuscript.recognizers import TRBA


TRBA.export_to_onnx(
    weights_path=r"C:\Users\USER\Desktop\trba_exp_lite\best_acc_weights.pth",
    config_path=r"C:\Users\USER\Desktop\trba_exp_lite\config.json",
    charset_path=r"C:\Users\USER\Desktop\trba_exp_lite\charset.txt",
    output_path=r"C:\Users\USER\Desktop\trba_exp_lite\trba_exp_lite.onnx",
)
