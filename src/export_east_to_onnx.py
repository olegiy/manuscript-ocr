"""
Скрипт для экспорта модели EAST в формат ONNX.

Usage:
    python scripts/export_east_to_onnx.py
"""

from manuscript.detectors import EAST


EAST.export_to_onnx(
    weights_path=r"C:\Users\USER\Desktop\OCR_MODELS\best.pth",
    output_path="east_model_101.onnx",
    backbone_name="resnet101",
    input_size=1024,
)
