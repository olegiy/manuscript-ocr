"""
Скрипт для экспорта модели EAST в формат ONNX.

Usage:
    python scripts/export_east_to_onnx.py
"""

from manuscript.detectors import EAST


EAST.export_to_onnx(
    weights_path=r"C:\Users\USER\.manuscript\east\east_quad_23_05.pth",
    output_path="east_model.onnx",
)
