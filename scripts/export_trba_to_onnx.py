"""
Скрипт для экспорта TRBA модели в ONNX формат.

Поддерживает два режима использования:
1. CLI с парсингом аргументов (для запуска из командной строки)
2. Прямой вызов TRBA.export_to_onnx с переменными (для использования в коде)
"""

import argparse
import sys
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from manuscript.recognizers import TRBA


def main():
    """Точка входа для CLI."""
    parser = argparse.ArgumentParser(
        description="Export TRBA model to ONNX format",
        epilog="Example: python export_trba_to_onnx.py --weights best_model.pth --config config.json --output model.onnx"
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to TRBA weights (.pth file)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config.json",
    )
    parser.add_argument(
        "--charset",
        type=str,
        required=True,
        help="Path to charset.txt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="trba.onnx",
        help="Output path for ONNX model (default: trba.onnx)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version (default: 14, recommended 14+)",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Disable ONNX graph simplification",
    )
    
    args = parser.parse_args()
    
    # Экспортируем используя статический метод TRBA
    TRBA.export_to_onnx(
        weights_path=args.weights,
        config_path=args.config,
        charset_path=args.charset,
        output_path=args.output,
        opset_version=args.opset,
        simplify=not args.no_simplify,
    )
    
    print(f"\nNext steps:")
    print(f"  1. Test ONNX model: recognizer = TRBA(model_path='{args.output}')")
    print(f"  2. Benchmark inference speed: ONNX vs PyTorch")
    print(f"  3. Deploy with onnxruntime for production")


# ============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ БЕЗ ПАРСИНГА АРГУМЕНТОВ
# ============================================================================

def example_usage():
    """
    Пример прямого использования TRBA.export_to_onnx с переменными.
    
    Раскомментируйте нужный блок и запустите скрипт напрямую.
    """
    
    # ========== Пример 1: Базовый экспорт ==========
    weights_path = r"C:\Users\USER\Desktop\trba_exp_1_64\trba_exp_1_64.pth"
    config_path = r"C:\Users\USER\Desktop\trba_exp_1_64\trba_exp_1_64.json"
    charset_path = r"C:\Users\USER\Desktop\trba_exp_1_64\charset.txt"  # charset из папки эксперимента
    output_path = r"C:\Users\USER\Desktop\trba_exp_1_64\trba_exp_1_64.onnx"
    
    TRBA.export_to_onnx(
        weights_path=weights_path,
        config_path=config_path,
        charset_path=charset_path,
        output_path=output_path,
    )
    
    # ========== Пример 2: С кастомными параметрами ==========
    # weights_path = r"C:\Users\USER\Desktop\trba_exp_1_64\trba_exp_1_64.pth"
    # config_path = r"C:\Users\USER\Desktop\trba_exp_1_64\trba_exp_1_64.json"
    # charset_path = r"C:\Users\USER\Desktop\trba_exp_1_64\charset.txt"
    # output_path = r"trba_exp_1_64.onnx"
    # 
    # TRBA.export_to_onnx(
    #     weights_path=weights_path,
    #     config_path=config_path,
    #     charset_path=charset_path,
    #     output_path=output_path,
    #     opset_version=16,
    #     simplify=False,
    # )


if __name__ == "__main__":
    # Для использования через командную строку (с аргументами)
    # Пример: python export_trba_to_onnx.py --weights model.pth --output model.onnx
    # main()
    
    # Для прямого использования (с переменными)
    # Раскомментируйте следующую строку и закомментируйте main() выше:
    example_usage()
