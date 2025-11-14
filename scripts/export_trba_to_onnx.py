"""
Скрипт для экспорта модели TRBA V2 в формат ONNX.

Model V2 уже ONNX-compatible из коробки!
Поддерживает два режима:
1. CTC - быстрый декодер
2. Attention (greedy only) - точный декодер

Usage:
    # CTC экспорт
    python scripts/export_trba_to_onnx.py --mode ctc --output trba_ctc.onnx
    
    # Attention экспорт  
    python scripts/export_trba_to_onnx.py --mode attention --output trba_attention.onnx
    
    # С кастомными весами
    python scripts/export_trba_to_onnx.py --weights path/to/weights.pth --config path/to/config.json
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from manuscript.recognizers import TRBA


class TRBACTCWrapper(nn.Module):
    """Wrapper для CTC экспорта."""
    
    def __init__(self, trba_recognizer):
        super().__init__()
        self.model = trba_recognizer.model
        
    def forward(self, x):
        return self.model.forward_ctc(x)


class TRBAAttentionWrapper(nn.Module):
    """Wrapper для Attention экспорта с фиксированными параметрами."""
    
    def __init__(self, trba_recognizer, max_length=40):
        super().__init__()
        self.model = trba_recognizer.model
        self.max_length = max_length
        
    def forward(self, x):
        logits, _ = self.model.forward_attention(
            x,
            is_train=False,
            batch_max_length=self.max_length,
            onnx_mode=True  # Всегда max_length шагов
        )
        return logits


def export_trba_to_onnx(
    weights_path: str = None,
    config_path: str = None,
    output_path: str = "trba.onnx",
    mode: str = "attention",
    max_length: int = None,
    img_h: int = 64,
    img_w: int = 256,
    opset_version: int = 14,
    simplify: bool = True,
):
    """
    Экспортирует модель TRBA V2 в формат ONNX.

    Parameters
    ----------
    weights_path : str, optional
        Путь к весам модели PyTorch (.pth файл)
    config_path : str, optional
        Путь к config.json для автоматического определения max_length
    output_path : str
        Путь для сохранения ONNX модели
    mode : str
        Режим экспорта: "ctc" или "attention"
    max_length : int, optional
        Максимальная длина для attention режима
    img_h : int
        Высота входного изображения
    img_w : int
        Ширина входного изображения
    opset_version : int
        Версия ONNX opset (рекомендуется 14+)
    simplify : bool
        Использовать onnx-simplifier для оптимизации графа
    """
    # Загружаем config если указан
    if config_path is not None and max_length is None:
        import json
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            max_length = config.get('max_len', 40)
            print(f"Loaded max_length={max_length} from config: {config_path}")
    
    # Используем дефолтное значение если не указано
    if max_length is None:
        max_length = 40
    
    print(f"Loading TRBA V2 model...")
    print(f"Export mode: {mode.upper()}")
    
    # Загружаем модель на CPU (для ONNX)
    if weights_path is not None:
        recognizer = TRBA(model_path=weights_path, device='cpu')
    else:
        recognizer = TRBA(device='cpu')
    
    recognizer.model.eval()
    
    print(f"\nModel loaded successfully")
    print(f"  - Hidden size: {recognizer.hidden_size}")
    print(f"  - Num classes: {len(recognizer.itos)}")
    print(f"  - Encoder type: {recognizer.encoder_type}")
    print(f"  - Decoder type: {recognizer.decoder_type}")
    if mode == "attention":
        print(f"  - Max length: {max_length}")
    print(f"  - Device: CPU (required for ONNX export)")
    
    # Создаём wrapper в зависимости от режима
    print(f"\nCreating ONNX wrapper for {mode} mode...")
    
    if mode == "ctc":
        onnx_model = TRBACTCWrapper(recognizer)
        output_names = ["ctc_logits"]
    elif mode == "attention":
        onnx_model = TRBAAttentionWrapper(recognizer, max_length=max_length)
        output_names = ["attention_logits"]
    else:
        raise ValueError(f"mode должен быть 'ctc' или 'attention', получен: {mode}")
    
    onnx_model.eval()
    
    # Создаём dummy input
    dummy_input = torch.randn(1, 3, img_h, img_w)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Проверяем что модель работает
    print(f"\nTesting model before export...")
    with torch.no_grad():
        output = onnx_model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output logits range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Декодируем для проверки
    if mode == "attention":
        preds = output.argmax(dim=-1)[0]  # [max_length]
        print(f"Predicted token IDs (first 10): {preds[:10].tolist()}")
    
    # Экспортируем в ONNX
    print(f"\nExporting to ONNX (opset {opset_version})...")
    
    torch.onnx.export(
        onnx_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=output_names,
        dynamic_axes={
            "input": {0: "batch_size"},
            output_names[0]: {0: "batch_size"},
        },
        verbose=False,
    )
    
    print(f"[OK] ONNX model saved to: {output_path}")
    
    # Проверяем ONNX модель
    print("\nVerifying ONNX model...")
    try:
        import onnx
        
        onnx_model_proto = onnx.load(output_path)
        onnx.checker.check_model(onnx_model_proto)
        print("[OK] ONNX model is valid")
        
        # Опционально: упрощаем граф
        if simplify:
            try:
                import onnxsim
                
                print("\nSimplifying ONNX model...")
                model_simplified, check = onnxsim.simplify(onnx_model_proto)
                
                if check:
                    onnx.save(model_simplified, output_path)
                    print("[OK] ONNX model simplified and saved")
                else:
                    print("[WARNING] Simplification check failed, keeping original model")
            except ImportError:
                print("[WARNING] onnx-simplifier not installed, skipping simplification")
                print("  Install with: pip install onnx-simplifier")
        
        # Тестируем ONNX inference
        print(f"\nTesting ONNX inference...")
        try:
            import onnxruntime as ort
            
            session = ort.InferenceSession(output_path)
            
            # Тестовый запуск
            ort_inputs = {"input": dummy_input.numpy()}
            ort_outputs = session.run(None, ort_inputs)
            
            print(f"[OK] ONNX inference works!")
            print(f"  Output shape: {ort_outputs[0].shape}")
            
            # Сравниваем с PyTorch
            torch_output = output.numpy()
            onnx_output = ort_outputs[0]
            
            max_diff = abs(torch_output - onnx_output).max()
            print(f"  Max difference vs PyTorch: {max_diff:.6f}")
            
            if max_diff < 1e-4:
                print(f"  [OK] Outputs match!")
            else:
                print(f"  [WARNING] Outputs differ significantly")
                
        except ImportError:
            print("[WARNING] onnxruntime not installed, skipping inference test")
            print("  Install with: pip install onnxruntime")
        
        # Выводим информацию о модели
        print(f"\n=== ONNX Model Info ===")
        print(f"File size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")
        print(f"Opset version: {opset_version}")
        print(f"Inputs: {[inp.name for inp in onnx_model_proto.graph.input]}")
        print(f"Outputs: {[out.name for out in onnx_model_proto.graph.output]}")
        print(f"\nInput shape: [batch_size, 3, {img_h}, {img_w}]")
        if mode == "ctc":
            print(f"Output shape: [batch_size, W, {len(recognizer.itos)}] (W зависит от ширины)")
        else:
            print(f"Output shape: [batch_size, {max_length}, {len(recognizer.itos)}]")
            
    except ImportError:
        print("[WARNING] onnx package not installed")
        print("  Install with: pip install onnx")


def main():
    parser = argparse.ArgumentParser(
        description="Export TRBA V2 model to ONNX format"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to TRBA weights (.pth file). If not specified, uses default weights",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.json to automatically load max_length",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for ONNX model (default: trba_{mode}.onnx)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="attention",
        choices=["ctc", "attention"],
        help="Export mode: 'ctc' (fast) or 'attention' (accurate, greedy only). Default: attention",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum decoding length for attention mode (default: from config or 40)",
    )
    parser.add_argument(
        "--img-h",
        type=int,
        default=64,
        help="Input image height (default: 64)",
    )
    parser.add_argument(
        "--img-w",
        type=int,
        default=256,
        help="Input image width (default: 256)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Disable ONNX graph simplification",
    )
    
    args = parser.parse_args()
    
    # Определяем дефолтное имя файла если не указано
    if args.output is None:
        args.output = f"trba_{args.mode}.onnx"
    
    # Экспортируем
    export_trba_to_onnx(
        weights_path=args.weights,
        config_path=args.config,
        output_path=args.output,
        mode=args.mode,
        max_length=args.max_length,
        img_h=args.img_h,
        img_w=args.img_w,
        opset_version=args.opset,
        simplify=not args.no_simplify,
    )
    
    print(f"\n[OK] Export completed successfully!")
    print(f"\nNext steps:")
    print(f"  1. Test ONNX model with real images")
    print(f"  2. Benchmark inference speed: ONNX vs PyTorch")
    print(f"  3. Deploy with onnxruntime for production")
    print(f"\nExported model: {args.output}")
    print(f"Mode: {args.mode.upper()}")
    if args.mode == "attention":
        max_len = args.max_length if args.max_length else "from config or 40"
        print(f"Max length: {max_len}")
    print(f"\nNote:")
    print(f"  - Model V2 is ONNX-compatible out of the box!")
    print(f"  - Beam search removed (only greedy for attention)")
    print(f"  - Use --mode ctc for fast inference")
    print(f"  - Use --mode attention for better accuracy")


if __name__ == "__main__":
    main()

