"""
Скрипт для конвертации старых TRBA весов в новый формат.

Использование:
    python scripts/convert_old_trba_weights.py

Конвертирует:
    C:\Users\USER\Desktop\OCR_MODELS\exp_2\best_acc_weights.pth
    -> C:\Users\USER\Desktop\OCR_MODELS\exp_2\best_acc_weights_converted.pth
"""

import sys
from pathlib import Path
import torch
import json

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from manuscript.recognizers._trba.model.model import TRBAModel
from manuscript.recognizers._trba.data.transforms import load_charset


def load_old_config(config_path):
    """Загружает конфиг из старой модели."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def convert_weights(old_weights_path, config_path, charset_path, output_path):
    """
    Конвертирует старые веса в новый формат.
    
    Параметры
    ---------
    old_weights_path : str
        Путь к старым весам (.pth)
    config_path : str
        Путь к config.json
    charset_path : str
        Путь к charset.txt
    output_path : str
        Путь для сохранения конвертированных весов
    """
    print(f"Loading config from: {config_path}")
    config = load_old_config(config_path)
    
    print(f"Loading charset from: {charset_path}")
    itos, stoi = load_charset(charset_path)
    num_classes = len(itos)
    
    print(f"\nConfig params:")
    print(f"  num_classes: {num_classes}")
    print(f"  hidden_size: {config.get('hidden_size', 256)}")
    print(f"  img_h: {config.get('img_h', 64)}")
    print(f"  img_w: {config.get('img_w', 256)}")
    
    # Создаем новую модель с параметрами из конфига
    model = TRBAModel(
        num_classes=num_classes,
        hidden_size=config.get('hidden_size', 256),
        num_encoder_layers=config.get('num_encoder_layers', 2),
        img_h=config.get('img_h', 64),
        img_w=config.get('img_w', 256),
        cnn_in_channels=config.get('cnn_in_channels', 3),
        cnn_out_channels=config.get('cnn_out_channels', 512),
        cnn_backbone=config.get('cnn_backbone', 'seresnet31'),
        sos_id=stoi['<SOS>'],
        eos_id=stoi['<EOS>'],
        pad_id=stoi['<PAD>'],
        blank_id=stoi.get('<BLANK>', None),
        enc_dropout_p=0.1,
        use_ctc_head=False,  # Старая модель не имела CTC head
        use_attention_head=True,
    )
    
    print(f"\nLoading old weights from: {old_weights_path}")
    old_state = torch.load(old_weights_path, map_location='cpu')
    
    # Если это полный checkpoint с метаданными
    if isinstance(old_state, dict) and 'model_state_dict' in old_state:
        old_state = old_state['model_state_dict']
    
    print(f"\nOld state_dict keys (first 10):")
    old_keys = list(old_state.keys())[:10]
    for k in old_keys:
        print(f"  {k}")
    
    print(f"\nNew model keys (first 10):")
    new_keys = list(model.state_dict().keys())[:10]
    for k in new_keys:
        print(f"  {k}")
    
    # Попытка загрузить веса напрямую
    print("\nAttempting to load weights...")
    try:
        missing_keys, unexpected_keys = model.load_state_dict(old_state, strict=False)
        
        if missing_keys:
            print(f"\nMissing keys in new model:")
            for k in missing_keys[:20]:  # Показываем первые 20
                print(f"  - {k}")
            if len(missing_keys) > 20:
                print(f"  ... and {len(missing_keys) - 20} more")
        
        if unexpected_keys:
            print(f"\nUnexpected keys from old model:")
            for k in unexpected_keys[:20]:
                print(f"  - {k}")
            if len(unexpected_keys) > 20:
                print(f"  ... and {len(unexpected_keys) - 20} more")
        
        if not missing_keys and not unexpected_keys:
            print("✓ Perfect match! All weights loaded successfully.")
        elif not missing_keys:
            print("✓ All new model weights loaded (some old keys ignored).")
        else:
            print("⚠ Some weights missing - may need manual conversion.")
            print("\nChecking critical components:")
            
            # Проверяем основные компоненты
            cnn_loaded = all(k.startswith('cnn.') for k in missing_keys if k.startswith('cnn.'))
            enc_loaded = all(k.startswith('enc_rnn.') for k in missing_keys if k.startswith('enc_rnn.'))
            attn_loaded = all(k.startswith('attn.') for k in missing_keys if k.startswith('attn.'))
            
            print(f"  CNN encoder: {'✓' if not cnn_loaded else '✗'}")
            print(f"  BiLSTM encoder: {'✓' if not enc_loaded else '✗'}")
            print(f"  Attention decoder: {'✓' if not attn_loaded else '✗'}")
        
        # Сохраняем конвертированные веса
        print(f"\nSaving converted weights to: {output_path}")
        torch.save(model.state_dict(), output_path)
        print("✓ Weights saved successfully!")
        
        # Сохраняем также полный checkpoint для совместимости
        checkpoint_path = output_path.replace('.pth', '_ckpt.pth')
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': config,
            'num_classes': num_classes,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Full checkpoint saved to: {checkpoint_path}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error loading weights: {e}")
        return False


def main():
    # Пути к файлам
    old_weights = r"C:\Users\USER\Desktop\OCR_MODELS\exp_2\best_acc_weights.pth"
    config_path = r"C:\Users\USER\Desktop\OCR_MODELS\exp_2\config.json"
    
    # Используем charset из пакета
    current_dir = Path(__file__).parent.parent / "src" / "manuscript" / "recognizers" / "_trba"
    charset_path = current_dir / "configs" / "charset.txt"
    
    output_path = r"C:\Users\USER\Desktop\OCR_MODELS\exp_2\best_acc_weights_converted.pth"
    
    print("=" * 80)
    print("TRBA Weight Conversion Script")
    print("=" * 80)
    
    # Проверяем что файлы существуют
    if not Path(old_weights).exists():
        print(f"✗ Old weights not found: {old_weights}")
        return
    
    if not Path(config_path).exists():
        print(f"✗ Config not found: {config_path}")
        return
    
    if not Path(charset_path).exists():
        print(f"✗ Charset not found: {charset_path}")
        return
    
    # Конвертируем
    success = convert_weights(old_weights, config_path, charset_path, output_path)
    
    if success:
        print("\n" + "=" * 80)
        print("✓ Conversion completed successfully!")
        print("=" * 80)
        print("\nNext steps:")
        print(f"1. Test with: python scripts/trba_infer_speed_test.py --model-path \"{output_path}\" --config-path \"{config_path}\" --folder <test_folder> --gt-csv <test_csv>")
        print(f"2. Export to ONNX: python scripts/export_trba_to_onnx.py")
    else:
        print("\n" + "=" * 80)
        print("✗ Conversion failed")
        print("=" * 80)


if __name__ == "__main__":
    main()
