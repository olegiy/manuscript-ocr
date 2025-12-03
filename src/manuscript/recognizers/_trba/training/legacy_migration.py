"""
Утилита для миграции весов со старой архитектуры TRBA на новую.

Старая архитектура (без CTC head):
- cnn (SEResNet31)
- pool
- enc_rnn (Sequential of BidirectionalLSTM)
- enc_dropout
- attn (Attention decoder)

Новая архитектура (с опциональными CTC/Attention heads):
- cnn (SEResNet31/SEVGG/etc)
- enc_rnn (Sequential of BiRNN)
- enc_dropout
- attention_decoder (Attention decoder с GRU/LSTM вариантами)
- ctc_head (опционально)
"""

import torch
from typing import Dict, Any


def build_legacy_key_mapping() -> Dict[str, str]:
    """
    Маппинг ключей со старой архитектуры на новую.
    
    Returns:
        Dict[old_key, new_key]
    """
    mapping = {}
    
    # CNN backbone - без изменений
    # cnn.* -> cnn.*
    
    # Encoder RNN - без изменений
    # enc_rnn.* -> enc_rnn.*
    # enc_dropout.* -> enc_dropout.*
    
    # Attention decoder - переименование
    # OLD: attn.attention_cell.*
    # NEW: attention_decoder.attention_cell.*
    mapping_patterns = [
        ("attn.attention_cell.", "attention_decoder.attention_cell."),
        ("attn.hidden_size", "attention_decoder.hidden_size"),
        ("attn.num_classes", "attention_decoder.num_classes"),
        ("attn.generator.", "attention_decoder.generator."),
    ]
    
    return mapping_patterns


def migrate_legacy_weights(
    old_state_dict: Dict[str, torch.Tensor],
    use_ctc_head: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Конвертирует веса со старой архитектуры на новую.
    
    Args:
        old_state_dict: Старый state_dict
        use_ctc_head: В старых весах CTC не было
        
    Returns:
        Новый state_dict совместимый с новой архитектурой
    """
    new_state_dict = {}
    mapping_patterns = build_legacy_key_mapping()
    
    for old_key, tensor in old_state_dict.items():
        new_key = old_key
        
        # Применяем маппинг
        for old_pattern, new_pattern in mapping_patterns:
            if old_key.startswith(old_pattern):
                new_key = old_key.replace(old_pattern, new_pattern, 1)
                break
            
        new_state_dict[new_key] = tensor
    
    return new_state_dict


def load_legacy_trba_weights(
    model,
    checkpoint_path: str,
    strict: bool = False,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Загружает старые TRBA веса в новую модель.
    
    Args:
        model: Новая TRBAModel
        checkpoint_path: Путь к старому чекпоинту
        strict: Строгая загрузка (должны совпасть все ключи)
        device: Устройство для загрузки
        
    Returns:
        Статистика загрузки
        
    Example:
        >>> model = TRBAModel(...)
        >>> stats = load_legacy_trba_weights(model, "old_trba.pth")
        >>> print(f"Loaded {stats['loaded']}/{stats['total']} keys")
    """
    # Загружаем старый чекпоинт
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Извлекаем state_dict
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            old_state = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            old_state = checkpoint["model"]
        elif "state_dict" in checkpoint:
            old_state = checkpoint["state_dict"]
        else:
            old_state = checkpoint
    else:
        old_state = checkpoint
    
    # Мигрируем ключи
    new_state = migrate_legacy_weights(
        old_state,
        use_ctc_head=model.use_ctc_head,
    )
    
    # Загружаем в модель
    missing_keys, unexpected_keys = model.load_state_dict(new_state, strict=strict)
    
    return {
        "loaded": len(new_state),
        "total": len(old_state),
        "missing": missing_keys,
        "unexpected": unexpected_keys,
    }
