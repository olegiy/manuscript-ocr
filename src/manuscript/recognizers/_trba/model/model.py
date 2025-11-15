import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from .seresnet31 import SEResNet31
from .seresnetlite31 import SEResNet31Lite


CNN_BACKBONES = {
    "seresnet31": SEResNet31,
    "resnet31": SEResNet31,
    "seresnet31-lite": SEResNet31Lite,
    "seresnetlite31": SEResNet31Lite,
    "lite": SEResNet31Lite,
}

class TinySelfAttention(nn.Module):
    """
    Миниатюрный self-attention блок для горизонтального контекста.
    Встроен в энкодер, работает параллельно (в отличие от autoregressive attention).
    """

    def __init__(self, dim: int, nhead: int = 2, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # стабильнее при малых батчах
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C]
        """
        out = self.transformer(x)
        return self.norm(out)
# ============================================================================
# Encoder Components
# ============================================================================

class BidirectionalLSTM(nn.Module):
    """BiLSTM encoder layer - ONNX compatible."""
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size, hidden_size, bidirectional=True, batch_first=True
        )
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        h, _ = self.rnn(x)  # [B, T, 2H]
        out = self.linear(h)  # [B, T, D]
        return out

class ConvEncoder1D(nn.Module):
    """
    Лёгкий CNN-энкодер вместо BiLSTM.
    Использует стек 1D-свёрток (Conv1d) для контекста вдоль ширины текста.
    Полностью ONNX-friendly и параллелится на CPU/GPU.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_ch = input_size
        for i in range(num_layers):
            layers += [
                nn.Conv1d(in_ch, hidden_size, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            in_ch = hidden_size
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C] → [B, C, T]
        out: [B, T, C]
        """
        x = x.transpose(1, 2)          # [B, C, T]
        x = self.net(x)                # [B, hidden, T]
        x = x.transpose(1, 2)          # [B, T, hidden]
        return x


# ============================================================================
# Attention Decoder Components
# ============================================================================

class AttentionCell(nn.Module):
    """Attention cell with LSTM - ONNX compatible."""
    
    def __init__(self, input_size, hidden_size, num_embeddings, dropout_p=0.1):
        super().__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p

    def forward(self, prev_hidden, batch_H, char_onehots):
        """
        Args:
            prev_hidden: (h, c) tuple for LSTM
            batch_H: [B, Tenc, C] encoder output
            char_onehots: [B, V] one-hot encoded character
            
        Returns:
            cur_hidden: (h, c) tuple
            alpha: [B, Tenc, 1] attention weights
        """
        # Attention mechanism
        proj_H = self.i2h(batch_H)  # [B, Tenc, H]
        proj_h = self.h2h(prev_hidden[0]).unsqueeze(1)
        e = self.score(torch.tanh(proj_H + proj_h))  # [B, Tenc, 1]

        alpha = F.softmax(e, dim=1)  # [B, Tenc, 1]
        alpha = F.dropout(alpha, p=self.dropout_p, training=self.training)

        # Context vector
        context = torch.bmm(alpha.transpose(1, 2), batch_H).squeeze(1)  # [B, C]
        
        # Concatenate context and character embedding
        x = torch.cat([context, char_onehots], 1)  # [B, C + V]
        
        # Decoder step
        cur_hidden = self.rnn(x, prev_hidden)  # (h, c)
        return cur_hidden, alpha


class AttentionDecoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        sos_id: int,
        eos_id: int,
        pad_id: int,
        blank_id: Optional[int] = None,
        dropout_p: float = 0.1,
        sampling_prob: float = 0.0,
    ):
        super().__init__()

        self.attention_cell = AttentionCell(
            input_size, hidden_size, num_classes, dropout_p=dropout_p
        )
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.blank_id = blank_id

        self.generator = nn.Linear(hidden_size, num_classes)
        self.dropout_p = dropout_p
        self.sampling_prob = sampling_prob

    def _char_to_onehot(self, input_char: torch.Tensor) -> torch.Tensor:
        return F.one_hot(input_char, num_classes=self.num_classes).float()

    def _mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Маскирование blank токена если он есть."""
        if self.blank_id is not None:
            if logits.dim() == 3:
                logits[:, :, self.blank_id] = -1e4
            else:
                logits[:, self.blank_id] = -1e4
        return logits

    @torch.no_grad()
    def greedy_decode(
        self, 
        batch_H: torch.Tensor, 
        batch_max_length: int = 25,
        onnx_mode: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Greedy декодирование.
        
        Args:
            batch_H: [B, W, hidden_size] encoder output
            batch_max_length: максимальная длина последовательности
            onnx_mode: если True, всегда декодирует max_length шагов (для ONNX)
                      если False, останавливается при EOS (для PyTorch)
        
        Returns:
            probs: [B, T, num_classes] логиты
            preds: [B, T] предсказанные токены
        """
        B = batch_H.size(0)
        device = batch_H.device

        # Инициализация скрытых состояний
        h = torch.zeros(B, self.hidden_size, device=device)
        c = torch.zeros(B, self.hidden_size, device=device)
        hidden = (h, c)

        # Начинаем с SOS токена
        targets = torch.full((B,), self.sos_id, dtype=torch.long, device=device)

        all_probs = []
        all_preds = []

        # Декодирование
        max_steps = batch_max_length if onnx_mode else (batch_max_length + 1)
        
        for t in range(max_steps):
            # One-hot encoding (ONNX-friendly)
            onehots = self._char_to_onehot(targets)
            
            # Attention step
            hidden, _ = self.attention_cell(hidden, batch_H, onehots)
            h_out = hidden[0]

            # Generate logits
            out = F.dropout(h_out, p=self.dropout_p, training=self.training)
            logits_t = self.generator(out)
            logits_t = self._mask_logits(logits_t)

            all_probs.append(logits_t.unsqueeze(1))

            # Greedy choice
            next_tokens = logits_t.argmax(1)
            all_preds.append(next_tokens.unsqueeze(1))

            targets = next_tokens.clone()

            # Early stopping (только для PyTorch режима)
            if not onnx_mode and (next_tokens == self.eos_id).all():
                break

        probs = torch.cat(all_probs, dim=1)  # [B, T, V]
        preds = torch.cat(all_preds, dim=1)  # [B, T]
        
        return probs, preds

    def forward_training(
        self,
        batch_H: torch.Tensor,
        text: torch.Tensor,
        batch_max_length: int = 25
    ) -> torch.Tensor:
        """
        Training forward pass с teacher forcing.
        
        Args:
            batch_H: [B, W, hidden_size] encoder output
            text: [B, T] target tokens (с SOS токеном в начале)
            batch_max_length: максимальная длина
            
        Returns:
            logits: [B, T, num_classes]
        """
        device = batch_H.device
        B = batch_H.size(0)
        steps = batch_max_length + 1

        # Инициализация
        h = torch.zeros(B, self.hidden_size, device=device)
        c = torch.zeros(B, self.hidden_size, device=device)
        hidden = (h, c)

        out_hid = torch.zeros(B, steps, self.hidden_size, device=device)
        targets = text[:, 0]  # <SOS>

        for t in range(steps):
            onehots = self._char_to_onehot(targets)
            hidden, _ = self.attention_cell(hidden, batch_H, onehots)

            h_out = hidden[0]
            out_hid[:, t, :] = h_out

            out = F.dropout(h_out, p=self.dropout_p, training=self.training)
            logits_t = self.generator(out)

            # Scheduled sampling
            if t < steps - 1:
                if torch.rand(1).item() < self.sampling_prob:
                    targets = logits_t.argmax(1)
                else:
                    targets = text[:, t + 1]

        logits = self.generator(out_hid)
        logits = self._mask_logits(logits)
        return logits


# ============================================================================
# Main Model
# ============================================================================

class TRBAModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_size: int = 256,
        num_encoder_layers: int = 2,
        img_h: int = 64,
        img_w: int = 256,
        cnn_in_channels: int = 3,
        cnn_out_channels: int = 512,
        cnn_backbone: str = "seresnet31",
        sos_id: int = 1,
        eos_id: int = 2,
        pad_id: int = 0,
        blank_id: Optional[int] = 3,
        enc_dropout_p: float = 0.1,
        use_ctc_head: bool = True,
        use_attention_head: bool = True,
    ):
        super().__init__()

        assert use_ctc_head or use_attention_head, "Хотя бы одна голова должна быть включена"

        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_encoder_layers = num_encoder_layers
        self.img_h = img_h
        self.img_w = img_w
        self.use_ctc_head = use_ctc_head
        self.use_attention_head = use_attention_head
        self.cnn_backbone = cnn_backbone.lower()

        # ===== CNN Encoder =====
        backbone_cls = CNN_BACKBONES.get(self.cnn_backbone)
        if backbone_cls is None:
            available = ", ".join(sorted(CNN_BACKBONES))
            raise ValueError(
                f"Unsupported cnn_backbone '{cnn_backbone}'. Available: {available}"
            )

        self.cnn = backbone_cls(
            in_channels=cnn_in_channels,
            out_channels=cnn_out_channels,
        )

        # ===== 1D-CNN Encoder =====
        enc_dim = self.cnn.out_channels
        self.enc_conv = ConvEncoder1D(
            input_size=enc_dim,
            hidden_size=hidden_size,
            num_layers=num_encoder_layers,
            dropout=enc_dropout_p,
        )
        self.enc_dropout = nn.Dropout(enc_dropout_p)

        # ===== CTC Head (опционально) =====
        if self.use_ctc_head:
            self.ctc_head = nn.Linear(hidden_size, num_classes)
            self.ctc_loss_fn = nn.CTCLoss(
                blank=blank_id, reduction="mean", zero_infinity=True
            )

        # ===== Attention Head (опционально) =====
        if self.use_attention_head:
            self.attention_decoder = AttentionDecoder(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_classes=num_classes,
                sos_id=sos_id,
                eos_id=eos_id,
                pad_id=pad_id,
                blank_id=blank_id,
                dropout_p=0.1,
                sampling_prob=0.0,
            )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # CNN
        f = self.cnn(x)  # [B, C, H', W']
        
        # Pooling по высоте (ONNX-compatible: mean вместо AdaptiveAvgPool)
        f = f.mean(dim=2)  # [B, C, W']
        
        # Permute для RNN
        f = f.permute(0, 2, 1)  # [B, W', C]
        
        # 1D-CNN encoder (без flatten_parameters для ONNX)
        f = self.enc_conv(f)  # [B, W', hidden_size]
        f = self.enc_dropout(f)
        
        return f

    def forward_attention(
        self,
        x: torch.Tensor,
        text: Optional[torch.Tensor] = None,
        is_train: bool = False,
        batch_max_length: int = 25,
        onnx_mode: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass для Attention декодирования.
        
        Args:
            x: [B, 3, H, W] input images
            text: [B, T] target tokens (только для обучения)
            is_train: режим обучения или инференс
            batch_max_length: максимальная длина последовательности
            onnx_mode: ONNX режим (всегда max_length шагов)
            
        Returns:
            logits: [B, T, num_classes]
            preds: [B, T] (только для инференса) или None
        """
        assert self.use_attention_head, "Attention head не включена"
        
        enc_output = self.encode(x)  # [B, W, hidden_size]
        
        if is_train:
            assert text is not None, "text обязателен для обучения"
            logits = self.attention_decoder.forward_training(
                enc_output, text, batch_max_length
            )
            return logits, None
        else:
            logits, preds = self.attention_decoder.greedy_decode(
                enc_output, batch_max_length, onnx_mode=onnx_mode
            )
            return logits, preds

    def forward(
        self,
        x: torch.Tensor,
        text: Optional[torch.Tensor] = None,
        is_train: bool = True,
        batch_max_length: int = 25,
        mode: str = "attention",
        onnx_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Unified forward pass.
        
        Использование:
            Training (PyTorch):
                result = model(images, text=targets, mode="both", is_train=True)
                # Получаем оба выхода для dual loss
                
            Inference (PyTorch):
                result = model(images, mode="attention", is_train=False)
                preds = result["attention_preds"]
                
            ONNX Export:
                # Используйте wrapper из export_trba_to_onnx.py
                # Не нужно вызывать этот метод напрямую
        
        Args:
            x: [B, 3, H, W] input images
            text: [B, T] target tokens (только для обучения)
            is_train: режим обучения
            batch_max_length: максимальная длина последовательности
            mode: "attention" | "both"
            onnx_mode: True для ONNX экспорта (всегда max_length шагов)
            
        Returns:
            dict с ключами в зависимости от mode:
                "ctc_logits": [B, W, num_classes] (??? mode="both")
                "attention_logits": [B, T, num_classes]
                "attention_preds": [B, T] (только inference)
        """
        result: Dict[str, Any] = {}

        if mode == "attention":
            logits, preds = self.forward_attention(
                x, text, is_train, batch_max_length, onnx_mode
            )
            result["attention_logits"] = logits
            if preds is not None:
                result["attention_preds"] = preds
        elif mode == "both":
            enc_output = self.encode(x)

            if self.use_ctc_head:
                result["ctc_logits"] = self.ctc_head(enc_output)

            if self.use_attention_head:
                if is_train:
                    assert text is not None, "text ???+???????'??>??? ???>?? ???+?????????"
                    logits = self.attention_decoder.forward_training(
                        enc_output, text, batch_max_length
                    )
                    result["attention_logits"] = logits
                else:
                    logits, preds = self.attention_decoder.greedy_decode(
                        enc_output, batch_max_length, onnx_mode=onnx_mode
                    )
                    result["attention_logits"] = logits
                    result["attention_preds"] = preds
        else:
            raise ValueError(f"mode ?????>????? ?+?<?'?? 'attention' ??>?? 'both', ?????>??????: {mode}")

        return result

    def compute_ctc_loss(
        self, 
        ctc_logits: torch.Tensor, 
        targets: torch.Tensor, 
        target_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Вычисление CTC loss.
        
        Args:
            ctc_logits: [B, W, num_classes]
            targets: [B, T] target tokens
            target_lengths: [B] длины целевых последовательностей
            
        Returns:
            ctc_loss
        """
        if not self.use_ctc_head:
            # Return zero that's part of computational graph
            return ctc_logits.sum() * 0.0
        
        # CTC expects [W, B, num_classes]
        ctc_logits = ctc_logits.permute(1, 0, 2)  # [W, B, num_classes]
        
        B, W = ctc_logits.size(1), ctc_logits.size(0)
        input_lengths = torch.full((B,), W, dtype=torch.long, device=ctc_logits.device)
        
        # Apply log_softmax before CTC loss
        log_probs = ctc_logits.log_softmax(2)
        
        try:
            ctc_loss = nn.functional.ctc_loss(
                log_probs, 
                targets, 
                input_lengths, 
                target_lengths,
                blank=0,
                reduction='mean',
                zero_infinity=True
            )
        except Exception as e:
            # Если CTC падает, возвращаем 0 подключенный к графу
            print(f"Warning: CTC loss failed: {e}")
            ctc_loss = ctc_logits.sum() * 0.0
        
        return ctc_loss
