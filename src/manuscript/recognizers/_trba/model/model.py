import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .seresnet31 import SEResNet31
from .unet import CompactUNet


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size, hidden_size, bidirectional=True, batch_first=True
        )
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        self.rnn.flatten_parameters()
        h, _ = self.rnn(x)  # [B, T, 2H]
        out = self.linear(h)  # [B, T, D]
        return out


class BidirectionalGRU(nn.Module):
    """Bidirectional GRU encoder layer (faster than LSTM, less memory)."""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        self.rnn.flatten_parameters()
        h, _ = self.rnn(x)  # [B, T, 2H]
        out = self.linear(h)  # [B, T, D]
        return out


class AttentionCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_embeddings, dropout_p=0.1):
        super().__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p

    def forward(self, prev_hidden, batch_H, char_onehots):
        # batch_H: [B, Tenc, C]
        proj_H = self.i2h(batch_H)  # [B, Tenc, H]
        proj_h = self.h2h(prev_hidden[0]).unsqueeze(1)
        e = self.score(torch.tanh(proj_H + proj_h))  # [B, Tenc, 1]

        alpha = F.softmax(e, dim=1)  # [B, Tenc, 1]
        alpha = F.dropout(alpha, p=self.dropout_p, training=self.training)

        context = torch.bmm(alpha.transpose(1, 2), batch_H).squeeze(1)  # [B, C]
        x = torch.cat([context, char_onehots], 1)  # [B, C + V]
        cur_hidden = self.rnn(x, prev_hidden)  # (h, c)
        return cur_hidden, alpha


class AttentionCellGRU(nn.Module):
    """Attention cell with GRU decoder (faster, less memory than LSTM)."""

    def __init__(self, input_size, hidden_size, num_embeddings, dropout_p=0.1):
        super().__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.GRUCell(input_size + num_embeddings, hidden_size)
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p

    def forward(self, prev_hidden, batch_H, char_onehots):
        # batch_H: [B, Tenc, C]
        # prev_hidden: [B, H] for GRU (single hidden state)
        proj_H = self.i2h(batch_H)  # [B, Tenc, H]
        proj_h = self.h2h(prev_hidden).unsqueeze(1)
        e = self.score(torch.tanh(proj_H + proj_h))  # [B, Tenc, 1]

        alpha = F.softmax(e, dim=1)  # [B, Tenc, 1]
        alpha = F.dropout(alpha, p=self.dropout_p, training=self.training)

        context = torch.bmm(alpha.transpose(1, 2), batch_H).squeeze(1)  # [B, C]
        x = torch.cat([context, char_onehots], 1)  # [B, C + V]
        cur_hidden = self.rnn(x, prev_hidden)  # [B, H]
        return cur_hidden, alpha


class Attention(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_classes,
        sos_id: int,
        eos_id: int,
        pad_id: int,
        blank_id: Optional[int] = None,
        dropout_p: float = 0.1,
        sampling_prob: float = 0.0,
        decoder_type: str = "LSTM",  # "LSTM" or "GRU"
    ):
        super().__init__()

        # Choose decoder cell type
        decoder_type = decoder_type.upper()
        if decoder_type == "LSTM":
            self.attention_cell = AttentionCell(
                input_size, hidden_size, num_classes, dropout_p=dropout_p
            )
            self.is_lstm = True
        elif decoder_type == "GRU":
            self.attention_cell = AttentionCellGRU(
                input_size, hidden_size, num_classes, dropout_p=dropout_p
            )
            self.is_lstm = False
        else:
            raise ValueError(
                f"Unknown decoder_type: {decoder_type}. Use 'LSTM' or 'GRU'"
            )

        self.decoder_type = decoder_type
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.blank_id = blank_id

        self.generator = nn.Linear(hidden_size, num_classes)
        self.dropout_p = dropout_p
        self.sampling_prob = sampling_prob

    def _char_to_onehot(self, input_char, device):
        B = input_char.size(0)
        one_hot = torch.zeros(B, self.num_classes, device=device)
        one_hot.scatter_(1, input_char.unsqueeze(1), 1.0)
        return one_hot

    def _mask_logits(self, logits):
        if self.blank_id is not None:
            if logits.dim() == 3:
                logits[:, :, self.blank_id] = -1e4
            else:
                logits[:, self.blank_id] = -1e4
        return logits

    @torch.no_grad()
    def _beam_decode(
        self,
        batch_H,
        batch_max_length=25,
        beam_size=5,
        alpha: float = 0.9,
        temperature: float = 1.7,
    ):
        B = batch_H.size(0)
        device = batch_H.device
        H, V = self.hidden_size, self.num_classes

        h0 = torch.zeros(B, H, device=device)

        # Initialize hidden state based on decoder type
        if self.is_lstm:
            c0 = torch.zeros(B, H, device=device)
            beam_h = h0.unsqueeze(1).repeat(1, beam_size, 1).contiguous()
            beam_c = c0.unsqueeze(1).repeat(1, beam_size, 1).contiguous()
        else:
            # GRU only needs h
            beam_h = h0.unsqueeze(1).repeat(1, beam_size, 1).contiguous()
            beam_c = None

        beam_tokens = torch.full(
            (B, beam_size, 1), self.sos_id, dtype=torch.long, device=device
        )
        beam_scores = torch.full((B, beam_size), float("-inf"), device=device)
        beam_scores[:, 0] = 0.0

        finished = torch.zeros(B, beam_size, dtype=torch.bool, device=device)

        probs_trace = None

        for t in range(batch_max_length):
            last_tok = beam_tokens[:, :, -1].reshape(B * beam_size)  # [B*beam]
            flat_h = beam_h.reshape(B * beam_size, H)

            onehots = self._char_to_onehot(last_tok, device)  # [B*beam, V]

            # Prepare hidden state based on decoder type
            if self.is_lstm:
                flat_c = beam_c.reshape(B * beam_size, H)
                prev_hidden = (flat_h, flat_c)
            else:
                prev_hidden = flat_h

            hidden, _ = self.attention_cell(
                prev_hidden,
                batch_H.repeat_interleave(beam_size, dim=0),
                onehots,
            )

            # Extract h and c from hidden state
            if self.is_lstm:
                h_out, c_out = hidden
            else:
                h_out = hidden
                c_out = None

            out = F.dropout(h_out, p=self.dropout_p, training=self.training)
            logits_t = self.generator(out)  # [B*beam, V]
            logits_t = self._mask_logits(logits_t)
            if temperature != 1.0:
                eps = 1e-6
                logits_t = logits_t / max(temperature, eps)

            log_probs = F.log_softmax(logits_t, dim=-1)

            log_probs = log_probs.view(B, beam_size, V)
            h_new = h_out.view(B, beam_size, H)
            if self.is_lstm:
                c_new = c_out.view(B, beam_size, H)
            else:
                c_new = None

            if finished.any():
                mask = finished.unsqueeze(-1)  # [B, beam, 1]
                log_probs = torch.where(
                    mask.expand_as(log_probs),
                    torch.full_like(log_probs, float("-inf")),
                    log_probs,
                )
                log_probs[..., self.eos_id] = torch.where(
                    mask.squeeze(-1),
                    torch.zeros_like(log_probs[..., self.eos_id]),
                    log_probs[..., self.eos_id],
                )

            next_sum = beam_scores.unsqueeze(-1) + log_probs  # [B, beam, V]
            if alpha > 0:
                lp = ((5.0 + (t + 1)) ** alpha) / (6.0**alpha)
                next_scores = next_sum / lp
            else:
                next_scores = next_sum

            next_scores_flat = next_scores.view(B, -1)  # [B, beam*V]
            top_scores, top_idx = torch.topk(
                next_scores_flat, k=beam_size, dim=-1
            )  # [B, beam]

            next_beam = top_idx // V  # [B, beam]
            next_token = (top_idx % V).clamp(0, V - 1)  # [B, beam]

            gather_h = h_new.gather(
                1, next_beam.unsqueeze(-1).expand(-1, -1, H)
            )  # [B, beam, H]

            if self.is_lstm:
                gather_c = c_new.gather(1, next_beam.unsqueeze(-1).expand(-1, -1, H))
            else:
                gather_c = None

            beam_tokens = torch.cat(
                [
                    beam_tokens.gather(
                        1, next_beam.unsqueeze(-1).expand(-1, -1, beam_tokens.size(-1))
                    ),
                    next_token.unsqueeze(-1),
                ],
                dim=-1,
            )

            if alpha > 0:
                sum_logp = top_scores * lp
            else:
                sum_logp = top_scores
            beam_scores = sum_logp

            beam_h = gather_h
            if self.is_lstm:
                beam_c = gather_c
            else:
                beam_c = None

            finished = finished.gather(1, next_beam) | (next_token == self.eos_id)

            cur_logits = logits_t.view(B, beam_size, V)  # от текущего шага ДО выбора
            cur_logits_sel = cur_logits.gather(
                1, next_beam.unsqueeze(-1).expand(-1, -1, V)
            )
            if probs_trace is None:
                probs_trace = cur_logits_sel.unsqueeze(2)  # [B, beam, 1, V]
            else:
                probs_trace = probs_trace.gather(
                    1,
                    next_beam.unsqueeze(-1)
                    .unsqueeze(-1)
                    .expand(-1, -1, probs_trace.size(2), V),
                )
                probs_trace = torch.cat(
                    [probs_trace, cur_logits_sel.unsqueeze(2)], dim=2
                )

            if finished.all():
                break

        best_idx = beam_scores.argmax(-1)  # [B]
        best_tokens = beam_tokens[
            torch.arange(B, device=device), best_idx
        ]  # [B, T+1], с SOS в начале

        probs_best = probs_trace[torch.arange(B, device=device), best_idx]  # [B, T, V]

        return probs_best, best_tokens[:, 1:]

    @torch.no_grad()
    def _greedy_decode(self, batch_H, batch_max_length=25):
        B = batch_H.size(0)
        device = batch_H.device
        steps = batch_max_length + 1

        h = torch.zeros(B, self.hidden_size, device=device)

        # Initialize hidden state based on decoder type
        if self.is_lstm:
            c = torch.zeros(B, self.hidden_size, device=device)
            hidden = (h, c)
        else:
            hidden = h

        targets = torch.full((B,), self.sos_id, dtype=torch.long, device=device)

        probs = []
        preds = []

        for t in range(steps):
            onehots = self._char_to_onehot(targets, device=device)
            hidden, _ = self.attention_cell(hidden, batch_H, onehots)

            # Extract h from hidden state
            if self.is_lstm:
                h_out = hidden[0]
            else:
                h_out = hidden

            out = F.dropout(h_out, p=self.dropout_p, training=self.training)
            logits_t = self.generator(out)
            logits_t = self._mask_logits(logits_t)

            probs.append(logits_t.unsqueeze(1))
            next_tokens = logits_t.argmax(1)
            preds.append(next_tokens.unsqueeze(1))

            targets = next_tokens.clone()

            if (next_tokens == self.eos_id).all():
                break

        probs = torch.cat(probs, dim=1)  # [B, T, V]
        preds = torch.cat(preds, dim=1)  # [B, T]
        return probs, preds

    def forward(
        self,
        batch_H,
        text=None,
        is_train=True,
        batch_max_length=25,
        mode: str = "greedy",
        beam_size: int = 5,
        alpha: float = 0.6,
        temperature: float = 1.0,
    ):
        # ===== 1. Инференс =====
        if not is_train:
            if mode == "greedy":
                return self._greedy_decode(batch_H, batch_max_length)
            elif mode == "beam":
                return self._beam_decode(
                    batch_H,
                    batch_max_length,
                    beam_size=beam_size,
                    alpha=alpha,
                    temperature=temperature,
                )
            else:
                raise ValueError(f"Unknown decode mode: {mode}")

        # ===== 2. Обучение (teacher forcing) =====
        assert (
            text is not None
        ), "Для обучения необходимо подать `text` с <SOS> токеном в начале"

        device = batch_H.device
        B = batch_H.size(0)
        steps = batch_max_length + 1

        h = torch.zeros(B, self.hidden_size, device=device)

        # Initialize hidden state based on decoder type
        if self.is_lstm:
            c = torch.zeros(B, self.hidden_size, device=device)
            hidden = (h, c)
        else:
            hidden = h

        out_hid = torch.zeros(B, steps, self.hidden_size, device=device)
        targets = text[:, 0]  # <SOS>

        for t in range(steps):
            onehots = self._char_to_onehot(targets, device=device)
            hidden, _ = self.attention_cell(hidden, batch_H, onehots)

            # Extract h from hidden state
            if self.is_lstm:
                h_out = hidden[0]
            else:
                h_out = hidden

            out_hid[:, t, :] = h_out

            out = F.dropout(h_out, p=self.dropout_p, training=self.training)
            logits_t = self.generator(out)

            # scheduled sampling
            if t < steps - 1:
                if torch.rand(1).item() < self.sampling_prob:
                    targets = logits_t.argmax(1)
                else:
                    targets = text[:, t + 1]

        logits = self.generator(out_hid)
        logits = self._mask_logits(logits)
        return logits


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size, hidden_size, bidirectional=True, batch_first=True
        )
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        self.rnn.flatten_parameters()
        h, _ = self.rnn(x)  # [B, T, 2H]
        out = self.linear(h)  # [B, T, D]
        return out


class TRBAModel(nn.Module):
    def __init__(
        self,
        num_classes,
        hidden_size=256,
        num_encoder_layers=2,
        encoder_type="LSTM",  # "LSTM" or "GRU"
        decoder_type="LSTM",  # "LSTM" or "GRU"
        use_unet=False,  # NEW: U-Net для очистки фона
        img_h=64,
        img_w=256,
        cnn_in_channels=3,
        cnn_out_channels=512,
        sos_id: int = 1,
        eos_id: int = 2,
        pad_id: int = 0,
        blank_id: Optional[int] = 3,
        enc_dropout_p: float = 0.1,
        dropblock_p: float = 0.0,
        dropblock_block_size: int = 5,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_encoder_layers = num_encoder_layers
        self.encoder_type = encoder_type.upper()
        self.decoder_type = decoder_type.upper()
        self.use_unet = use_unet
        self.img_h = img_h
        self.img_w = img_w
        self.cnn_in_channels = cnn_in_channels
        self.cnn_out_channels = cnn_out_channels

        # U-Net для создания бинарной маски (опционально)
        if self.use_unet:
            self.unet = CompactUNet(
                in_channels=cnn_in_channels,
                features=[
                    32,
                    64,
                    128,
                ],  # Увеличенная архитектура для более тонкой работы
                hard_binarize=True,  # Жесткая бинаризация (0 или 1)
                threshold=0.5,  # Порог бинаризации
            )
        else:
            self.unet = None

        self.cnn = SEResNet31(
            in_channels=cnn_in_channels,
            out_channels=cnn_out_channels,
            dropblock_p=dropblock_p,
            dropblock_block_size=dropblock_block_size,
        )

        self.pool = nn.AdaptiveAvgPool2d((1, None))  # -> [B, C, 1, W]

        enc_dim = self.cnn.out_channels

        # Choose encoder type
        if self.encoder_type == "LSTM":
            encoder_class = BidirectionalLSTM
        elif self.encoder_type == "GRU":
            encoder_class = BidirectionalGRU
        else:
            raise ValueError(
                f"Unknown encoder_type: {encoder_type}. Use 'LSTM' or 'GRU'"
            )

        encoder_layers = []
        for i in range(num_encoder_layers):
            input_dim = enc_dim if i == 0 else hidden_size
            encoder_layers.append(encoder_class(input_dim, hidden_size, hidden_size))
        self.enc_rnn = nn.Sequential(*encoder_layers)
        enc_dim = hidden_size

        self.enc_dropout = nn.Dropout(enc_dropout_p)

        self.attn = Attention(
            input_size=enc_dim,
            hidden_size=hidden_size,
            num_classes=num_classes,
            sos_id=sos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            blank_id=blank_id,
            dropout_p=0.1,
            sampling_prob=0.0,
            decoder_type=self.decoder_type,
        )

    def encode(self, x):
        """
        Encode input images to feature sequences.

        Parameters
        ----------
        x : torch.Tensor
            Input images [B, C, H, W].

        Returns
        -------
        Tuple[torch.Tensor, Optional[torch.Tensor]]
            - Encoded features [B, W, H]
            - U-Net binary mask [B, 1, H, W] if use_unet=True, else None
        """
        # Apply U-Net binary mask if enabled
        unet_mask = None
        if self.use_unet:
            unet_mask = self.unet(x)  # [B, 1, H, W] - soft mask [0, 1] from sigmoid

            # РАДИКАЛЬНО: заменяем RGB изображение на бинарную маску!
            # Вместо x * mask, отправляем саму маску в CNN
            # Повторяем одноканальную маску на 3 канала для совместимости с ResNet
            x_input = unet_mask.repeat(1, 3, 1, 1)  # [B, 1, H, W] -> [B, 3, H, W]
        else:
            x_input = x

        f = self.cnn(x_input)  # [B, C, H, W]
        f = self.pool(f).squeeze(2)  # [B, C, W]
        f = f.permute(0, 2, 1)  # [B, W, C]
        f = self.enc_rnn(f)  # [B, W, H]
        f = self.enc_dropout(f)
        return f, unet_mask

    def forward(
        self,
        x,
        text=None,
        is_train=True,
        batch_max_length=25,
        mode: str = "greedy",
        beam_size: int = 5,
        alpha: float = 0.6,
        temperature: float = 1.0,
        return_unet_output: bool = False,
        return_dict: bool = False,
    ):
        """
        Forward pass through the model.

        Parameters
        ----------
        return_unet_output : bool, optional
            If True, returns tuple of (probs, preds, unet_output).
            If False, returns tuple of (probs, preds) as before.
            Default is False for backward compatibility.
        return_dict : bool, optional
            If True, returns dict with keys: 'output', 'unet_output'.
            Useful for training with U-Net auxiliary loss.
            Default is False.

        Returns
        -------
        tuple, Tensor, or dict
            - If return_dict=True: {'output': logits/probs/preds, 'unet_output': x_cleaned}
            - If is_train=True and not return_dict: logits [B, T, V]
            - If is_train=False and return_unet_output=False: (probs, preds)
            - If is_train=False and return_unet_output=True: (probs, preds, unet_mask)
        """
        enc, unet_mask = self.encode(x)

        result = self.attn(
            enc,
            text=text,
            is_train=is_train,
            batch_max_length=batch_max_length,
            mode=mode,
            beam_size=beam_size,
            alpha=alpha,
            temperature=temperature,
        )

        # Return dict format for visualization
        if return_dict:
            return {
                "output": result,
                "unet_mask": unet_mask,
            }

        # В training mode возвращается только logits
        if is_train:
            return result  # logits

        # В inference mode возвращается (probs, preds)
        probs, preds = result

        if return_unet_output:
            return probs, preds, unet_mask
        else:
            return probs, preds
