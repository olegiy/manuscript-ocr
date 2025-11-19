import os
import json
from pathlib import Path
from typing import List, Union, Tuple, Optional, Sequence, Dict, Any

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

try:
    import gdown
except ImportError:  # pragma: no cover - optional dependency for default weights
    gdown = None  # type: ignore[assignment]

from .data.transforms import load_charset
from .training.train import Config, run_training


class TRBA:
    _DEFAULT_PRESET_NAME = "exp_1_baseline"
    _DEFAULT_RELEASE_WEIGHTS_URL = (
        "https://github.com/konstantinkozhin/manuscript-ocr/"
        "releases/download/v0.1.0/trba_exp_1_64.onnx"
    )
    _DEFAULT_RELEASE_CONFIG_URL = (
        "https://github.com/konstantinkozhin/manuscript-ocr/"
        "releases/download/v0.1.0/trba_exp_1_64.json"
    )
    _DEFAULT_STORAGE_ROOT = Path.home() / ".manuscript" / "trba"
    _DEFAULT_WEIGHTS_FILENAME = "weights.onnx"
    _DEFAULT_CONFIG_FILENAME = "config.json"

    def __init__(
        self,
        weights_path: Optional[str] = None,
        charset_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize TRBA text recognition model with ONNX Runtime.

        Parameters
        ----------
        weights_path : str or Path, optional
            Path to the ONNX model file (.onnx). If ``None``,
            default pretrained ONNX model will be automatically downloaded to
            ``~/.manuscript/trba/exp_1_baseline/weights.onnx``.
        charset_path : str or Path, optional
            Path to character set file. If ``None``, uses the default charset
            located at ``configs/charset.txt`` within the package.
        config_path : str or Path, optional
            Path to model configuration JSON file. If ``None``, attempts to
            infer config path from model location or downloads default config
            when using default weights.
        device : {"cuda", "cpu"}, optional
            Compute device. If ``None``, automatically selects CUDA if
            available (requires onnxruntime-gpu), otherwise CPU.

        Raises
        ------
        FileNotFoundError
            If specified model, config, or charset files do not exist.
        RuntimeError
            If gdown is not installed and default weights need to be
            downloaded.

        Notes
        -----
        The class provides three main public methods:

        - ``predict`` — run text recognition inference on cropped word images.
        - ``train`` — high-level training entrypoint to train a TRBA model
          on custom datasets.
        - ``export_to_onnx`` — static method to export PyTorch model to ONNX format.

        Examples
        --------
        Create recognizer with ONNX model:

        >>> from manuscript.recognizers import TRBA
        >>> recognizer = TRBA(weights_path="path/to/model.onnx")

        With explicit config:

        >>> recognizer = TRBA(
        ...     weights_path="path/to/model.onnx",
        ...     config_path="path/to/config.json"
        ... )

        Force CPU execution:

        >>> recognizer = TRBA(weights_path="model.onnx", device="cpu")
        
        Notes
        -----
        Model uses ONNX Runtime for fast inference on CPU and GPU.
        For GPU acceleration, install: ``pip install onnxruntime-gpu``
        
        To export PyTorch model to ONNX:
        ```
        python scripts/export_trba_to_onnx.py \
            --weights best_weights.pth \
            --config config.json \
            --output model.onnx
        ```
        """
        resolved_model_path, resolved_config_path = (
            self._resolve_model_and_config_paths(
                weights_path,
                config_path,
            )
        )

        self.model_path = resolved_model_path

        if charset_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            charset_path = os.path.join(current_dir, "configs", "charset.txt")

        self.charset_path = os.fspath(charset_path)
        self.config_path = resolved_config_path

        if not os.path.exists(self.charset_path):
            raise FileNotFoundError(f"Charset file not found: {self.charset_path}")

        if self.config_path is not None:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = {}

        self.max_length = config.get("max_len", 25)
        self.hidden_size = config.get("hidden_size", 256)
        self.num_encoder_layers = config.get("num_encoder_layers", 2)
        self.img_h = config.get("img_h", 32)
        self.img_w = config.get("img_w", 256)
        self.cnn_in_channels = config.get("cnn_in_channels", 3)
        self.cnn_out_channels = config.get("cnn_out_channels", 512)
        self.cnn_backbone = config.get("cnn_backbone", "seresnet31")

        # Определяем device для ONNX Runtime
        self.device = device or ("cuda" if ort.get_device() == "GPU" else "cpu")

        self.itos, self.stoi = load_charset(charset_path)
        self.pad_id = self.stoi["<PAD>"]
        self.sos_id = self.stoi["<SOS>"]
        self.eos_id = self.stoi["<EOS>"]
        self.blank_id = self.stoi.get("<BLANK>", None)

        # Проверяем что файл существует и это ONNX модель
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        model_ext = Path(self.model_path).suffix.lower()
        if model_ext != ".onnx":
            raise ValueError(
                f"Expected .onnx file, got: {self.model_path}"
            )

        # Загружаем ONNX модель
        providers = []
        if self.device == "cuda":
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        self.onnx_session = ort.InferenceSession(str(self.model_path), providers=providers)

    def _resolve_model_and_config_paths(
        self,
        model_path: Optional[str],
        config_path: Optional[str],
    ) -> Tuple[str, Optional[str]]:
        if model_path is None:
            return self._ensure_default_artifacts(config_path)

        resolved_model_path = os.fspath(model_path)
        if not os.path.exists(resolved_model_path):
            raise FileNotFoundError(
                f"Model checkpoint not found: {resolved_model_path}"
            )

        if config_path is not None:
            resolved_config_path = os.fspath(config_path)
            if not os.path.exists(resolved_config_path):
                raise FileNotFoundError(
                    f"Config file not found: {resolved_config_path}"
                )
        else:
            resolved_config_path = self._infer_config_path_from_weights(
                resolved_model_path
            )

        return resolved_model_path, resolved_config_path

    def _infer_config_path_from_weights(self, weights_path: str) -> Optional[str]:
        weights_file = Path(weights_path)
        candidates = [
            weights_file.with_suffix(".json"),
            weights_file.parent / "config.json",
        ]

        for candidate in candidates:
            if candidate.exists():
                return os.fspath(candidate)
        return None

    def _ensure_default_artifacts(
        self,
        config_path: Optional[str],
    ) -> Tuple[str, Optional[str]]:
        storage_root = self._DEFAULT_STORAGE_ROOT.expanduser()
        target_dir = storage_root / self._DEFAULT_PRESET_NAME
        target_dir.mkdir(parents=True, exist_ok=True)

        weights_dest = target_dir / self._DEFAULT_WEIGHTS_FILENAME
        if not weights_dest.exists():
            self._download_file(self._DEFAULT_RELEASE_WEIGHTS_URL, weights_dest)

        if config_path is not None:
            resolved_config_path = os.fspath(config_path)
            if not os.path.exists(resolved_config_path):
                raise FileNotFoundError(
                    f"Config file not found: {resolved_config_path}"
                )
        else:
            config_dest = target_dir / self._DEFAULT_CONFIG_FILENAME
            if not config_dest.exists():
                self._download_file(self._DEFAULT_RELEASE_CONFIG_URL, config_dest)
            resolved_config_path = os.fspath(config_dest)

        return os.fspath(weights_dest), resolved_config_path

    def _download_file(self, url: str, destination: Path) -> None:
        if gdown is None:
            raise RuntimeError(
                "gdown is required to download default TRBA weights. "
                "Install gdown or pass weights_path/model_path explicitly."
            )

        print(f"Downloading TRBA artifact from {url} -> {destination}")
        gdown.download(url, os.fspath(destination), quiet=False)
        if not destination.exists():
            raise RuntimeError(f"Failed to download artifact from {url}")

    def _preprocess_image(
        self, image: Union[np.ndarray, str, Image.Image]
    ) -> np.ndarray:
        """
        Preprocess image for ONNX inference. Returns [1, 3, H, W] numpy array.
        
        Applies same preprocessing as training:
        1. Resize with aspect ratio preservation and padding
        2. Normalize with mean=0.5, std=0.5 (same as albumentations)
        """
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Cannot read image: {image}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            img = np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            img = image.copy()
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Resize with aspect ratio preservation and padding (like ResizeAndPadA)
        h, w = img.shape[:2]
        scale = min(self.img_h / max(h, 1), self.img_w / max(w, 1))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        
        # Choose interpolation
        if new_h < h or new_w < w:
            interp = cv2.INTER_AREA
        else:
            interp = cv2.INTER_LINEAR
        
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
        
        # Create white canvas and paste resized image (center alignment)
        canvas = np.full((self.img_h, self.img_w, 3), 255, dtype=np.uint8)
        
        # Center vertically
        y_offset = (self.img_h - new_h) // 2
        # Left align horizontally
        x_offset = 0
        
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_resized
        
        # Normalize: mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
        # This is equivalent to: (x / 255.0 - 0.5) / 0.5 = (x - 127.5) / 127.5
        img_normalized = (canvas.astype(np.float32) - 127.5) / 127.5
        
        # Convert to CHW format
        img_chw = np.transpose(img_normalized, (2, 0, 1))  # HWC -> CHW
        img_batch = np.expand_dims(img_chw, axis=0)  # Add batch dimension
        
        return img_batch

    def predict(
        self,
        images: Union[
            np.ndarray, str, Image.Image, List[Union[np.ndarray, str, Image.Image]]
        ],
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        """
        Run text recognition on one or more word images.

        Parameters
        ----------
        images : str, Path, numpy.ndarray, PIL.Image, or list thereof
            Single image or list of images to recognize. Each image can be:

            - Path to image file (str or Path)
            - RGB numpy array with shape ``(H, W, 3)`` in ``uint8``
            - PIL Image object

        batch_size : int, optional
            Number of images to process simultaneously. Larger batches are
            faster but require more memory. Default is 32.

        Returns
        -------
        list of dict
            Recognition results as list of dictionaries, each containing:

            - ``"text"`` : str — recognized text
            - ``"confidence"`` : float — recognition confidence in [0, 1]

            If input is a single image, returns a list with one element.

        Examples
        --------
        Recognize single image:

        >>> from manuscript.recognizers import TRBA
        >>> recognizer = TRBA()
        >>> results = recognizer.predict("word_image.jpg")
        >>> print(f"Text: '{results[0]['text']}' (confidence: {results[0]['confidence']:.3f})")

        Process numpy arrays:

        >>> import cv2
        >>> img = cv2.imread("word.jpg")
        >>> img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        >>> results = recognizer.predict(img_rgb)
        >>> print(results[0]["text"])
        """

        if not isinstance(images, list):
            images_list = [images]
        else:
            images_list = images

        results: List[Dict[str, Any]] = []

        # Process images in batches
        for i in range(0, len(images_list), batch_size):
            batch_images = images_list[i : i + batch_size]

            # Preprocess batch
            batch_tensors = []
            for img in batch_images:
                tensor = self._preprocess_image(img)  # [1, 3, H, W]
                batch_tensors.append(tensor[0])  # Remove batch dim
            
            batch_input = np.stack(batch_tensors, axis=0)  # [B, 3, H, W]

            # ONNX inference
            input_name = self.onnx_session.get_inputs()[0].name
            output_name = self.onnx_session.get_outputs()[0].name
            
            ort_outputs = self.onnx_session.run(
                [output_name],
                {input_name: batch_input}
            )
            logits = ort_outputs[0]  # [B, max_length, num_classes]

            # Decode predictions
            preds = np.argmax(logits, axis=-1)  # [B, max_length]
            
            # Calculate confidence (softmax + mean log prob)
            probs = self._softmax(logits, axis=-1)  # [B, T, num_classes]

            for j in range(len(batch_images)):
                pred_row = preds[j]  # [max_length]
                
                # Decode to text
                decoded_chars = []
                for token_id in pred_row:
                    if token_id == self.eos_id:
                        break
                    if token_id not in [self.pad_id, self.sos_id]:
                        if token_id < len(self.itos):
                            decoded_chars.append(self.itos[token_id])
                
                text = "".join(decoded_chars)
                
                # Calculate confidence
                seq_probs = []
                for t, token_id in enumerate(pred_row):
                    if token_id == self.eos_id:
                        break
                    if token_id not in [self.pad_id, self.sos_id]:
                        seq_probs.append(probs[j, t, token_id])
                
                confidence = float(np.mean(seq_probs)) if seq_probs else 0.0
                
                results.append({"text": text, "confidence": confidence})

        return results

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Compute softmax along axis."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    @staticmethod
    def train(
        train_csvs: Union[str, Sequence[str]],
        train_roots: Union[str, Sequence[str]],
        val_csvs: Optional[Union[str, Sequence[str]]] = None,
        val_roots: Optional[Union[str, Sequence[str]]] = None,
        *,
        exp_dir: Optional[str] = None,
        charset_path: Optional[str] = None,
        encoding: str = "utf-8",
        img_h: int = 64,
        img_w: int = 256,
        max_len: int = 25,
        hidden_size: int = 256,
        num_encoder_layers: int = 3,
        cnn_in_channels: int = 3,
        cnn_out_channels: int = 512,
        cnn_backbone: str = "seresnet31",
        ctc_weight: float = 0.3,
        ctc_weight_decay_epochs: int = 50,
        ctc_weight_min: float = 0.01,
        max_grad_norm: float = 5.0,
        batch_size: int = 32,
        epochs: int = 20,
        lr: float = 1e-3,
        optimizer: str = "AdamW",
        scheduler: str = "OneCycleLR",
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        val_interval: int = 1,
        val_size: int = 3000,
        train_proportions: Optional[Sequence[float]] = None,
        num_workers: int = 0,
        seed: int = 42,
        resume_from: Optional[str] = None,
        save_interval: Optional[int] = None,
        device: str = "cuda",
        freeze_cnn: str = "none",
        freeze_enc_rnn: str = "none",
        freeze_attention: str = "none",
        pretrain_weights: Optional[object] = "default",
        **extra_config: Any,
    ):
        """
        Train TRBA text recognition model on custom datasets.

        Parameters
        ----------
        train_csvs : str, Path or sequence of paths
            Path(s) to training CSV files. Each CSV should have columns:
            ``image_path`` (relative to ``train_roots``) and ``text`` (ground
            truth transcription).
        train_roots : str, Path or sequence of paths
            Root directory/directories containing training images. Must have
            same length as ``train_csvs``.
        val_csvs : str, Path, sequence of paths, or None, optional
            Path(s) to validation CSV files with same format as ``train_csvs``.
            If ``None``, no validation is performed. Default is ``None``.
        val_roots : str, Path, sequence of paths, or None, optional
            Root directory/directories for validation images. Must match length
            of ``val_csvs`` if provided. Default is ``None``.
        exp_dir : str or Path, optional
            Experiment directory where checkpoints and logs will be saved.
            If ``None``, auto-generated based on timestamp. Default is ``None``.
        charset_path : str or Path, optional
            Path to character set file. If ``None``, uses default charset from
            package. Default is ``None``.
        encoding : str, optional
            Text encoding for reading CSV files. Default is ``"utf-8"``.
        img_h : int, optional
            Target height for input images (pixels). Default is 64.
        img_w : int, optional
            Target width for input images (pixels). Default is 256.
        max_len : int, optional
            Maximum sequence length for text recognition. Default is 25.
        hidden_size : int, optional
            Hidden dimension size for RNN encoder/decoder. Default is 256.
        num_encoder_layers : int, optional
            Number of Bidirectional LSTM layers in the encoder. Default is 2.
        cnn_in_channels : int, optional
            Number of input channels for CNN backbone (3 for RGB, 1 for grayscale). Default is 3.
        cnn_out_channels : int, optional
            Number of output channels from CNN backbone. Default is 512.
        cnn_backbone : {"seresnet31", "seresnet31-lite"}, optional
            CNN backbone variant. ``"seresnet31"`` keeps the standard SE-ResNet-31,
            while ``"seresnet31-lite"`` enables a depthwise-lite version. Default is ``"seresnet31"``.
        ctc_weight : float, optional
            Initial weight for CTC loss during training (CTC always used for stability):
            ``loss = attn_loss * (1 - ctc_weight) + ctc_loss * ctc_weight``.
            CTC weight decays over epochs. Default is 0.3.
        ctc_weight_decay_epochs : int, optional
            Number of epochs for CTC weight to decay to minimum. Default is 50.
        ctc_weight_min : float, optional
            Minimum value for CTC weight after decay. Default is 0.0.
        max_grad_norm : float, optional
            Maximum gradient norm for clipping (prevents gradient explosion/NaN).
            Default is 5.0.
        batch_size : int, optional
            Training batch size. Default is 32.
        epochs : int, optional
            Number of training epochs. Default is 20.
        lr : float, optional
            Learning rate. Default is 1e-3.
        optimizer : {"Adam", "SGD", "AdamW"}, optional
            Optimizer type. Default is ``"AdamW"``.
        scheduler : {"ReduceLROnPlateau", "CosineAnnealingLR", "OneCycleLR", "None"}, optional
            Learning rate scheduler type:

            - ``"OneCycleLR"`` — one-cycle policy with cosine annealing (default, recommended)
            - ``"ReduceLROnPlateau"`` — reduce LR on validation loss plateau
            - ``"CosineAnnealingLR"`` — cosine annealing over epochs
            - ``"None"`` or ``None`` — constant learning rate

            Default is ``"OneCycleLR"``.
        weight_decay : float, optional
            L2 weight decay coefficient. Default is 0.0.
        momentum : float, optional
            Momentum for SGD optimizer. Default is 0.9.
        val_interval : int, optional
            Perform validation every N epochs. Default is 1.
        val_size : int, optional
            Maximum number of validation samples to use. Default is 3000.
        train_proportions : sequence of float, optional
            Sampling proportions for multiple training datasets. Must sum to 1.0
            and match length of ``train_csvs``. If ``None``, datasets are
            concatenated equally. Default is ``None``.
        num_workers : int, optional
            Number of data loading workers. Default is 0.
        seed : int, optional
            Random seed for reproducibility. Default is 42.
        resume_from : str or Path, optional
            Path to checkpoint file to resume training from. Default is ``None``.
        save_interval : int, optional
            Save checkpoint every N epochs. If ``None``, only saves best model.
            Default is ``None``.
        device : {"cuda", "cpu"}, optional
            Training device. Default is ``"cuda"``.
        freeze_cnn : {"none", "all", "first", "last"}, optional
            CNN freezing policy. Default is ``"none"``.
        freeze_enc_rnn : {"none", "all", "first", "last"}, optional
            Encoder RNN freezing policy. Default is ``"none"``.
        freeze_attention : {"none", "all"}, optional
            Attention module freezing policy. Default is ``"none"``.
        pretrain_weights : str, Path, bool, or None, optional
            Pretrained weights to initialize from:

            - ``"default"`` or ``True`` — use release weights
            - ``None`` or ``False`` — train from scratch
            - str/Path — path or URL to custom weights file

            Default is ``"default"``.
        **extra_config : dict, optional
            Additional configuration parameters passed to training config.

        Returns
        -------
        str
            Path to the best model checkpoint saved during training.

        Examples
        --------
        Train on single dataset with validation:

        >>> from manuscript.recognizers import TRBA
        >>>
        >>> best_model = TRBA.train(
        ...     train_csvs="data/train.csv",
        ...     train_roots="data/train_images",
        ...     val_csvs="data/val.csv",
        ...     val_roots="data/val_images",
        ...     exp_dir="./experiments/trba_exp1",
        ...     epochs=50,
        ...     batch_size=64,
        ...     img_h=64,
        ...     img_w=256,
        ... )
        >>> print(f"Best model saved at: {best_model}")

        Train on multiple datasets with custom proportions:

        >>> train_csvs = ["data/dataset1/train.csv", "data/dataset2/train.csv"]
        >>> train_roots = ["data/dataset1/images", "data/dataset2/images"]
        >>> train_proportions = [0.7, 0.3]  # 70% from dataset1, 30% from dataset2
        >>>
        >>> best_model = TRBA.train(
        ...     train_csvs=train_csvs,
        ...     train_roots=train_roots,
        ...     train_proportions=train_proportions,
        ...     val_csvs="data/val.csv",
        ...     val_roots="data/val_images",
        ...     epochs=100,
        ...     lr=5e-4,
        ...     optimizer="AdamW",
        ...     weight_decay=1e-4,
        ... )

        Resume training from checkpoint:

        >>> best_model = TRBA.train(
        ...     train_csvs="data/train.csv",
        ...     train_roots="data/train_images",
        ...     resume_from="experiments/trba_exp1/checkpoints/last.pth",
        ...     epochs=100,
        ... )

        Fine-tune from pretrained weights with frozen CNN:

        >>> best_model = TRBA.train(
        ...     train_csvs="data/finetune.csv",
        ...     train_roots="data/finetune_images",
        ...     pretrain_weights="default",
        ...     freeze_cnn="all",
        ...     epochs=20,
        ...     lr=1e-4,
        ... )

        Train with CTC for stability (always enabled):

        >>> best_model = TRBA.train(
        ...     train_csvs="data/train.csv",
        ...     train_roots="data/train_images",
        ...     optimizer="AdamW",
        ...     scheduler="OneCycleLR",
        ...     lr=1e-3,
        ...     ctc_weight=0.3,
        ...     ctc_weight_decay_epochs=50,
        ...     max_grad_norm=5.0,
        ...     epochs=100,
        ... )
        """

        def _ensure_path_list(
            value: Optional[Union[str, Sequence[Optional[str]]]],
            field_name: str,
            allow_none: bool = False,
            allow_item_none: bool = False,
        ) -> Optional[List[Optional[str]]]:
            if value is None:
                if allow_none:
                    return None
                raise ValueError(f"{field_name} must be provided")

            if isinstance(value, (list, tuple)):
                raw_items = list(value)
            else:
                raw_items = [value]

            if not raw_items:
                raise ValueError(f"{field_name} must not be empty")

            result: List[Optional[str]] = []
            for item in raw_items:
                if item is None:
                    if allow_item_none:
                        result.append(None)
                    else:
                        raise ValueError(
                            f"{field_name} contains None but allow_item_none is False"
                        )
                else:
                    result.append(os.fspath(item))
            return result

        train_csvs_list = _ensure_path_list(train_csvs, "train_csvs")
        train_roots_list = _ensure_path_list(train_roots, "train_roots")

        if len(train_csvs_list) != len(train_roots_list):
            raise ValueError(
                "train_csvs and train_roots must contain the same number of items"
            )

        val_csvs_list = _ensure_path_list(
            val_csvs, "val_csvs", allow_none=True, allow_item_none=True
        )
        val_roots_list = _ensure_path_list(
            val_roots, "val_roots", allow_none=True, allow_item_none=True
        )

        if (val_csvs_list is None) ^ (val_roots_list is None):
            raise ValueError(
                "val_csvs and val_roots must both be provided or both be None"
            )
        if val_csvs_list is not None and len(val_csvs_list) != len(val_roots_list):
            raise ValueError(
                "val_csvs and val_roots must contain the same number of items"
            )

        resolved_charset = charset_path
        if resolved_charset is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            resolved_charset = os.path.join(current_dir, "configs", "charset.txt")

        config_payload: Dict[str, Any] = {
            "train_csvs": train_csvs_list,
            "train_roots": train_roots_list,
            "charset_path": resolved_charset,
            "encoding": encoding,
            "img_h": img_h,
            "img_w": img_w,
            "max_len": max_len,
            "hidden_size": hidden_size,
            "num_encoder_layers": num_encoder_layers,
            "cnn_in_channels": cnn_in_channels,
            "cnn_out_channels": cnn_out_channels,
            "cnn_backbone": cnn_backbone,
            "ctc_weight": ctc_weight,
            "ctc_weight_decay_epochs": ctc_weight_decay_epochs,
            "ctc_weight_min": ctc_weight_min,
            "max_grad_norm": max_grad_norm,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "val_interval": val_interval,
            "val_size": val_size,
            "num_workers": num_workers,
            "seed": seed,
        }

        if exp_dir is not None:
            config_payload["exp_dir"] = exp_dir
        if val_csvs_list is not None:
            config_payload["val_csvs"] = val_csvs_list
            config_payload["val_roots"] = val_roots_list
        if train_proportions is not None:
            config_payload["train_proportions"] = list(train_proportions)
        if resume_from is not None:
            config_payload["resume_from"] = resume_from
        if save_interval is not None:
            config_payload["save_interval"] = save_interval
        # Pretrained weights option:
        # - None/False/"none": skip
        # - "default"/True: use release weights
        # - str: path/URL to .pth/.pt/.ckpt
        if pretrain_weights is not None:
            config_payload["pretrain_weights"] = pretrain_weights

        if extra_config:
            config_payload.update(extra_config)

        # Freeze policies for model submodules
        config_payload["freeze_cnn"] = freeze_cnn
        config_payload["freeze_enc_rnn"] = freeze_enc_rnn
        config_payload["freeze_attention"] = freeze_attention

        config = Config(config_payload)
        return run_training(config, device=device)

    @staticmethod
    def export_to_onnx(
        weights_path: Union[str, Path],
        config_path: Union[str, Path],
        charset_path: Union[str, Path],
        output_path: Union[str, Path],
        opset_version: int = 14,
        simplify: bool = True,
    ) -> None:
        """
        Export TRBA PyTorch model to ONNX format.

        This method converts a trained TRBA model from PyTorch to ONNX format,
        which can be used for faster inference with ONNX Runtime. The exported
        model can be loaded using ``TRBA(model_path="model.onnx")``.

        Parameters
        ----------
        weights_path : str or Path
            Path to the PyTorch model weights file (.pth).
        config_path : str or Path
            Path to the model configuration JSON file. Used to determine
            model architecture (img_h, img_w, max_len, hidden_size, etc.).
        charset_path : str or Path
            Path to the charset file (charset.txt). Used to determine
            num_classes for the model.
        output_path : str or Path
            Path where the ONNX model will be saved (.onnx).
        opset_version : int, optional
            ONNX opset version to use for export. Default is 14.
        simplify : bool, optional
            If True, applies ONNX graph simplification using onnx-simplifier
            to optimize the model. Requires ``onnx-simplifier`` package.
            Default is True.

        Returns
        -------
        None
            The ONNX model is saved to ``output_path``.

        Raises
        ------
        ImportError
            If required packages (torch, onnx) are not installed.
        FileNotFoundError
            If ``weights_path`` or ``config_path`` do not exist.

        Notes
        -----
        The exported ONNX model has one output:

        - ``logits``: Character predictions with shape ``(batch, max_length, num_classes)``

        The model uses greedy decoding (argmax) and supports dynamic batch size.
        The sequence length is fixed to ``max_length`` from the config.

        Architecture exported:
        - CNN backbone (SE-ResNet-31 or SE-ResNet-31-Lite)
        - Bidirectional LSTM encoder
        - Attention decoder (greedy decoding)

        Note: Only the attention decoder is exported. CTC head is used only
        during training and is not included in the ONNX model.

        Examples
        --------
        Export TRBA model to ONNX:

        >>> from manuscript.recognizers import TRBA
        >>> TRBA.export_to_onnx(
        ...     weights_path="experiments/best_model/best_acc_weights.pth",
        ...     config_path="experiments/best_model/config.json",
        ...     charset_path="configs/charset.txt",
        ...     output_path="trba_model.onnx"
        ... )
        Loading TRBA model...
        === TRBA ONNX Export ===
        Max decoding length: 40
        Input size: 64x256
        [OK] ONNX model saved to: trba_model.onnx

        Export with custom opset:

        >>> TRBA.export_to_onnx(
        ...     weights_path="model.pth",
        ...     config_path="config.json",
        ...     charset_path="charset.txt",
        ...     output_path="model.onnx",
        ...     opset_version=16,
        ...     simplify=False
        ... )

        Use the exported model for inference:

        >>> recognizer = TRBA(weights_path="trba_model.onnx")
        >>> result = recognizer.predict("word_image.jpg")

        See Also
        --------
        TRBA.__init__ : Initialize TRBA recognizer with ONNX model.
        """
        import torch
        from .model.model import TRBAModel, TRBAONNXWrapper

        weights_path = Path(weights_path)
        config_path = Path(config_path)
        charset_path = Path(charset_path)
        output_path = Path(output_path)

        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not charset_path.exists():
            raise FileNotFoundError(f"Charset file not found: {charset_path}")

        # Load config
        print(f"Loading config from {config_path}...")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Extract model parameters
        max_length = config.get('max_len', 40)
        img_h = config.get('img_h', 64)
        img_w = config.get('img_w', 256)
        hidden_size = config.get('hidden_size', 256)
        num_encoder_layers = config.get('num_encoder_layers', 2)
        cnn_in_channels = config.get('cnn_in_channels', 3)
        cnn_out_channels = config.get('cnn_out_channels', 512)
        cnn_backbone = config.get('cnn_backbone', 'seresnet31')
        
        # Load charset to determine num_classes
        print(f"Loading charset from {charset_path}...")
        itos, stoi = load_charset(str(charset_path))
        num_classes = len(itos)  # itos already includes special tokens (PAD, SOS, EOS, BLANK, ...)
        print(f"Charset loaded: {len(itos)} total classes (including special tokens)")
        print(f"  First 4 tokens (special): {itos[:4]}")
        print(f"  Regular characters: {len(itos) - 4}")
        
        # Load weights
        print(f"\nLoading checkpoint from {weights_path}...")
        checkpoint = torch.load(str(weights_path), map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        print(f"\n=== TRBA ONNX Export ===")
        print(f"Max decoding length: {max_length}")
        print(f"Input size: {img_h}x{img_w}")
        print(f"Architecture: {cnn_backbone}")
        print(f"Hidden size: {hidden_size}")
        print(f"Num classes: {num_classes}")

        # Create PyTorch model with correct num_classes
        print(f"\nCreating model architecture...")
        model = TRBAModel(
            num_classes=num_classes,
            hidden_size=hidden_size,
            num_encoder_layers=num_encoder_layers,
            img_h=img_h,
            img_w=img_w,
            cnn_in_channels=cnn_in_channels,
            cnn_out_channels=cnn_out_channels,
            cnn_backbone=cnn_backbone,
            use_ctc_head=False,  # ONNX export only needs attention head
            use_attention_head=True,
        )

        # Load weights
        print(f"Loading weights into model...")
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        print("[OK] Model loaded")

        # Create ONNX wrapper
        print(f"\nCreating ONNX wrapper...")
        onnx_model = TRBAONNXWrapper(model, max_length=max_length)
        onnx_model.eval()

        # Create dummy input
        dummy_input = torch.randn(1, 3, img_h, img_w)

        print(f"Input shape: {dummy_input.shape}")

        # Test model before export
        print(f"\nTesting model before export...")
        with torch.no_grad():
            output = onnx_model(dummy_input)

        print(f"Output shape: {output.shape}")
        print(f"Expected: [1, {max_length}, {num_classes}]")

        # Export to ONNX
        print(f"\nExporting to ONNX (opset {opset_version})...")
        torch.onnx.export(
            onnx_model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            verbose=False,
        )

        print(f"[OK] ONNX model saved to: {output_path}")

        # Verify ONNX model
        import onnx

        print("\nVerifying ONNX model...")
        onnx_model_proto = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model_proto)
        print("[OK] ONNX model is valid")

        # Simplify if requested
        if simplify:
            try:
                import onnxsim

                print("\nSimplifying ONNX model...")
                model_simplified, check = onnxsim.simplify(onnx_model_proto)
                if check:
                    onnx.save(model_simplified, str(output_path))
                    print("[OK] ONNX model simplified")
                else:
                    print("[WARNING] Simplification failed, using original model")
            except ImportError:
                print("[WARNING] onnx-simplifier not installed, skipping simplification")
                print("  Install with: pip install onnx-simplifier")

        # Test ONNX inference
        try:
            print(f"\nTesting ONNX inference...")
            session = ort.InferenceSession(str(output_path))

            ort_inputs = {"input": dummy_input.numpy()}
            ort_outputs = session.run(None, ort_inputs)

            print(f"[OK] ONNX inference works!")
            print(f"  Output shape: {ort_outputs[0].shape}")

            # Compare with PyTorch
            torch_output = output.numpy()
            onnx_output = ort_outputs[0]

            max_diff = abs(torch_output - onnx_output).max()
            print(f"  Max difference vs PyTorch: {max_diff:.6f}")

            if max_diff < 1e-4:
                print(f"  [OK] Outputs match!")
            else:
                print(f"  [WARNING] Outputs differ slightly")

        except Exception as e:
            print(f"[WARNING] ONNX inference test failed: {e}")

        # Print summary
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\n[OK] Export complete! Model size: {file_size_mb:.1f} MB")
        print(f"\nInput shape: [batch_size, 3, {img_h}, {img_w}]")
        print(f"Output shape: [batch_size, {max_length}, {num_classes}]")
        print(f"Decoding: Greedy (argmax over last dimension)")

