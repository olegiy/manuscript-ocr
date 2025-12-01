import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import gdown
import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import ConcatDataset

from ...data import Block, Page, Word
from .dataset import EASTDataset
from .east import EASTModel
from .lanms import locality_aware_nms
from .train_utils import _run_training
from .utils import (
    decode_quads_from_maps,
    expand_boxes,
    read_image,
    sort_boxes_reading_order_with_resolutions,
    visualize_page,
)


class EAST:
    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        target_size: int = 1280,
        expand_ratio_w: float = 1.4,
        expand_ratio_h: float = 1.5,
        expand_power: float = 0.6,
        score_thresh: float = 0.6,
        iou_threshold: float = 0.05,
        iou_threshold_standard: Optional[float] = 0.05,
        score_geo_scale: float = 0.25,
        quantization: int = 2,
        axis_aligned_output: bool = True,
        remove_area_anomalies: bool = False,
        anomaly_sigma_threshold: float = 5.0,
        anomaly_min_box_count: int = 30,
    ):
        """
        Initialize EAST text detector with ONNX Runtime.

        Parameters
        ----------
        weights_path : str or Path, optional
            Path to ONNX model weights. If None, the model will be
            automatically downloaded to ``~/.manuscript/east/east_quad_23_05.onnx``.
        device : str, optional
            Compute device: ``"cuda"`` or ``"cpu"``. If None, selected automatically.
            Note: CUDA requires onnxruntime-gpu package.
        target_size : int, optional
            Input image size for inference. Images are resized to
            ``(target_size, target_size)``. Default is 1280.
        expand_ratio_w : float, optional
            Horizontal expansion factor applied to detected boxes after NMS.
            Default is 0.7.
        expand_ratio_h : float, optional
            Vertical expansion factor applied to detected boxes after NMS.
            Default is 0.7.
        expand_power : float, optional
            Power for non-linear box expansion. Controls how expansion scales with box size.
            - 1.0 = linear (small and large boxes expand equally)
            - <1.0 = small boxes expand more (e.g., 0.5, recommended for character-level detection)
            - >1.0 = large boxes expand more
            Default is 0.5.
        score_thresh : float, optional
            Confidence threshold for selecting candidate detections before NMS.
            Default is 0.7.
        iou_threshold : float, optional
            IoU threshold for locality-aware NMS merging phase. Default is 0.2.
        iou_threshold_standard : float, optional
            IoU threshold for standard NMS after locality-aware merging.
            If None, uses the same value as iou_threshold. Default is None.
        score_geo_scale : float, optional
            Scale factor for decoding geometry/score maps. Default is 0.25.
        quantization : int, optional
            Quantization resolution for point coordinates during decoding.
            Default is 2.
        axis_aligned_output : bool, optional
            If True, outputs axis-aligned rectangles instead of original quads.
            Default is True.
        remove_area_anomalies : bool, optional
            If True, removes quads with extremely large area relative to the
            distribution. Default is False.
        anomaly_sigma_threshold : float, optional
            Sigma threshold for anomaly area filtering. Default is 5.0.
        anomaly_min_box_count : int, optional
            Minimum number of boxes required before anomaly filtering.
            Default is 30.

        Notes
        -----
        The class provides two main public methods:

        - ``predict`` — run inference on a single image and return detections.
        - ``train`` — high-level training entrypoint to train an EAST model
          on custom datasets.

        The detector uses ONNX Runtime for fast inference on CPU and GPU.
        For GPU acceleration, install: ``pip install onnxruntime-gpu``
        """
        self.device = device or ("cuda" if ort.get_device() == "GPU" else "cpu")

        if weights_path is None:
            url = (
                "https://github.com/konstantinkozhin/manuscript-ocr"
                "/releases/download/v0.1.0/east_quad_23_05.onnx"
            )
            weights_dir = Path.home() / ".manuscript" / "east"
            weights_dir.mkdir(parents=True, exist_ok=True)
            out = weights_dir / "east_quad_23_05.onnx"
            if not out.exists():
                print(f"Downloading EAST ONNX model from {url} …")
                gdown.download(url, str(out), quiet=False)
            weights_path = str(out)

        if not Path(weights_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {weights_path}")

        providers = []
        if self.device == "cuda":
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        self.onnx_session = ort.InferenceSession(str(weights_path), providers=providers)

        self.target_size = target_size
        self.score_geo_scale = score_geo_scale
        self.expand_ratio_w = expand_ratio_w
        self.expand_ratio_h = expand_ratio_h
        self.expand_power = expand_power
        self.score_thresh = score_thresh
        self.iou_threshold = iou_threshold
        self.iou_threshold_standard = iou_threshold_standard
        self.quantization = quantization
        self.axis_aligned_output = axis_aligned_output
        self.remove_area_anomalies = remove_area_anomalies
        self.anomaly_sigma_threshold = anomaly_sigma_threshold
        self.anomaly_min_box_count = anomaly_min_box_count

    def _scale_boxes_to_original(
        self, boxes: np.ndarray, orig_size: Tuple[int, int]
    ) -> np.ndarray:
        if len(boxes) == 0:
            return boxes

        orig_h, orig_w = orig_size
        scale_x = orig_w / self.target_size
        scale_y = orig_h / self.target_size

        scaled = boxes.copy()
        scaled[:, 0:8:2] *= scale_x
        scaled[:, 1:8:2] *= scale_y
        return scaled

    def _convert_to_axis_aligned(self, quads: np.ndarray) -> np.ndarray:
        if len(quads) == 0:
            return quads
        aligned = quads.copy()
        coords = aligned[:, :8].reshape(-1, 4, 2)
        x_min = coords[:, :, 0].min(axis=1)
        x_max = coords[:, :, 0].max(axis=1)
        y_min = coords[:, :, 1].min(axis=1)
        y_max = coords[:, :, 1].max(axis=1)
        rects = np.stack(
            [
                x_min,
                y_min,
                x_max,
                y_min,
                x_max,
                y_max,
                x_min,
                y_max,
            ],
            axis=1,
        )
        aligned[:, :8] = rects.reshape(-1, 8)
        return aligned

    @staticmethod
    def _polygon_area_batch(polys: np.ndarray) -> np.ndarray:
        if polys.size == 0:
            return np.zeros((0,), dtype=np.float32)
        x = polys[:, :, 0]
        y = polys[:, :, 1]
        return 0.5 * np.abs(
            np.sum(x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1), axis=1)
        )

    def _is_quad_inside(self, inner: np.ndarray, outer: np.ndarray) -> bool:
        contour = outer.reshape(-1, 1, 2).astype(np.float32)
        for point in inner.astype(np.float32):
            if (
                cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False)
                < 0
            ):
                return False
        return True

    def _remove_fully_contained_boxes(self, quads: np.ndarray) -> np.ndarray:
        if len(quads) <= 1:
            return quads
        coords = quads[:, :8].reshape(-1, 4, 2)
        areas = self._polygon_area_batch(coords)
        keep = np.ones(len(quads), dtype=bool)
        order = np.argsort(areas)
        for idx in order:
            if not keep[idx]:
                continue
            inner = coords[idx]
            inner_area = areas[idx]
            for jdx in range(len(quads)):
                if idx == jdx or not keep[jdx]:
                    continue
                if areas[jdx] + 1e-6 < inner_area:
                    continue
                if self._is_quad_inside(inner, coords[jdx]):
                    keep[idx] = False
                    break
        return quads[keep]

    def _remove_area_anomalies(self, quads: np.ndarray) -> np.ndarray:
        if (
            not self.remove_area_anomalies
            or len(quads) == 0
            or len(quads) <= self.anomaly_min_box_count
        ):
            return quads
        coords = quads[:, :8].reshape(-1, 4, 2)
        areas = self._polygon_area_batch(coords).astype(np.float32)
        mean = float(np.mean(areas))
        std = float(np.std(areas))
        if std == 0.0:
            return quads
        threshold = mean + self.anomaly_sigma_threshold * std
        keep = areas <= threshold
        if not np.any(keep):
            return quads
        return quads[keep]

    def predict(
        self,
        img_or_path: Union[str, Path, np.ndarray],
        vis: bool = False,
        profile: bool = False,
        return_maps: bool = False,
        sort_reading_order: bool = True,
    ) -> Dict[str, Any]:
        """
        Run EAST inference and return detection results.

        Parameters
        ----------
        img_or_path : str or pathlib.Path or numpy.ndarray
            Path to an image file or an RGB image provided as a NumPy array
            with shape ``(H, W, 3)`` in ``uint8`` format.
        vis : bool, optional
            If True, a visualization image with rendered detections will be
            returned under the key ``"vis_image"``. Default is False.
        profile : bool, optional
            If True, prints timing information for the main inference stages.
            Default is False.
        return_maps : bool, optional
            If True, returns raw model score and geometry maps under keys
            ``"score_map"`` and ``"geo_map"``. Default is False.
        sort_reading_order : bool, optional
            If True, sorts detected words in natural reading order
            (left-to-right, top-to-bottom). Default is True.

        Returns
        -------
        dict
            Dictionary with the following keys:

            - ``"page"`` : Page
                Parsed detection result containing detected ``Word`` objects
                with polygon coordinates and confidence scores.
            - ``"vis_image"`` : PIL.Image.Image or None
                Visualization image with drawn bounding boxes if ``vis=True``,
                otherwise ``None``.
            - ``"score_map"`` : numpy.ndarray or None
                Raw score map produced by the network if ``return_maps=True``.
            - ``"geo_map"`` : numpy.ndarray or None
                Raw geometry map if ``return_maps=True``.

        Notes
        -----
        The method performs:
        (1) image loading, (2) resizing and normalization, (3) model inference,
        (4) quad decoding, (5) NMS, (6) box expansion, (7) scaling coordinates
        back to original size, and (8) optional visualization.

        Examples
        --------
        Perform inference with visualization:

        >>> from manuscript.detectors import EAST
        >>> model = EAST()
        >>> img_path = r"example/ocr_example_image.jpg"
        >>> result = model.predict(img_path, vis=True)
        >>> page = result["page"]
        >>> vis_img = result["vis_image"]
        >>> vis_img.show()

        """
        img = read_image(img_or_path)
        resized = cv2.resize(img, (self.target_size, self.target_size))

        t0 = time.time()

        img_norm = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
        img_input = img_norm.transpose(2, 0, 1)[np.newaxis, :, :, :]

        input_name = self.onnx_session.get_inputs()[0].name
        output_names = [out.name for out in self.onnx_session.get_outputs()]

        outputs = self.onnx_session.run(output_names, {input_name: img_input})

        score_map = outputs[0].squeeze(0).squeeze(0)
        geo_map = outputs[1].squeeze(0)

        if profile:
            print(f"  Model inference: {time.time() - t0:.3f}s")

        t0 = time.time()
        final_quads = decode_quads_from_maps(
            score_map=score_map,
            geo_map=geo_map.transpose(1, 2, 0),
            score_thresh=self.score_thresh,
            scale=1.0 / self.score_geo_scale,
            quantization=self.quantization,
            profile=profile,
        )
        if profile:
            print(f"  Decode boxes: {time.time() - t0:.3f}s")

        # 5) Apply NMS
        t0 = time.time()
        final_quads_nms = locality_aware_nms(
            final_quads, 
            iou_threshold=self.iou_threshold,
            iou_threshold_standard=self.iou_threshold_standard
        )
        if profile:
            print(f"  NMS: {time.time() - t0:.3f}s")
            print(f"    Boxes after NMS: {len(final_quads_nms)}")

        # 6) Expand (inverse shrink) with non-linear scaling
        final_quads_nms_expanded = expand_boxes(
            final_quads_nms,
            expand_w=self.expand_ratio_w,
            expand_h=self.expand_ratio_h,
            expand_power=self.expand_power,
        )

        # 7) Scale coordinates back to original image size
        orig_h, orig_w = img.shape[:2]
        scaled_quads = self._scale_boxes_to_original(
            final_quads_nms_expanded, (orig_h, orig_w)
        )

        processed_quads = self._remove_fully_contained_boxes(scaled_quads)
        processed_quads = self._remove_area_anomalies(processed_quads)
        output_quads = (
            self._convert_to_axis_aligned(processed_quads)
            if self.axis_aligned_output
            else processed_quads
        )

        # 8) Build Page with scaled coordinates (after NMS & expand)
        words: List[Word] = []
        for quad in output_quads:
            pts = quad[:8].reshape(4, 2)
            score = float(np.clip(quad[8], 0.0, 1.0))
            words.append(Word(polygon=pts.tolist(), detection_confidence=score))

        # 9) Optional sorting in reading order
        if sort_reading_order and len(words) > 0:
            # Convert to boxes (x_min, y_min, x_max, y_max) for sorting
            boxes = []
            for w in words:
                poly = np.array(w.polygon, dtype=np.int32)
                x_min, y_min = np.min(poly, axis=0)
                x_max, y_max = np.max(poly, axis=0)
                boxes.append((x_min, y_min, x_max, y_max))

            # Sort boxes in reading order
            sorted_boxes = sort_boxes_reading_order_with_resolutions(boxes)

            # Reorder words based on sorted boxes
            new_order = []
            for bx in sorted_boxes:
                for w in words:
                    poly = np.array(w.polygon, dtype=np.int32)
                    x_min, y_min = np.min(poly, axis=0)
                    x_max, y_max = np.max(poly, axis=0)
                    if (x_min, y_min, x_max, y_max) == bx:
                        new_order.append(w)
                        break
            words = new_order

        page = Page(blocks=[Block(words=words)])

        # 10) Optional visualization
        vis_img = visualize_page(img, page, show_order=False) if vis else None

        result: Dict[str, Any] = {
            "page": page,
            "vis_image": vis_img,
            "score_map": score_map if return_maps else None,
            "geo_map": geo_map if return_maps else None,
        }

        return result

    @staticmethod
    def train(
        train_images: Union[str, Path, Sequence[Union[str, Path]]],
        train_anns: Union[str, Path, Sequence[Union[str, Path]]],
        val_images: Union[str, Path, Sequence[Union[str, Path]]],
        val_anns: Union[str, Path, Sequence[Union[str, Path]]],
        *,
        experiment_root: str = "./experiments",
        model_name: str = "resnet_quad",
        backbone_name: str = "resnet50",
        pretrained_backbone: bool = True,
        freeze_first: bool = True,
        target_size: int = 1024,
        score_geo_scale: Optional[float] = None,
        epochs: int = 500,
        batch_size: int = 3,
        accumulation_steps: int = 1,
        lr: float = 1e-3,
        grad_clip: float = 5.0,
        early_stop: int = 100,
        use_sam: bool = True,
        sam_type: str = "asam",
        use_lookahead: bool = True,
        use_ema: bool = False,
        use_multiscale: bool = True,
        use_ohem: bool = True,
        ohem_ratio: float = 0.5,
        use_focal_geo: bool = True,
        focal_gamma: float = 2.0,
        resume_from: Optional[Union[str, Path]] = None,
        val_interval: int = 1,
        num_workers: int = 0,
        device: Optional[torch.device] = None,
    ) -> torch.nn.Module:
        """
        Train EAST model on custom datasets.

        Parameters
        ----------
        train_images : str, Path or sequence of paths
            Path(s) to training image folders.
        train_anns : str, Path or sequence of paths
            Path(s) to COCO-format JSON annotation files corresponding to
            ``train_images``.
        val_images : str, Path or sequence of paths
            Path(s) to validation image folders.
        val_anns : str, Path or sequence of paths
            Path(s) to COCO-format JSON annotation files corresponding to
            ``val_images``.
        experiment_root : str, optional
            Base directory where experiment folders will be created.
            Default is ``"./experiments"``.
        model_name : str, optional
            Folder name inside ``experiment_root`` for logs and checkpoints.
            Default is ``"resnet_quad"``.
        backbone_name : {"resnet50", "resnet101"}, optional
            Backbone architecture to use. Options:
            
            - ``"resnet50"`` — ResNet-50 (faster, less parameters)
            - ``"resnet101"`` — ResNet-101 (slower, more capacity)
            
            Default is ``"resnet50"``.
        pretrained_backbone : bool, optional
            Use ImageNet-pretrained backbone weights. Default ``True``.
        freeze_first : bool, optional
            Freeze lowest layers of the backbone. Default ``True``.
        target_size : int, optional
            Resize shorter side of images to this size. Default ``1024``.
        score_geo_scale : float, optional
            Multiplier to recover original coordinates from score/geo maps.
            If None, automatically taken from the model. Default ``None``.
        epochs : int, optional
            Number of training epochs. Default ``500``.
        batch_size : int, optional
            Batch size per GPU. Default ``3``.
        accumulation_steps : int, optional
            Number of gradient accumulation steps. Effective batch size will be
            ``batch_size * accumulation_steps``. Use this to train with larger
            effective batch sizes when GPU memory is limited. For example:
            
            - ``batch_size=2, accumulation_steps=4`` → effective batch size = 8
            - ``batch_size=1, accumulation_steps=8`` → effective batch size = 8
            
            Default is ``1`` (no accumulation).
        lr : float, optional
            Learning rate. Default ``1e-3``.
        grad_clip : float, optional
            Gradient clipping value (L2 norm). Default ``5.0``.
        early_stop : int, optional
            Patience (epochs without improvement) for early stopping.
            Default ``100``.
        use_sam : bool, optional
            Enable SAM optimizer. Default ``True``.
        sam_type : {"sam", "asam"}, optional
            Variant of SAM to use. Default ``"asam"``.
        use_lookahead : bool, optional
            Wrap optimizer with Lookahead. Default ``True``.
        use_ema : bool, optional
            Maintain EMA version of model weights. Default ``False``.
        use_multiscale : bool, optional
            Random multi-scale training. Default ``True``.
        use_ohem : bool, optional
            Online Hard Example Mining. Default ``True``.
        ohem_ratio : float, optional
            Ratio of hard negatives for OHEM. Default ``0.5``.
        use_focal_geo : bool, optional
            Apply focal loss to geometry channels. Default ``True``.
        focal_gamma : float, optional
            Gamma for focal geometry loss. Default ``2.0``.
        resume_from : str or Path, optional
            Resume training from a previous experiment:
            a) experiment directory,
            b) `.../checkpoints/`,
            c) direct path to `last_state.pt`.
            Default ``None``.
        val_interval : int, optional
            Run validation every N epochs. Default ``1``.
        num_workers : int, optional
            Number of worker processes for data loading. Set to 0 for single-process
            loading (safer on Windows). Default ``0``.
        device : torch.device, optional
            CUDA or CPU device. Auto-selects if None.

        Returns
        -------
        torch.nn.Module
            Best model weights (EMA if enabled, otherwise base model).

        Examples
        --------
        Train on two datasets with validation:

        >>> from manuscript.detectors import EAST
        >>>
        >>> train_images = [
        ...     "/data/archive/train_images",
        ...     "/data/ddi/train_images"
        ... ]
        >>> train_anns = [
        ...     "/data/archive/train.json",
        ...     "/data/ddi/train.json"
        ... ]
        >>> val_images = [
        ...     "/data/archive/test_images",
        ...     "/data/ddi/test_images"
        ... ]
        >>> val_anns = [
        ...     "/data/archive/test.json",
        ...     "/data/ddi/test.json"
        ... ]
        >>>
        >>> best_model = EAST.train(
        ...     train_images=train_images,
        ...     train_anns=train_anns,
        ...     val_images=val_images,
        ...     val_anns=val_anns,
        ...     backbone_name="resnet50",
        ...     target_size=256,
        ...     epochs=20,
        ...     batch_size=4,
        ...     use_sam=False,
        ...     freeze_first=False,
        ...     val_interval=3,
        ... )
        >>> print("Best checkpoint loaded:", best_model)
        """

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = EASTModel(
            backbone_name=backbone_name,
            pretrained_backbone=pretrained_backbone,
            freeze_first=freeze_first,
        ).to(device)

        if score_geo_scale is None:
            score_geo_scale = model.score_scale

        def make_dataset(imgs, anns, name: Optional[str] = None):
            return EASTDataset(
                images_folder=imgs,
                coco_annotation_file=anns,
                target_size=target_size,
                score_geo_scale=score_geo_scale,
                dataset_name=name,
            )

        def _dataset_base_name(
            img_path: Union[str, Path], ann_path: Union[str, Path]
        ) -> str:
            ann = Path(os.fspath(ann_path))
            parts: List[str] = []
            if ann.parent.name:
                parts.append(ann.parent.name)
            if ann.stem:
                parts.append(ann.stem)
            if not parts:
                img = Path(os.fspath(img_path))
                if img.parent.name:
                    parts.append(img.parent.name)
                stem = img.stem or img.name
                if stem:
                    parts.append(stem)
            return "/".join(parts)

        def _unique_dataset_name(
            img_path: Union[str, Path],
            ann_path: Union[str, Path],
            counts: Dict[str, int],
            idx: int,
            kind: str,
        ) -> str:
            base = _dataset_base_name(img_path, ann_path)
            if not base:
                base = f"{kind}_{idx}"
            count = counts.get(base, 0)
            counts[base] = count + 1
            if count == 0:
                return base
            return f"{base}_{count + 1}"

        train_imgs_list = (
            train_images if isinstance(train_images, (list, tuple)) else [train_images]
        )
        train_anns_list = (
            train_anns if isinstance(train_anns, (list, tuple)) else [train_anns]
        )
        val_imgs_list = (
            val_images if isinstance(val_images, (list, tuple)) else [val_images]
        )
        val_anns_list = val_anns if isinstance(val_anns, (list, tuple)) else [val_anns]
        assert len(train_imgs_list) == len(
            train_anns_list
        ), "train_images and train_anns must have the same length"
        assert len(val_imgs_list) == len(
            val_anns_list
        ), "val_images and val_anns must have the same length"

        train_datasets = []
        train_name_counts: Dict[str, int] = {}
        for idx, (imgs, anns) in enumerate(
            zip(train_imgs_list, train_anns_list), start=1
        ):
            dataset_name = _unique_dataset_name(
                imgs, anns, train_name_counts, idx=idx, kind="train"
            )
            train_datasets.append(make_dataset(imgs, anns, name=dataset_name))

        val_datasets = []
        val_name_counts: Dict[str, int] = {}
        for idx, (imgs, anns) in enumerate(zip(val_imgs_list, val_anns_list), start=1):
            dataset_name = _unique_dataset_name(
                imgs, anns, val_name_counts, idx=idx, kind="val"
            )
            val_datasets.append(make_dataset(imgs, anns, name=dataset_name))

        train_ds = ConcatDataset(train_datasets)
        val_ds = ConcatDataset(val_datasets)
        val_dataset_names = [ds.dataset_name for ds in val_datasets]

        def _resolve_path(path: Union[str, Path]) -> Path:
            p = Path(path)
            if p.is_absolute():
                return p
            project_root = Path(__file__).resolve().parents[4]
            candidate = (project_root / p).resolve()
            if candidate.exists():
                return candidate
            return (Path.cwd() / p).resolve()

        def _resolve_resume_target(
            target: Union[str, Path],
        ) -> Tuple[str, Optional[Path]]:
            resolved = _resolve_path(target)
            if not resolved.exists():
                raise FileNotFoundError(
                    f"resume_from target does not exist: {resolved}"
                )

            if resolved.is_file():
                resume_state = resolved
                checkpoints_dir = resolved.parent
                if checkpoints_dir.name == "checkpoints":
                    experiment_dir = checkpoints_dir.parent
                else:
                    experiment_dir = checkpoints_dir
                return os.path.abspath(os.fspath(experiment_dir)), resume_state

            experiment_dir = resolved
            checkpoints_dir = (
                resolved if resolved.name == "checkpoints" else resolved / "checkpoints"
            )
            default_state = checkpoints_dir / "last_state.pt"
            resume_state = default_state if default_state.exists() else None
            return os.path.abspath(os.fspath(experiment_dir)), resume_state

        resume_state_path: Optional[Path] = None
        if resume_from is None:
            experiment_dir = os.path.abspath(os.path.join(experiment_root, model_name))
            resume_flag = False
        else:
            experiment_dir, resume_state_path = _resolve_resume_target(resume_from)
            resume_flag = True

        best_model = _run_training(
            experiment_dir=experiment_dir,
            model=model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            device=device,
            num_epochs=epochs,
            batch_size=batch_size,
            accumulation_steps=accumulation_steps,
            lr=lr,
            grad_clip=grad_clip,
            early_stop=early_stop,
            use_sam=use_sam,
            sam_type=sam_type,
            use_lookahead=use_lookahead,
            use_ema=use_ema,
            use_multiscale=use_multiscale,
            use_ohem=use_ohem,
            ohem_ratio=ohem_ratio,
            use_focal_geo=use_focal_geo,
            focal_gamma=focal_gamma,
            val_interval=val_interval,
            num_workers=num_workers,
            backbone_name=backbone_name,
            target_size=target_size,
            pretrained_backbone=pretrained_backbone,
            val_datasets=val_datasets,
            val_dataset_names=val_dataset_names,
            resume=resume_flag,
            resume_state_path=(
                os.fspath(resume_state_path) if resume_state_path else None
            ),
        )
        return best_model

    @staticmethod
    def export_to_onnx(
        weights_path: Union[str, Path],
        output_path: Union[str, Path],
        backbone_name: str = None,
        input_size: int = 1280,
        opset_version: int = 14,
        simplify: bool = True,
    ) -> None:
        """
        Export EAST PyTorch model to ONNX format.

        This method converts a trained EAST model from PyTorch to ONNX format,
        which can be used for faster inference with ONNX Runtime. The exported
        model can be loaded using ``EAST(weights_path="model.onnx", use_onnx=True)``.

        Parameters
        ----------
        weights_path : str or Path
            Path to the PyTorch model weights file (.pth).
        output_path : str or Path
            Path where the ONNX model will be saved (.onnx).
        backbone_name : {"resnet50", "resnet101"}, optional
            Backbone architecture of the model. If None, will be automatically
            detected from the checkpoint. Must match the architecture used during
            training. Default is None (auto-detect).
        input_size : int, optional
            Input image size (height and width). The model will accept
            images of shape ``(batch, 3, input_size, input_size)``.
            Default is 1280.
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
            If ``weights_path`` does not exist.
        ValueError
            If backbone_name doesn't match the checkpoint architecture.

        Notes
        -----
        The exported ONNX model has two outputs:

        - ``score_map``: Text confidence map with shape ``(batch, 1, H, W)``
        - ``geo_map``: Geometry map with shape ``(batch, 8, H, W)``

        The model supports dynamic batch size and image dimensions through
        dynamic axes configuration.

        **Automatic Backbone Detection:**
        
        The method automatically detects the backbone architecture from the checkpoint
        by analyzing the number of parameters in layer4. This prevents mismatches between
        checkpoint and architecture that could lead to incorrect exports.

        Examples
        --------
        Export with automatic backbone detection:

        >>> from manuscript.detectors import EAST
        >>> EAST.export_to_onnx(
        ...     weights_path="east_resnet50.pth",
        ...     output_path="east_model.onnx"
        ... )
        Auto-detected backbone: resnet50
        Exporting to ONNX (opset 14)...
        [OK] ONNX model saved to: east_model.onnx

        Export with explicit backbone:

        >>> EAST.export_to_onnx(
        ...     weights_path="custom_weights.pth",
        ...     output_path="custom_model.onnx",
        ...     backbone_name="resnet101",
        ...     input_size=1024,
        ...     simplify=False
        ... )

        Use the exported model for inference:

        >>> detector = EAST(
        ...     weights_path="east_model.onnx",
        ...     use_onnx=True,
        ...     device="cuda"
        ... )
        >>> result = detector.predict("image.jpg")

        See Also
        --------
        EAST.__init__ : Initialize EAST detector with ONNX support using ``use_onnx=True``.
        """

        class EASTWrapper(torch.nn.Module):
            def __init__(self, east_model):
                super().__init__()
                self.east = east_model

            def forward(self, x):
                output = self.east(x)
                return output["score"], output["geometry"]

        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        print(f"Loading checkpoint from {weights_path}...")
        checkpoint = torch.load(str(weights_path), map_location="cpu")
        
        # Auto-detect backbone if not specified
        if backbone_name is None:
            print("Auto-detecting backbone architecture...")
            # Detect by checking layer3 parameters count
            # ResNet50: layer3 has 6 bottleneck blocks (layer3.0 - layer3.5)
            # ResNet101: layer3 has 23 bottleneck blocks (layer3.0 - layer3.22)
            
            # Count backbone.extractor.* parameters starting with "layer3"
            layer3_keys = [k for k in checkpoint.keys() if "backbone.extractor.layer3" in k]
            
            # Check if layer3.10 exists (only in resnet101)
            has_layer3_10 = any("layer3.10" in k for k in layer3_keys)
            
            if has_layer3_10:
                detected_backbone = "resnet101"
            else:
                detected_backbone = "resnet50"
            
            print(f"   Detected {len(layer3_keys)} layer3 parameters")
            print(f"   Auto-detected backbone: {detected_backbone}")
            backbone_name = detected_backbone
        else:
            print(f"Using specified backbone: {backbone_name}")
            
            # Verify backbone matches checkpoint
            print("Verifying backbone matches checkpoint...")
            layer3_keys = [k for k in checkpoint.keys() if "backbone.extractor.layer3" in k]
            has_layer3_10 = any("layer3.10" in k for k in layer3_keys)
            
            expected_backbone = "resnet101" if has_layer3_10 else "resnet50"
            
            if expected_backbone != backbone_name:
                raise ValueError(
                    f"Backbone mismatch! Checkpoint is {expected_backbone}, "
                    f"but you specified {backbone_name}. "
                    f"Either use backbone_name='{expected_backbone}' or set backbone_name=None for auto-detection."
                )
            print(f"   [OK] Backbone matches checkpoint ({backbone_name})")

        print(f"\nLoading PyTorch model...")
        east_model = EASTModel(
            backbone_name=backbone_name,
            pretrained_backbone=False,
            pretrained_model_path=str(weights_path),
        )
        east_model.eval()

        model = EASTWrapper(east_model)
        model.eval()

        dummy_input = torch.randn(1, 3, input_size, input_size)

        print(f"Model architecture: {model.__class__.__name__}")
        print(f"Input shape: {dummy_input.shape}")

        with torch.no_grad():
            score_map, geo_map = model(dummy_input)

        print(f"Output shapes:")
        print(f"  - score_map: {score_map.shape}")
        print(f"  - geo_map: {geo_map.shape}")

        print(f"\nExporting to ONNX (opset {opset_version})...")
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["score_map", "geo_map"],
            dynamic_axes={
                "input": {0: "batch_size", 2: "height", 3: "width"},
                "score_map": {0: "batch_size", 2: "height", 3: "width"},
                "geo_map": {0: "batch_size", 2: "height", 3: "width"},
            },
            verbose=False,
        )

        print(f"[OK] ONNX model saved to: {output_path}")

        import onnx
        import onnxsim

        print("\nVerifying ONNX model...")
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("[OK] ONNX model is valid")

        if simplify:
            print("\nSimplifying ONNX model...")
            model_simplified, check = onnxsim.simplify(onnx_model)
            if check:
                onnx.save(model_simplified, str(output_path))
                print("[OK] ONNX model simplified")
            else:
                print("[WARNING] Simplification failed, using original model")

        # Test ONNX inference and compare with PyTorch
        try:
            import onnxruntime as ort
            
            print(f"\nTesting ONNX inference...")
            session = ort.InferenceSession(
                str(output_path),
                providers=["CPUExecutionProvider"]
            )

            ort_inputs = {"input": dummy_input.numpy()}
            ort_outputs = session.run(None, ort_inputs)

            print(f"[OK] ONNX inference works!")
            print(f"  ONNX score_map shape: {ort_outputs[0].shape}")
            print(f"  ONNX geo_map shape: {ort_outputs[1].shape}")

            # Compare with PyTorch
            print(f"\nComparing ONNX vs PyTorch outputs...")
            torch_score = score_map.numpy()
            torch_geo = geo_map.numpy()
            onnx_score = ort_outputs[0]
            onnx_geo = ort_outputs[1]

            score_max_diff = abs(torch_score - onnx_score).max()
            geo_max_diff = abs(torch_geo - onnx_geo).max()
            
            print(f"  Max difference in score_map: {score_max_diff:.6f}")
            print(f"  Max difference in geo_map: {geo_max_diff:.6f}")

            if score_max_diff < 1e-4 and geo_max_diff < 1e-4:
                print(f"  [OK] Outputs match!")
            elif score_max_diff < 1e-3 and geo_max_diff < 1e-3:
                print(f"  [WARNING] Small differences detected (acceptable)")
            else:
                print(f"  [ERROR] Outputs differ significantly!")
                print(f"  This may indicate a problem with the export.")

        except ImportError:
            print(f"[WARNING] onnxruntime not installed, skipping inference test")
            print(f"  Install with: pip install onnxruntime")
        except Exception as e:
            print(f"[WARNING] ONNX inference test failed: {e}")

        file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"\n[OK] Export complete! Model size: {file_size_mb:.1f} MB")
        print(f"\n=== Summary ===")
        print(f"Backbone: {backbone_name}")
        print(f"Input shape: [batch_size, 3, {input_size}, {input_size}]")
        print(f"Output shapes:")
        print(f"  - score_map: [batch_size, 1, H, W]")
        print(f"  - geo_map: [batch_size, 8, H, W]")

