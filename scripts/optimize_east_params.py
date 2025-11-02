"""
Оптимизация параметров EAST детектора с помощью Optuna.

Использует Bayesian optimization для быстрого поиска оптимальных параметров
детектора с целью максимизации F1@0.5:0.95 метрики.

Usage:
    python scripts/optimize_east_params.py \
        --folder path/to/images \
        --annotations path/to/annotations.json \
        --sample-ratio 0.3 \
        --ntrials 50

    python scripts/optimize_east_params.py --folder "C:\shared\data02065\Archives020525\test_images" --annotations "C:\shared\data02065\Archives020525\test.json"--sample-ratio 0.01 -ntrials 100 --output results.json
"""

import argparse
import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
import optuna
from optuna.samplers import TPESampler

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from manuscript.detectors import EAST
from manuscript.detectors._east.utils import evaluate_dataset


def get_image_files(folder: str) -> List[str]:
    """Находит все изображения в папке."""
    folder_path = Path(folder)
    if not folder_path.exists():
        raise ValueError(f"Folder not found: {folder}")

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_files = set()

    for ext in extensions:
        image_files.update(folder_path.glob(f"*{ext}"))
        image_files.update(folder_path.glob(f"*{ext.upper()}"))

    return [str(f) for f in sorted(image_files)]


def load_ground_truth(annotation_file: str, images_folder: str) -> Dict[str, List]:
    """Загружает ground truth аннотации из COCO формата."""
    with open(annotation_file, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    images_info = {img["id"]: img for img in coco_data["images"]}
    ground_truths = {}

    for ann in coco_data["annotations"]:
        image_id = ann["image_id"]
        if image_id not in images_info:
            continue

        filename = images_info[image_id]["file_name"]
        seg = ann.get("segmentation")
        if not seg or len(seg) == 0:
            continue

        seg_parts = seg if isinstance(seg[0], list) else [seg]

        for seg_poly in seg_parts:
            if len(seg_poly) < 8:
                continue

            pts = np.array(seg_poly, dtype=np.float32).reshape(-1, 2)
            x_min = float(np.min(pts[:, 0]))
            y_min = float(np.min(pts[:, 1]))
            x_max = float(np.max(pts[:, 0]))
            y_max = float(np.max(pts[:, 1]))

            box = (x_min, y_min, x_max, y_max)

            if filename not in ground_truths:
                ground_truths[filename] = []
            ground_truths[filename].append(box)

    return ground_truths


def evaluate_params(
    image_files: List[str],
    ground_truths: Dict[str, List],
    params: Dict[str, Any],
    device: str = "cpu",
    target_size: int = 1280,
) -> float:
    """
    Оценивает параметры детектора на выборке изображений.

    Returns
    -------
    float
        F1@0.5:0.95 метрика
    """
    # Создаём детектор с заданными параметрами
    detector = EAST(
        device=device,
        target_size=target_size,
        expand_ratio_w=params["expand_ratio_w"],
        expand_ratio_h=params["expand_ratio_h"],
        score_thresh=params["score_thresh"],
        iou_threshold=params["iou_threshold"],
        quantization=params["quantization"],
        remove_area_anomalies=params["remove_area_anomalies"],
    )

    # Собираем предсказания
    predictions = {}
    gt_subset = {}  # Ground truth только для обрабатываемых изображений

    for img_path in image_files:
        filename = Path(img_path).name

        # Пропускаем изображения без аннотаций
        if filename not in ground_truths:
            continue

        result = detector.predict(img_path, vis=False, profile=False)

        boxes = []
        for block in result["page"].blocks:
            for word in block.words:
                if word.polygon and len(word.polygon) > 0:
                    xs = [pt[0] for pt in word.polygon]
                    ys = [pt[1] for pt in word.polygon]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    boxes.append((x_min, y_min, x_max, y_max))

        predictions[filename] = boxes
        gt_subset[filename] = ground_truths[filename]

    # Оцениваем метрики только на обработанных изображениях
    # Используем параллельную обработку для ускорения (n_jobs=None = все CPU)
    metrics = evaluate_dataset(predictions, gt_subset, verbose=False, n_jobs=None)

    return metrics["f1@0.5:0.95"]


def create_objective(
    image_files: List[str],
    ground_truths: Dict[str, List],
    device: str = "cpu",
    target_size: int = 1280,
):
    """
    Создает objective функцию для Optuna.

    Returns
    -------
    callable
        Objective функция для оптимизации
    """

    def objective(trial: optuna.Trial) -> float:
        """Objective функция для Optuna optimization."""
        # Предлагаем параметры
        params = {
            "expand_ratio_w": trial.suggest_float("expand_ratio_w", 0.7, 1.0, step=0.1),
            "expand_ratio_h": trial.suggest_float("expand_ratio_h", 0.7, 1.0, step=0.1),
            "score_thresh": trial.suggest_float("score_thresh", 0.5, 0.8, step=0.1),
            "iou_threshold": trial.suggest_float("iou_threshold", 0.1, 0.3, step=0.1),
            "quantization": trial.suggest_categorical("quantization", [1, 2, 4]),
            "remove_area_anomalies": trial.suggest_categorical(
                "remove_area_anomalies", [True, False]
            ),
        }

        try:
            score = evaluate_params(
                image_files, ground_truths, params, device, target_size
            )
            return score
        except Exception as e:
            print(f"\nError with params {params}: {e}")
            return 0.0

    return objective


def optimize_with_optuna(
    image_files: List[str],
    ground_truths: Dict[str, List],
    n_trials: int = 50,
    device: str = "cpu",
    target_size: int = 1280,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Выполняет оптимизацию параметров с помощью Optuna.

    Returns
    -------
    dict
        Результаты: лучшие параметры, лучшая метрика, история всех trials
    """
    print(f"Starting Optuna optimization with {n_trials} trials")
    print(f"Testing on {len(image_files)} images")
    print(f"Images with annotations: {len(ground_truths)}")
    print()

    # Создаем study с TPE sampler
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(
        direction="maximize", sampler=sampler, study_name="east_optimization"
    )

    # Создаем objective функцию
    objective = create_objective(image_files, ground_truths, device, target_size)

    # Запускаем оптимизацию
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Собираем результаты
    best_trial = study.best_trial
    best_params = best_trial.params
    best_score = best_trial.value

    # История всех trials
    history = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            history.append(
                {
                    "trial_number": trial.number,
                    "params": trial.params,
                    "f1_score": trial.value,
                }
            )

    return {
        "best_params": best_params,
        "best_score": best_score,
        "history": history,
        "n_trials": len(study.trials),
        "study": study,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Optimize EAST detector parameters using Optuna (Bayesian Optimization)"
    )
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Path to folder with test images",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        required=True,
        help="Path to COCO format annotations JSON file",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=0.3,
        help="Ratio of images to use for optimization (0.0-1.0, default: 0.3)",
    )
    parser.add_argument(
        "--ntrials",
        type=int,
        default=50,
        help="Number of optimization trials (default: 50)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=1280,
        help="Target image size (default: 1280)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use (default: cpu)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="optimization_results.json",
        help="Save results to JSON file (default: optimization_results.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Устанавливаем seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Проверяем папку
    if not Path(args.folder).exists():
        print(f"Error: Folder not found: {args.folder}")
        sys.exit(1)

    if not Path(args.annotations).exists():
        print(f"Error: Annotations file not found: {args.annotations}")
        sys.exit(1)

    # Загружаем изображения
    print("Loading images...")
    all_image_files = get_image_files(args.folder)
    print(f"Found {len(all_image_files)} total images")

    # Загружаем ground truth
    print(f"Loading ground truth annotations...")
    ground_truths = load_ground_truth(args.annotations, args.folder)
    print(f"Loaded annotations for {len(ground_truths)} images")

    # Фильтруем только изображения с аннотациями
    image_files_with_gt = [
        img for img in all_image_files if Path(img).name in ground_truths
    ]
    print(f"Images with annotations: {len(image_files_with_gt)}")

    # Семплируем изображения
    sample_size = int(len(image_files_with_gt) * args.sample_ratio)
    sample_size = max(1, min(sample_size, len(image_files_with_gt)))

    sampled_images = random.sample(image_files_with_gt, sample_size)
    print(f"Using {sample_size} images for optimization (ratio: {args.sample_ratio})")
    print()

    # Выполняем оптимизацию с Optuna
    results = optimize_with_optuna(
        sampled_images,
        ground_truths,
        n_trials=args.ntrials,
        device=args.device,
        target_size=args.target_size,
        seed=args.seed,
    )

    # Выводим результаты
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"\nBest F1@0.5:0.95 Score: {results['best_score']:.4f}")
    print(f"Found after {results['n_trials']} trials")
    print("\nBest Parameters:")
    for param, value in results["best_params"].items():
        print(f"  {param}: {value}")

    # Топ-5 лучших комбинаций
    sorted_history = sorted(
        results["history"], key=lambda x: x["f1_score"], reverse=True
    )

    print("\nTop 5 parameter combinations:")
    for i, result in enumerate(sorted_history[:5], 1):
        print(
            f"\n{i}. F1@0.5:0.95 = {result['f1_score']:.4f} (Trial #{result['trial_number']})"
        )
        for param, value in result["params"].items():
            print(f"   {param}: {value}")

    # Сохраняем результаты
    output_data = {
        "best_params": results["best_params"],
        "best_score": float(results["best_score"]),
        "optimization_settings": {
            "method": "Optuna (TPE Sampler)",
            "folder": args.folder,
            "annotations": args.annotations,
            "sample_ratio": args.sample_ratio,
            "n_trials": args.ntrials,
            "target_size": args.target_size,
            "device": args.device,
            "seed": args.seed,
            "total_images": len(all_image_files),
            "images_with_annotations": len(image_files_with_gt),
            "sampled_images": sample_size,
        },
        "parameter_ranges": {
            "expand_ratio_w": [0.7, 1.3],
            "expand_ratio_h": [0.7, 1.3],
            "score_thresh": [0.5, 0.95],
            "iou_threshold": [0.1, 0.3],
            "quantization": [1, 2, 4],
            "remove_area_anomalies": [True, False],
        },
        "n_trials_completed": results["n_trials"],
        "history": sorted_history,  # Сохраняем отсортированную историю
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Results saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
