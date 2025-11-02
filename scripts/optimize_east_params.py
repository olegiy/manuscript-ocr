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
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import optuna
from optuna.samplers import TPESampler

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from manuscript.detectors import EAST
from manuscript.detectors._east.utils import evaluate_dataset


# ============================================================================
# ГЛОБАЛЬНЫЕ НАСТРОЙКИ ОПТИМИЗАЦИИ
# ============================================================================
# Определяет какие параметры оптимизируются, а какие фиксированы

OPTIMIZATION_CONFIG = {
    # Параметры для оптимизации (True = оптимизируется, False = фиксированное значение)
    "optimize": {
        "expand_ratio_w": False,
        "expand_ratio_h": False,
        "expand_power": False,
        "score_thresh": False,
        "iou_threshold": True,
        "quantization": False,
        "remove_area_anomalies": False,  # Фиксируем как False
    },
    # Фиксированные значения (используются когда optimize = False)
    "fixed_values": {
        "expand_ratio_w": 1.7,
        "expand_ratio_h": 1.7,
        "expand_power": 0.5,
        "score_thresh": 0.7,
        "iou_threshold": 0.2,
        "quantization": 2,
        "remove_area_anomalies": False,
    },
    # Диапазоны для оптимизации (используются когда optimize = True)
    "ranges": {
        "expand_ratio_w": (0.3, 2.0, 0.1),  # (min, max, step)
        "expand_ratio_h": (0.3, 2.0, 0.1),
        "expand_power": (0.1, 2.0, 0.1),
        "score_thresh": (0.5, 0.8, 0.1),
        "iou_threshold": (0.05, 0.6, 0.05),
        "quantization": [1, 2, 4],  # categorical
        "remove_area_anomalies": [True, False],  # categorical
    },
}


# Глобальный кеш для хранения результатов уже вычисленных параметров
PARAMS_CACHE: Dict[str, float] = {}


def params_to_hash(params: Dict[str, Any]) -> str:
    """
    Создает хеш из параметров для кеширования.

    Parameters
    ----------
    params : dict
        Словарь параметров

    Returns
    -------
    str
        MD5 хеш параметров
    """
    # Сортируем ключи для консистентности
    sorted_params = {k: params[k] for k in sorted(params.keys())}
    params_str = json.dumps(sorted_params, sort_keys=True)
    return hashlib.md5(params_str.encode()).hexdigest()


def get_cached_score(params: Dict[str, Any]) -> Optional[float]:
    """
    Получает закешированный score для параметров.

    Parameters
    ----------
    params : dict
        Словарь параметров

    Returns
    -------
    float or None
        Закешированный score или None если не найден
    """
    params_hash = params_to_hash(params)
    return PARAMS_CACHE.get(params_hash)


def cache_score(params: Dict[str, Any], score: float) -> None:
    """
    Кеширует score для параметров.

    Parameters
    ----------
    params : dict
        Словарь параметров
    score : float
        Полученный score
    """
    params_hash = params_to_hash(params)
    PARAMS_CACHE[params_hash] = score


def save_cache(cache_file: str) -> None:
    """
    Сохраняет кеш в файл.

    Parameters
    ----------
    cache_file : str
        Путь к файлу для сохранения кеша
    """
    cache_data = {
        "cache": PARAMS_CACHE,
        "config": OPTIMIZATION_CONFIG,
    }

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, indent=2)

    print(f"Cache saved to {cache_file} ({len(PARAMS_CACHE)} entries)")


def load_cache(cache_file: str) -> int:
    """
    Загружает кеш из файла.

    Parameters
    ----------
    cache_file : str
        Путь к файлу кеша

    Returns
    -------
    int
        Количество загруженных записей
    """
    global PARAMS_CACHE

    if not Path(cache_file).exists():
        print(f"Cache file not found: {cache_file}")
        return 0

    with open(cache_file, "r", encoding="utf-8") as f:
        cache_data = json.load(f)

    PARAMS_CACHE = cache_data.get("cache", {})

    print(f"Cache loaded from {cache_file} ({len(PARAMS_CACHE)} entries)")

    # Проверяем совместимость конфигурации
    saved_config = cache_data.get("config")
    if saved_config and saved_config != OPTIMIZATION_CONFIG:
        print("WARNING: Cached configuration differs from current OPTIMIZATION_CONFIG!")
        print("Consider clearing cache if optimization ranges have changed.")

    return len(PARAMS_CACHE)


# ============================================================================
# ОСНОВНЫЕ ФУНКЦИИ
# ============================================================================


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
        expand_power=params["expand_power"],
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
    metrics = evaluate_dataset(predictions, gt_subset, verbose=False, n_jobs=1)

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
        # Собираем параметры: оптимизируемые + фиксированные
        params = {}

        for param_name, should_optimize in OPTIMIZATION_CONFIG["optimize"].items():
            if should_optimize:
                # Оптимизируем параметр
                param_range = OPTIMIZATION_CONFIG["ranges"][param_name]

                if isinstance(param_range, list):
                    # Categorical параметр
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_range
                    )
                else:
                    # Float параметр с диапазоном
                    min_val, max_val, step = param_range
                    params[param_name] = trial.suggest_float(
                        param_name, min_val, max_val, step=step
                    )
            else:
                # Используем фиксированное значение
                params[param_name] = OPTIMIZATION_CONFIG["fixed_values"][param_name]

        # Проверяем кеш
        cached_score = get_cached_score(params)
        if cached_score is not None:
            print(
                f"\n  [CACHE HIT] Trial {trial.number}: Using cached score {cached_score:.4f}"
            )
            print(f"    Params: {params}")
            return cached_score

        # Вычисляем score
        try:
            score = evaluate_params(
                image_files, ground_truths, params, device, target_size
            )

            # Кешируем результат
            cache_score(params, score)

            print(f"\n  [COMPUTED] Trial {trial.number}: F1@0.5:0.95 = {score:.4f}")
            print(f"    Params: {params}")

            return score
        except Exception as e:
            print(f"\n  [ERROR] Trial {trial.number}: {e}")
            print(f"    Params: {params}")
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
    print("=" * 80)
    print("OPTIMIZATION CONFIGURATION")
    print("=" * 80)
    print("\nOptimized parameters:")
    for param, is_opt in OPTIMIZATION_CONFIG["optimize"].items():
        if is_opt:
            range_val = OPTIMIZATION_CONFIG["ranges"][param]
            print(f"  ✓ {param:25s}: {range_val}")

    print("\nFixed parameters:")
    for param, is_opt in OPTIMIZATION_CONFIG["optimize"].items():
        if not is_opt:
            fixed_val = OPTIMIZATION_CONFIG["fixed_values"][param]
            print(f"  • {param:25s}: {fixed_val}")

    print("\n" + "=" * 80)
    print(f"Starting Optuna optimization with {n_trials} trials")
    print(f"Testing on {len(image_files)} images")
    print(f"Images with annotations: {len(ground_truths)}")
    print(f"Cache size at start: {len(PARAMS_CACHE)} entries")
    print("=" * 80 + "\n")

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
    cache_hits = 0
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            # Проверяем был ли это cache hit
            trial_params = {k: v for k, v in trial.params.items()}
            # Добавляем фиксированные параметры
            for param_name, should_optimize in OPTIMIZATION_CONFIG["optimize"].items():
                if not should_optimize:
                    trial_params[param_name] = OPTIMIZATION_CONFIG["fixed_values"][
                        param_name
                    ]

            history.append(
                {
                    "trial_number": trial.number,
                    "params": trial.params,
                    "all_params": trial_params,
                    "f1_score": trial.value,
                }
            )

    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)
    print(f"\nTotal trials: {len(study.trials)}")
    print(f"Cache entries created: {len(PARAMS_CACHE)}")
    print(
        f"Cache hit rate: {(len(study.trials) - len(PARAMS_CACHE)) / len(study.trials) * 100:.1f}%"
    )
    print(f"\nBest F1@0.5:0.95: {best_score:.4f}")
    print(f"\nBest parameters:")
    for param, value in best_params.items():
        print(f"  {param:25s}: {value}")

    # Показываем также фиксированные параметры
    print(f"\nFixed parameters (not optimized):")
    for param, is_opt in OPTIMIZATION_CONFIG["optimize"].items():
        if not is_opt:
            fixed_val = OPTIMIZATION_CONFIG["fixed_values"][param]
            print(f"  {param:25s}: {fixed_val}")

    print("=" * 80 + "\n")

    return {
        "best_params": best_params,
        "best_score": best_score,
        "history": history,
        "n_trials": len(study.trials),
        "cache_size": len(PARAMS_CACHE),
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
        "--cache",
        type=str,
        default=None,
        help="Cache file for storing/loading computed scores (optional)",
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

    # Загружаем кеш если указан
    if args.cache:
        load_cache(args.cache)

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

    # Сохраняем кеш если указан
    if args.cache:
        save_cache(args.cache)

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
