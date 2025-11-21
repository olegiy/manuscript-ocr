"""
Бенчмарк производительности и точности EAST детектора на CPU и GPU.

Измеряет:
- Среднее время инференса на изображение
- Использование памяти (RAM для CPU, VRAM для GPU)
- Throughput (изображений в секунду)
- Точность детекции (F1@0.5, F1@0.5:0.95) при наличии аннотаций

Usage:
    # Только скорость
    python scripts/east_infer_speed_test.py --folder path/to/images

    # Скорость + точность
    python scripts/east_infer_speed_test.py --folder path/to/images --annotations path/to/annotations.json
    python scripts/east_infer_speed_test.py --folder "C:\shared\data0205\data02065\Archives020525\test_images" --annotations "C:\shared\data0205\data02065\Archives020525\test.json" --gpu-only
    python scripts/east_infer_speed_test.py --folder "C:\shared\data0205\data02065\ICDAR2015\test_images" --annotations "C:\shared\data0205\data02065\ICDAR2015\test.json"
    python scripts/east_infer_speed_test.py --folder "C:\shared\data0205\data02065\school_notebooks_RU\test_images" --annotations "C:\shared\data0205\data02065\school_notebooks_RU\test.json"
    python scripts/east_infer_speed_test.py --folder "C:\shared\data0205\data02065\IAM\test_images" --annotations "C:\shared\data0205\data02065\IAM\test.json"
    python scripts/east_infer_speed_test.py --folder "C:\shared\data0205\data02065\TotalText\test_images" --annotations "C:\shared\data0205\data02065\TotalText\test.json"
    python scripts/east_infer_speed_test.py --folder "C:\shared\data0205\data02065\TotalText\test_images" --annotations "C:\shared\data0205\data02065\TotalText\test.json"
    # С кастомными параметрами
    python scripts/east_infer_speed_test.py --folder "C:\shared\data02065\Archives020525\test_images" --target-size 1280
"""

import argparse
import sys
import time
import gc
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import numpy as np
import cv2

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from manuscript.detectors import EAST
from manuscript.detectors._east.utils import evaluate_dataset


def get_image_files(folder: str) -> List[str]:
    """
    Находит все изображения в папке.

    Returns
    -------
    list
        Список путей к изображениям
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        raise ValueError(f"Folder not found: {folder}")

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_files = set()  # Используем set для избежания дубликатов

    for ext in extensions:
        # Ищем с учетом регистра (case-insensitive)
        image_files.update(folder_path.glob(f"*{ext}"))
        image_files.update(folder_path.glob(f"*{ext.upper()}"))

    return [str(f) for f in sorted(image_files)]


def load_ground_truth(annotation_file: str, images_folder: str) -> Dict[str, List]:
    """
    Загружает ground truth аннотации из COCO формата.

    Parameters
    ----------
    annotation_file : str
        Путь к JSON файлу с аннотациями в COCO формате
    images_folder : str
        Путь к папке с изображениями

    Returns
    -------
    dict
        Словарь {filename: list of boxes}, где boxes в формате (x_min, y_min, x_max, y_max)
    """
    with open(annotation_file, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    # Создаем маппинг image_id -> filename
    images_info = {img["id"]: img for img in coco_data["images"]}

    # Собираем аннотации по filename
    ground_truths = {}

    for ann in coco_data["annotations"]:
        image_id = ann["image_id"]
        if image_id not in images_info:
            continue

        filename = images_info[image_id]["file_name"]

        # Извлекаем segmentation и конвертируем в bbox
        seg = ann.get("segmentation")
        if not seg or len(seg) == 0:
            continue

        # segmentation может быть списком полигонов
        seg_parts = seg if isinstance(seg[0], list) else [seg]

        for seg_poly in seg_parts:
            if len(seg_poly) < 8:  # Минимум 4 точки (8 координат)
                continue

            # Конвертируем полигон в bbox
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


def get_memory_usage() -> Dict[str, float]:
    """
    Получает текущее использование памяти.

    Returns
    -------
    dict
        {"ram_mb": float, "gpu_mb": float or None}
    """
    import psutil

    # RAM
    process = psutil.Process()
    ram_mb = process.memory_info().rss / 1024 / 1024

    # GPU VRAM
    gpu_mb = None
    if torch.cuda.is_available():
        gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024

    return {"ram_mb": ram_mb, "gpu_mb": gpu_mb}


def benchmark_device(
    image_files: List[str],
    device: str,
    target_size: int = 1280,
    warmup_runs: int = 3,
    score_thresh: float = 0.6,
    collect_predictions: bool = False,
) -> Dict[str, Any]:
    """
    Бенчмарк EAST на указанном устройстве.

    Parameters
    ----------
    image_files : list
        Список путей к изображениям
    device : str
        Устройство ("cpu" или "cuda")
    target_size : int
        Размер входного изображения
    warmup_runs : int
        Количество прогревочных запусков
    score_thresh : float
        Порог уверенности
    collect_predictions : bool
        Собирать предсказания для оценки точности

    Returns
    -------
    dict
        Статистика бенчмарка
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking on {device.upper()}")
    print(f"{'='*60}")

    # Очищаем память
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Создаём детектор
    print(f"Initializing EAST detector on {device}...")
    detector = EAST(device=device)

    # Измеряем память после загрузки модели
    mem_after_load = get_memory_usage()
    print(f"Memory after model load:")
    print(f"  RAM: {mem_after_load['ram_mb']:.2f} MB")
    if mem_after_load["gpu_mb"] is not None:
        print(f"  GPU VRAM: {mem_after_load['gpu_mb']:.2f} MB")

    # Warmup
    print(f"\nWarmup ({warmup_runs} runs)...")
    warmup_images = image_files[: min(warmup_runs, len(image_files))]
    for img_path in warmup_images:
        _ = detector.predict(img_path, vis=False, profile=False)

    if device == "cuda":
        torch.cuda.synchronize()

    # Бенчмарк
    print(f"\nRunning benchmark on {len(image_files)} images...")

    inference_times = []
    detection_counts = []
    peak_memory = get_memory_usage()
    predictions = {} if collect_predictions else None

    for i, img_path in enumerate(image_files, 1):
        # Замеряем время
        start_time = time.time()

        result = detector.predict(img_path, vis=False, profile=False)

        if device == "cuda":
            torch.cuda.synchronize()

        inference_time = time.time() - start_time
        inference_times.append(inference_time)

        # Подсчитываем детекции
        num_detections = sum(len(block.words) for block in result["page"].blocks)
        detection_counts.append(num_detections)

        # Собираем предсказания для оценки точности
        if collect_predictions:
            filename = Path(img_path).name
            boxes = []

            for block in result["page"].blocks:
                for word in block.words:
                    # word.polygon - список вершин [(x1, y1), (x2, y2), ...]
                    # Конвертируем в bbox (x_min, y_min, x_max, y_max)
                    if word.polygon and len(word.polygon) > 0:
                        xs = [pt[0] for pt in word.polygon]
                        ys = [pt[1] for pt in word.polygon]
                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)
                        boxes.append((x_min, y_min, x_max, y_max))

            predictions[filename] = boxes

        # Отслеживаем пиковую память
        current_mem = get_memory_usage()
        if current_mem["ram_mb"] > peak_memory["ram_mb"]:
            peak_memory["ram_mb"] = current_mem["ram_mb"]
        if current_mem["gpu_mb"] is not None and peak_memory["gpu_mb"] is not None:
            if current_mem["gpu_mb"] > peak_memory["gpu_mb"]:
                peak_memory["gpu_mb"] = current_mem["gpu_mb"]

        # Прогресс
        if i % 10 == 0 or i == len(image_files):
            avg_time = (
                np.mean(inference_times[-10:])
                if len(inference_times) >= 10
                else np.mean(inference_times)
            )
            print(
                f"  [{i}/{len(image_files)}] Avg time (last 10): {avg_time*1000:.2f} ms"
            )

    # Статистика
    inference_times = np.array(inference_times)
    detection_counts = np.array(detection_counts)

    stats = {
        "device": device,
        "num_images": len(image_files),
        "target_size": target_size,
        # Время
        "mean_time_ms": float(np.mean(inference_times) * 1000),
        "median_time_ms": float(np.median(inference_times) * 1000),
        "std_time_ms": float(np.std(inference_times) * 1000),
        "min_time_ms": float(np.min(inference_times) * 1000),
        "max_time_ms": float(np.max(inference_times) * 1000),
        "total_time_s": float(np.sum(inference_times)),
        "throughput_fps": float(len(image_files) / np.sum(inference_times)),
        # Детекции
        "mean_detections": float(np.mean(detection_counts)),
        "total_detections": int(np.sum(detection_counts)),
        # Память
        "ram_after_load_mb": mem_after_load["ram_mb"],
        "ram_peak_mb": peak_memory["ram_mb"],
        "ram_delta_mb": peak_memory["ram_mb"] - mem_after_load["ram_mb"],
    }

    if device == "cuda":
        stats["gpu_after_load_mb"] = mem_after_load["gpu_mb"]
        stats["gpu_peak_mb"] = peak_memory["gpu_mb"]
        stats["gpu_delta_mb"] = peak_memory["gpu_mb"] - mem_after_load["gpu_mb"]

    if collect_predictions:
        stats["predictions"] = predictions

    return stats


def print_stats(stats: Dict[str, Any]):
    """Красиво печатает статистику."""
    print(f"\n{'='*60}")
    print(f"Results for {stats['device'].upper()}")
    print(f"{'='*60}")

    print(f"\nDataset:")
    print(f"  Images: {stats['num_images']}")
    print(f"  Target size: {stats['target_size']}x{stats['target_size']}")
    print(f"  Total detections: {stats['total_detections']}")
    print(f"  Avg detections/image: {stats['mean_detections']:.1f}")

    print(f"\nInference Time:")
    print(f"  Mean: {stats['mean_time_ms']:.2f} ms")
    print(f"  Median: {stats['median_time_ms']:.2f} ms")
    print(f"  Std: {stats['std_time_ms']:.2f} ms")
    print(f"  Min: {stats['min_time_ms']:.2f} ms")
    print(f"  Max: {stats['max_time_ms']:.2f} ms")
    print(f"  Total: {stats['total_time_s']:.2f} s")
    print(f"  Throughput: {stats['throughput_fps']:.2f} FPS")

    print(f"\nMemory Usage (RAM):")
    print(f"  After load: {stats['ram_after_load_mb']:.2f} MB")
    print(f"  Peak: {stats['ram_peak_mb']:.2f} MB")
    print(f"  Delta: {stats['ram_delta_mb']:.2f} MB")

    if "gpu_after_load_mb" in stats:
        print(f"\nMemory Usage (GPU VRAM):")
        print(f"  After load: {stats['gpu_after_load_mb']:.2f} MB")
        print(f"  Peak: {stats['gpu_peak_mb']:.2f} MB")
        print(f"  Delta: {stats['gpu_delta_mb']:.2f} MB")

    # Печатаем метрики точности, если они есть
    if "accuracy_metrics" in stats:
        metrics = stats["accuracy_metrics"]
        print(f"\nAccuracy Metrics:")
        print(f"  F1@0.5: {metrics['f1@0.5']:.4f}")
        print(f"  Precision@0.5: {metrics['precision@0.5']:.4f}")
        print(f"  Recall@0.5: {metrics['recall@0.5']:.4f}")
        print(f"  F1@0.5:0.95: {metrics['f1@0.5:0.95']:.4f}")
        print(f"\n  Total Predictions: {metrics['num_predictions']}")
        print(f"  Total Ground Truths: {metrics['num_ground_truths']}")
        print(f"  TP@0.5: {metrics.get('tp@0.50', 0)}")
        print(f"  FP@0.5: {metrics.get('fp@0.50', 0)}")
        print(f"  FN@0.5: {metrics.get('fn@0.50', 0)}")


def compare_devices(cpu_stats: Dict[str, Any], gpu_stats: Dict[str, Any]):
    """Сравнивает производительность CPU и GPU."""
    print(f"\n{'='*60}")
    print("CPU vs GPU Comparison")
    print(f"{'='*60}")

    speedup = cpu_stats["mean_time_ms"] / gpu_stats["mean_time_ms"]
    throughput_gain = gpu_stats["throughput_fps"] / cpu_stats["throughput_fps"]

    print(f"\nSpeed:")
    print(f"  CPU mean time: {cpu_stats['mean_time_ms']:.2f} ms")
    print(f"  GPU mean time: {gpu_stats['mean_time_ms']:.2f} ms")
    print(f"  Speedup: {speedup:.2f}x")

    print(f"\nThroughput:")
    print(f"  CPU: {cpu_stats['throughput_fps']:.2f} FPS")
    print(f"  GPU: {gpu_stats['throughput_fps']:.2f} FPS")
    print(f"  Gain: {throughput_gain:.2f}x")

    print(f"\nMemory:")
    print(f"  CPU RAM peak: {cpu_stats['ram_peak_mb']:.2f} MB")
    print(f"  GPU RAM peak: {gpu_stats['ram_peak_mb']:.2f} MB")
    print(f"  GPU VRAM peak: {gpu_stats['gpu_peak_mb']:.2f} MB")

    print(f"\nRecommendation:")
    if speedup > 2:
        print(f"  GPU is {speedup:.1f}x faster - strongly recommended for production")
    elif speedup > 1.5:
        print(f"  GPU is {speedup:.1f}x faster - recommended if available")
    else:
        print(f"  GPU is only {speedup:.1f}x faster - CPU may be sufficient")


def save_results(
    cpu_stats: Dict[str, Any], gpu_stats: Optional[Dict[str, Any]], output_file: str
):
    """Сохраняет результаты в JSON файл."""
    results = {
        "cpu": cpu_stats,
    }

    if gpu_stats:
        results["gpu"] = gpu_stats
        results["comparison"] = {
            "speedup": cpu_stats["mean_time_ms"] / gpu_stats["mean_time_ms"],
            "throughput_gain": gpu_stats["throughput_fps"]
            / cpu_stats["throughput_fps"],
        }

    # Удаляем predictions из сохранения (слишком большой объем данных)
    if "predictions" in results.get("cpu", {}):
        del results["cpu"]["predictions"]
    if gpu_stats and "predictions" in results.get("gpu", {}):
        del results["gpu"]["predictions"]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark EAST detector performance and accuracy on CPU and GPU"
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
        default=None,
        help="Path to COCO format annotations JSON file (optional, for accuracy evaluation)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=1280,
        help="Target image size (default: 1280)",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.6,
        help="Score threshold (default: 0.6)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup runs (default: 3)",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Benchmark only on CPU",
    )
    parser.add_argument(
        "--gpu-only",
        action="store_true",
        help="Benchmark only on GPU",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file",
    )

    args = parser.parse_args()

    # Проверяем папку
    if not Path(args.folder).exists():
        print(f"Error: Folder not found: {args.folder}")
        sys.exit(1)

    # Находим изображения
    print("Searching for images...")
    image_files = get_image_files(args.folder)

    if len(image_files) == 0:
        print(f"Error: No images found in {args.folder}")
        sys.exit(1)

    print(f"Found {len(image_files)} images")

    # Загружаем ground truth, если указан
    ground_truths = None
    if args.annotations:
        if not Path(args.annotations).exists():
            print(f"Error: Annotations file not found: {args.annotations}")
            sys.exit(1)

        print(f"Loading ground truth annotations from {args.annotations}...")
        ground_truths = load_ground_truth(args.annotations, args.folder)
        print(f"Loaded annotations for {len(ground_truths)} images")

    # Проверяем доступность CUDA
    cuda_available = torch.cuda.is_available()
    if args.gpu_only and not cuda_available:
        print("Error: GPU requested but CUDA not available")
        sys.exit(1)

    if cuda_available:
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, will benchmark only on CPU")

    # Бенчмарк
    cpu_stats = None
    gpu_stats = None

    # CPU
    if not args.gpu_only:
        cpu_stats = benchmark_device(
            image_files,
            device="cpu",
            target_size=args.target_size,
            warmup_runs=args.warmup,
            score_thresh=args.score_thresh,
            collect_predictions=(ground_truths is not None),
        )

        # Оцениваем точность, если есть ground truth
        if ground_truths and "predictions" in cpu_stats:
            print("\nEvaluating accuracy on CPU predictions...")
            # Используем параллельную обработку для ускорения вычисления метрик
            accuracy_metrics = evaluate_dataset(
                cpu_stats["predictions"], ground_truths, verbose=True, n_jobs=1
            )
            cpu_stats["accuracy_metrics"] = accuracy_metrics

        print_stats(cpu_stats)

    # GPU
    if cuda_available and not args.cpu_only:
        gpu_stats = benchmark_device(
            image_files,
            device="cuda",
            target_size=args.target_size,
            warmup_runs=args.warmup,
            score_thresh=args.score_thresh,
            collect_predictions=(ground_truths is not None),
        )

        # Оцениваем точность, если есть ground truth
        if ground_truths and "predictions" in gpu_stats:
            print("\nEvaluating accuracy on GPU predictions...")
            # Используем параллельную обработку для ускорения вычисления метрик
            accuracy_metrics = evaluate_dataset(
                gpu_stats["predictions"], ground_truths, verbose=True, n_jobs=1
            )
            gpu_stats["accuracy_metrics"] = accuracy_metrics

        print_stats(gpu_stats)

    # Сравнение
    if cpu_stats and gpu_stats:
        compare_devices(cpu_stats, gpu_stats)

    # Сохранение
    if args.output:
        save_results(cpu_stats, gpu_stats, args.output)

    print(f"\n{'='*60}")
    print("Benchmark completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
