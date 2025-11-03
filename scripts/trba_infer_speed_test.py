"""
Бенчмарк производительности и точности TRBA распознавателя на CPU и GPU.

Измеряет:
- Среднее время инференса на батч изображений
- Использование памяти (RAM для CPU, VRAM для GPU)
- Throughput (слов в секунду и батчей в минуту)
- Точность распознавания (Character Error Rate, Word Error Rate) при наличии GT

Usage:
    # Только скорость
    python scripts/trba_infer_speed_test.py --folder path/to/words --batch-size 32

    # Скорость + точность (если есть GT файл)
    python scripts/trba_infer_speed_test.py --folder path/to/words --gt-csv path/to/gt.csv

    # Примеры:
    python scripts/trba_infer_speed_test.py --folder "path/to/words" --batch-size 32
    python scripts/trba_infer_speed_test.py --folder "path/to/words" --gt-csv "path/to/ground_truth.csv" --gpu-only
    python scripts/trba_infer_speed_test.py --folder "path/to/words" --batch-size 64 --mode greedy

    python scripts/trba_infer_speed_test.py --folder "C:\\Users\\USER\\Desktop\\archive_25_09\\dataset\\handwritten\\val\\img" --gt-csv "C:\\Users\\USER\\Desktop\\archive_25_09\\dataset\\handwritten\\val\\labels.csv"

    # С кастомными параметрами
    python scripts/trba_infer_speed_test.py --folder "path/to/words" --batch-size 128 --mode beam --beam-size 5
"""

import argparse
import sys
import time
import gc
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
import numpy as np
from tqdm import tqdm

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from manuscript.recognizers import TRBA
from manuscript.recognizers._trba.training.metrics import (
    character_error_rate,
    word_error_rate,
    compute_accuracy,
)


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
    image_files = set()

    for ext in extensions:
        image_files.update(folder_path.glob(f"*{ext}"))
        image_files.update(folder_path.glob(f"*{ext.upper()}"))

    return sorted([str(f) for f in image_files])


def load_ground_truth(gt_csv: str) -> Dict[str, str]:
    """
    Загружает ground truth из CSV файла.

    Ожидает формат: image_filename,text

    Parameters
    ----------
    gt_csv : str
        Путь к CSV файлу

    Returns
    -------
    dict
        Словарь {filename: text}
    """
    ground_truths = {}

    try:
        with open(gt_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV file is empty or malformed")

            # Пытаемся найти колонки
            has_image_col = "image" in reader.fieldnames or "filename" in reader.fieldnames
            has_text_col = "text" in reader.fieldnames or "label" in reader.fieldnames

            if not (has_image_col and has_text_col):
                raise ValueError(
                    f"CSV должен иметь колонки 'image'/'filename' и 'text'/'label'. "
                    f"Найденные: {reader.fieldnames}"
                )

            image_col = "image" if "image" in reader.fieldnames else "filename"
            text_col = "text" if "text" in reader.fieldnames else "label"

            for row in reader:
                filename = Path(row[image_col]).name
                text = row[text_col]
                ground_truths[filename] = text

    except Exception as e:
        print(f"Warning: Failed to load ground truth: {e}")

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

    process = psutil.Process()
    ram_mb = process.memory_info().rss / 1024 / 1024

    gpu_mb = None
    if torch.cuda.is_available():
        gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024

    return {"ram_mb": ram_mb, "gpu_mb": gpu_mb}


def benchmark_device(
    image_files: List[str],
    device: str,
    batch_size: int = 32,
    warmup_runs: int = 3,
    mode: str = "greedy",
    beam_size: int = 8,
    collect_predictions: bool = False,
) -> Dict[str, Any]:
    """
    Бенчмарк TRBA на указанном устройстве.

    Parameters
    ----------
    image_files : list
        Список путей к изображениям слов
    device : str
        Устройство ("cpu" или "cuda")
    batch_size : int
        Размер батча для обработки
    warmup_runs : int
        Количество прогревочных батчей
    mode : str
        Режим декодирования ("greedy" или "beam")
    beam_size : int
        Размер луча для beam search
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

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Создаём распознаватель
    print(f"Initializing TRBA recognizer on {device}...")
    recognizer = TRBA(device=device)

    # Измеряем память после загрузки модели
    mem_after_load = get_memory_usage()
    print(f"Memory after model load:")
    print(f"  RAM: {mem_after_load['ram_mb']:.2f} MB")
    if mem_after_load["gpu_mb"] is not None:
        print(f"  GPU VRAM: {mem_after_load['gpu_mb']:.2f} MB")

    # Warmup
    print(f"\nWarmup ({warmup_runs} batches)...")
    num_warmup = min(warmup_runs * batch_size, len(image_files))
    warmup_images = image_files[:num_warmup]

    for i in range(0, len(warmup_images), batch_size):
        batch = warmup_images[i : i + batch_size]
        _ = recognizer.predict(batch)

    if device == "cuda":
        torch.cuda.synchronize()

    # Бенчмарк
    print(f"\nRunning benchmark on {len(image_files)} images (batch size: {batch_size})...")

    batch_times = []
    predictions = {} if collect_predictions else None
    peak_memory = get_memory_usage()

    num_batches = (len(image_files) + batch_size - 1) // batch_size
    processed_images = 0

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(image_files))
        batch = image_files[start_idx:end_idx]
        current_batch_size = len(batch)

        # Замеряем время
        start_time = time.time()

        results = recognizer.predict(batch)

        if device == "cuda":
            torch.cuda.synchronize()

        batch_time = time.time() - start_time
        batch_times.append(batch_time)
        processed_images += current_batch_size

        # Собираем предсказания
        if collect_predictions:
            for img_path, result in zip(batch, results):
                filename = Path(img_path).name
                predictions[filename] = result["text"]

        # Отслеживаем пиковую память
        current_mem = get_memory_usage()
        if current_mem["ram_mb"] > peak_memory["ram_mb"]:
            peak_memory["ram_mb"] = current_mem["ram_mb"]
        if current_mem["gpu_mb"] is not None and peak_memory["gpu_mb"] is not None:
            if current_mem["gpu_mb"] > peak_memory["gpu_mb"]:
                peak_memory["gpu_mb"] = current_mem["gpu_mb"]

        # Прогресс - показываем каждые 10 батчей
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
            avg_batch_time = (
                np.mean(batch_times[-10:]) if len(batch_times) >= 10 else np.mean(batch_times)
            )
            current_throughput = processed_images / sum(batch_times)
            avg_time_per_word = (sum(batch_times) / processed_images) * 1000
            
            print(
                f"  [{processed_images}/{len(image_files)}] "
                f"Batch time (last 10 avg): {avg_batch_time*1000:.2f} ms, "
                f"Per word: {avg_time_per_word:.2f} ms, "
                f"Throughput: {current_throughput:.1f} words/sec"
            )

    batch_times = np.array(batch_times)

    # Вычисляем статистику
    total_words = len(image_files)
    total_time = np.sum(batch_times)

    stats = {
        "device": device,
        "num_images": total_words,
        "batch_size": batch_size,
        "mode": mode,
        # Время батчей
        "mean_batch_time_ms": float(np.mean(batch_times) * 1000),
        "median_batch_time_ms": float(np.median(batch_times) * 1000),
        "std_batch_time_ms": float(np.std(batch_times) * 1000),
        "min_batch_time_ms": float(np.min(batch_times) * 1000),
        "max_batch_time_ms": float(np.max(batch_times) * 1000),
        "total_time_s": float(total_time),
        "throughput_words_per_sec": float(total_words / total_time),
        "throughput_batches_per_min": float((num_batches / total_time) * 60),
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
    print(f"  Batch size: {stats['batch_size']}")
    print(f"  Decoding mode: {stats['mode']}")

    print(f"\nBatch Inference Time:")
    print(f"  Mean: {stats['mean_batch_time_ms']:.2f} ms")
    print(f"  Median: {stats['median_batch_time_ms']:.2f} ms")
    print(f"  Std: {stats['std_batch_time_ms']:.2f} ms")
    print(f"  Min: {stats['min_batch_time_ms']:.2f} ms")
    print(f"  Max: {stats['max_batch_time_ms']:.2f} ms")
    print(f"  Total: {stats['total_time_s']:.2f} s")

    print(f"\nThroughput:")
    print(f"  Words/sec: {stats['throughput_words_per_sec']:.2f}")
    print(f"  Batches/min: {stats['throughput_batches_per_min']:.1f}")
    print(f"  Time per word: {(stats['total_time_s'] / stats['num_images']) * 1000:.2f} ms")

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
        print(f"  CER (Character Error Rate): {metrics['cer']:.4f}")
        print(f"  WER (Word Error Rate): {metrics['wer']:.4f}")
        print(f"  Correct predictions: {metrics['correct']} / {metrics['total']}")


def compare_devices(cpu_stats: Dict[str, Any], gpu_stats: Dict[str, Any]):
    """Сравнивает производительность CPU и GPU."""
    print(f"\n{'='*60}")
    print("CPU vs GPU Comparison")
    print(f"{'='*60}")

    speedup = cpu_stats["mean_batch_time_ms"] / gpu_stats["mean_batch_time_ms"]
    throughput_gain = gpu_stats["throughput_words_per_sec"] / cpu_stats["throughput_words_per_sec"]

    print(f"\nSpeed:")
    print(f"  CPU mean batch time: {cpu_stats['mean_batch_time_ms']:.2f} ms")
    print(f"  GPU mean batch time: {gpu_stats['mean_batch_time_ms']:.2f} ms")
    print(f"  Speedup: {speedup:.2f}x")

    print(f"\nThroughput:")
    print(f"  CPU: {cpu_stats['throughput_words_per_sec']:.2f} words/sec")
    print(f"  GPU: {gpu_stats['throughput_words_per_sec']:.2f} words/sec")
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
    import json

    results = {"cpu": cpu_stats}

    if gpu_stats:
        results["gpu"] = gpu_stats
        results["comparison"] = {
            "speedup": cpu_stats["mean_batch_time_ms"] / gpu_stats["mean_batch_time_ms"],
            "throughput_gain": gpu_stats["throughput_words_per_sec"]
            / cpu_stats["throughput_words_per_sec"],
        }

    # Удаляем predictions из сохранения
    if "predictions" in results.get("cpu", {}):
        del results["cpu"]["predictions"]
    if gpu_stats and "predictions" in results.get("gpu", {}):
        del results["gpu"]["predictions"]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark TRBA recognizer performance and accuracy on CPU and GPU"
    )
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Path to folder with word images",
    )
    parser.add_argument(
        "--gt-csv",
        type=str,
        default=None,
        help="Path to CSV with ground truth (columns: image, text)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["greedy", "beam"],
        default="greedy",
        help="Decoding mode (default: greedy)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=8,
        help="Beam size for beam search (default: 8)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup batches (default: 3)",
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
    if args.gt_csv:
        if not Path(args.gt_csv).exists():
            print(f"Error: Ground truth file not found: {args.gt_csv}")
            sys.exit(1)

        print(f"Loading ground truth from {args.gt_csv}...")
        ground_truths = load_ground_truth(args.gt_csv)
        print(f"Loaded ground truth for {len(ground_truths)} images")

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
            batch_size=args.batch_size,
            warmup_runs=args.warmup,
            mode=args.mode,
            beam_size=args.beam_size,
            collect_predictions=(ground_truths is not None),
        )

        # Оцениваем точность, если есть ground truth
        if ground_truths and "predictions" in cpu_stats:
            print("\nEvaluating accuracy on CPU predictions...")
            predictions = cpu_stats["predictions"]
            correct = 0
            total = 0
            cer_values = []
            wer_values = []

            for filename, pred_text in tqdm(predictions.items(), desc="Computing metrics", unit="word"):
                if filename in ground_truths:
                    gt_text = ground_truths[filename]
                    total += 1

                    if pred_text.strip() == gt_text.strip():
                        correct += 1

                    cer_values.append(character_error_rate(gt_text, pred_text))
                    wer_values.append(word_error_rate(gt_text, pred_text))

            if total > 0:
                cpu_stats["accuracy_metrics"] = {
                    "correct": correct,
                    "total": total,
                    "accuracy": correct / total,
                    "cer": np.mean(cer_values),
                    "wer": np.mean(wer_values),
                }

        print_stats(cpu_stats)

    # GPU
    if cuda_available and not args.cpu_only:
        gpu_stats = benchmark_device(
            image_files,
            device="cuda",
            batch_size=args.batch_size,
            warmup_runs=args.warmup,
            mode=args.mode,
            beam_size=args.beam_size,
            collect_predictions=(ground_truths is not None),
        )

        # Оцениваем точность, если есть ground truth
        if ground_truths and "predictions" in gpu_stats:
            print("\nEvaluating accuracy on GPU predictions...")
            predictions = gpu_stats["predictions"]
            correct = 0
            total = 0
            cer_values = []
            wer_values = []

            for filename, pred_text in tqdm(predictions.items(), desc="Computing metrics", unit="word"):
                if filename in ground_truths:
                    gt_text = ground_truths[filename]
                    total += 1

                    if pred_text.strip() == gt_text.strip():
                        correct += 1

                    cer_values.append(character_error_rate(gt_text, pred_text))
                    wer_values.append(word_error_rate(gt_text, pred_text))

            if total > 0:
                gpu_stats["accuracy_metrics"] = {
                    "correct": correct,
                    "total": total,
                    "accuracy": correct / total,
                    "cer": np.mean(cer_values),
                    "wer": np.mean(wer_values),
                }

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
