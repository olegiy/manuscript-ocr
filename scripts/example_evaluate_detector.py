"""
Пример использования функций оценки точности детектора.

Демонстрирует:
1. Вычисление IoU между двумя боксами
2. Оценку детекции на одном изображении
3. Оценку на датасете с несколькими порогами IoU
"""

import sys
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from manuscript.detectors._east.utils import (
    box_iou,
    evaluate_detection,
    evaluate_detection_multi_iou,
    evaluate_dataset,
)


def example_1_basic_iou():
    """Пример 1: Вычисление IoU между двумя боксами."""
    print("=" * 60)
    print("Пример 1: Вычисление IoU")
    print("=" * 60)

    box1 = (10, 10, 50, 50)  # (x_min, y_min, x_max, y_max)
    box2 = (30, 30, 70, 70)  # Перекрывающийся бокс

    iou = box_iou(box1, box2)
    print(f"Box 1: {box1}")
    print(f"Box 2: {box2}")
    print(f"IoU: {iou:.4f}")

    # Случай полного совпадения
    box3 = (10, 10, 50, 50)
    iou_perfect = box_iou(box1, box3)
    print(f"\nBox 1: {box1}")
    print(f"Box 3: {box3}")
    print(f"IoU (perfect match): {iou_perfect:.4f}")

    # Случай отсутствия пересечения
    box4 = (100, 100, 150, 150)
    iou_no_overlap = box_iou(box1, box4)
    print(f"\nBox 1: {box1}")
    print(f"Box 4: {box4}")
    print(f"IoU (no overlap): {iou_no_overlap:.4f}")


def example_2_single_image_evaluation():
    """Пример 2: Оценка детекции на одном изображении."""
    print("\n" + "=" * 60)
    print("Пример 2: Оценка детекции на одном изображении")
    print("=" * 60)

    # Предсказанные боксы
    pred_boxes = [
        (10, 10, 50, 50),
        (60, 60, 100, 100),
        (200, 200, 250, 250),  # False positive
    ]

    # Ground truth боксы
    gt_boxes = [
        (12, 12, 48, 48),  # Должен совпасть с первым pred
        (61, 61, 99, 99),  # Должен совпасть со вторым pred
        (300, 300, 350, 350),  # False negative (не обнаружен)
    ]

    # Оцениваем при IoU threshold = 0.5
    metrics = evaluate_detection(pred_boxes, gt_boxes, iou_threshold=0.5)

    print(f"Predicted boxes: {len(pred_boxes)}")
    print(f"Ground truth boxes: {len(gt_boxes)}")
    print(f"\nMetrics @ IoU 0.5:")
    print(f"  True Positives: {metrics['tp']}")
    print(f"  False Positives: {metrics['fp']}")
    print(f"  False Negatives: {metrics['fn']}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")


def example_3_multi_iou_thresholds():
    """Пример 3: Оценка с несколькими порогами IoU."""
    print("\n" + "=" * 60)
    print("Пример 3: Оценка с несколькими порогами IoU")
    print("=" * 60)

    pred_boxes = [
        (10, 10, 50, 50),
        (60, 60, 100, 100),
    ]

    gt_boxes = [
        (15, 15, 55, 55),  # Частичное совпадение с первым pred
        (61, 61, 99, 99),  # Хорошее совпадение со вторым pred
    ]

    metrics = evaluate_detection_multi_iou(pred_boxes, gt_boxes)

    print(f"Predicted boxes: {len(pred_boxes)}")
    print(f"Ground truth boxes: {len(gt_boxes)}")
    print(f"\nMetrics:")
    print(f"  F1@0.5: {metrics['f1@0.5']:.4f}")
    print(f"  F1@0.5:0.95 (mAP style): {metrics['f1@0.5:0.95']:.4f}")
    print(f"  Precision@0.5: {metrics['precision@0.5']:.4f}")
    print(f"  Recall@0.5: {metrics['recall@0.5']:.4f}")

    # Показываем детальные метрики для разных порогов
    print(f"\nDetailed metrics by IoU threshold:")
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        f1_key = f"f1@{threshold:.2f}"
        if f1_key in metrics:
            print(f"  F1@{threshold:.2f}: {metrics[f1_key]:.4f}")


def example_4_dataset_evaluation():
    """Пример 4: Оценка на датасете."""
    print("\n" + "=" * 60)
    print("Пример 4: Оценка на датасете")
    print("=" * 60)

    # Предсказания для нескольких изображений
    predictions = {
        "image1.jpg": [
            (10, 10, 50, 50),
            (60, 60, 100, 100),
        ],
        "image2.jpg": [
            (20, 20, 80, 80),
            (150, 150, 200, 200),
            (300, 300, 350, 350),
        ],
        "image3.jpg": [
            (5, 5, 45, 45),
        ],
    }

    # Ground truth для тех же изображений
    ground_truths = {
        "image1.jpg": [
            (12, 12, 48, 48),
            (61, 61, 99, 99),
            (200, 200, 250, 250),  # Пропущен детектором
        ],
        "image2.jpg": [
            (22, 22, 78, 78),
            (151, 151, 199, 199),
        ],
        "image3.jpg": [
            (7, 7, 43, 43),
            (100, 100, 140, 140),  # Пропущен детектором
        ],
    }

    # Оцениваем весь датасет
    # Можно использовать n_jobs для параллельной обработки (работает на Windows)
    # n_jobs=None - использовать все CPU
    # n_jobs=1 - последовательная обработка
    # n_jobs=4 - использовать 4 процесса
    metrics = evaluate_dataset(predictions, ground_truths, verbose=False, n_jobs=1)

    print(f"Dataset size: {metrics['num_images']} images")
    print(f"Total predictions: {metrics['num_predictions']}")
    print(f"Total ground truths: {metrics['num_ground_truths']}")

    print(f"\nMetrics @ IoU 0.5:")
    print(f"  True Positives: {metrics['tp@0.50']}")
    print(f"  False Positives: {metrics['fp@0.50']}")
    print(f"  False Negatives: {metrics['fn@0.50']}")
    print(f"  Precision: {metrics['precision@0.5']:.4f}")
    print(f"  Recall: {metrics['recall@0.5']:.4f}")
    print(f"  F1 Score: {metrics['f1@0.5']:.4f}")

    print(f"\nOverall metrics:")
    print(f"  F1@0.5:0.95 (mAP style): {metrics['f1@0.5:0.95']:.4f}")


def main():
    """Запускаем все примеры."""
    example_1_basic_iou()
    example_2_single_image_evaluation()
    example_3_multi_iou_thresholds()
    example_4_dataset_evaluation()

    print("\n" + "=" * 60)
    print("Все примеры выполнены!")
    print("=" * 60)


if __name__ == "__main__":
    main()
