# Scripts

Утилиты для работы с моделями EAST: экспорт в ONNX, бенчмарки производительности, оптимизация параметров и тестирование точности.

## Установка зависимостей

```bash
# Для бенчмарков
pip install psutil

# Для ONNX экспорта/инференса
pip install onnx onnxruntime

# Опционально для оптимизации графа:
pip install onnx-simplifier
```

## 0. Оптимизация параметров EAST (Grid Search)

### optimize_east_params.py

Автоматически подбирает оптимальные параметры EAST детектора для максимизации F1@0.5:0.95 метрики.

**Что оптимизирует:**
- `expand_ratio_w` - горизонтальное расширение боксов (0.7, 0.8, 0.9, 1.0)
- `expand_ratio_h` - вертикальное расширение боксов (0.7, 0.8, 0.9, 1.0)
- `score_thresh` - порог уверенности (0.5, 0.6, 0.7, 0.8)
- `iou_threshold` - порог IoU для NMS (0.1, 0.2, 0.3)
- `quantization` - квантизация координат (1, 2, 4)
- `remove_area_anomalies` - удаление аномальных областей (True, False)

**Использование:**

```bash
# Базовый запуск (использует 30% изображений)
python scripts/optimize_east_params.py \
    --folder path/to/images \
    --annotations path/to/annotations.json

# С настройками доли выборки и устройства
python scripts/optimize_east_params.py \
    --folder "C:\shared\data02065\Archives020525\test_images" \
    --annotations "C:\shared\data02065\Archives020525\test.json" \
    --sample-ratio 0.2 \
    --device cuda \
    --output optimization_results.json

# Быстрая оптимизация на малой выборке
python scripts/optimize_east_params.py \
    --folder path/to/images \
    --annotations path/to/annotations.json \
    --sample-ratio 0.1 \
    --seed 42
```

**Параметры:**
- `--folder`: Папка с изображениями (обязательный)
- `--annotations`: Путь к JSON файлу с аннотациями в COCO формате (обязательный)
- `--sample-ratio`: Доля изображений для оптимизации, 0.0-1.0 (default: 0.3)
- `--target-size`: Размер входа (default: 1280)
- `--device`: Устройство cpu или cuda (default: cpu)
- `--output`: Путь для сохранения результатов (default: optimization_results.json)
- `--seed`: Random seed для воспроизводимости (default: 42)

**Пример вывода:**

```
Total combinations to test: 576
Testing on 54 images
Images with annotations: 54

Grid Search: 100%|████████████| 576/576 [45:23<00:00, best_f1=0.8523]

============================================================
OPTIMIZATION RESULTS
============================================================

Best F1@0.5:0.95 Score: 0.8523

Best Parameters:
  expand_ratio_w: 0.9
  expand_ratio_h: 0.9
  score_thresh: 0.6
  iou_threshold: 0.2
  quantization: 2
  remove_area_anomalies: True

Top 5 parameter combinations:

1. F1@0.5:0.95 = 0.8523
   expand_ratio_w: 0.9
   expand_ratio_h: 0.9
   score_thresh: 0.6
   iou_threshold: 0.2
   quantization: 2
   remove_area_anomalies: True

2. F1@0.5:0.95 = 0.8501
   expand_ratio_w: 0.8
   expand_ratio_h: 0.9
   score_thresh: 0.6
   iou_threshold: 0.2
   quantization: 2
   remove_area_anomalies: True

[OK] Results saved to: optimization_results.json
```

**Формат выходного JSON:**

```json
{
  "best_params": {
    "expand_ratio_w": 0.9,
    "expand_ratio_h": 0.9,
    "score_thresh": 0.6,
    "iou_threshold": 0.2,
    "quantization": 2,
    "remove_area_anomalies": true
  },
  "best_score": 0.8523,
  "optimization_settings": {
    "sample_ratio": 0.3,
    "sampled_images": 54,
    "seed": 42
  },
  "history": [
    {
      "params": {...},
      "f1_score": 0.8523
    }
  ]
}
```

## 1. Бенчмарк производительности и точности EAST

### east_infer_speed_test.py

Измеряет производительность и точность EAST детектора на CPU и GPU.

**Что измеряет:**
- Среднее время инференса на изображение
- Throughput (FPS - изображений в секунду)
- Использование RAM и VRAM
- Сравнение CPU vs GPU
- **Метрики точности (опционально):**
  - F1@0.5 - F1 score при IoU порога 0.5
  - F1@0.5:0.95 - усредненный F1 score для IoU порогов от 0.5 до 0.95
  - Precision и Recall при IoU 0.5
  - True Positives, False Positives, False Negatives

**Использование:**

```bash
# Базовый бенчмарк (только скорость)
python scripts/east_infer_speed_test.py --folder path/to/images

# С оценкой точности (требуются аннотации в COCO формате)
python scripts/east_infer_speed_test.py \
    --folder path/to/images \
    --annotations path/to/annotations.json

# С настройками
python scripts/east_infer_speed_test.py \
    --folder path/to/images \
    --annotations path/to/annotations.json \
    --folder path/to/images \
    --annotations path/to/annotations.json \
    --output benchmark_results.json
```

**Параметры:**
- `--folder`: Папка с изображениями (обязательный)
- `--annotations`: Путь к JSON файлу с аннотациями в COCO формате (опционально, для оценки точности)
- `--target-size`: Размер входа (default: 1280)
- `--score-thresh`: Порог уверенности (default: 0.6)
- `--warmup`: Количество прогревочных запусков (default: 3)
- `--cpu-only`: Бенчмарк только на CPU
- `--gpu-only`: Бенчмарк только на GPU
- `--output`: Сохранить результаты в JSON

**Формат аннотаций:**
Аннотации должны быть в COCO формате:
```json
{
  "images": [
    {"id": 1, "file_name": "image1.jpg", "width": 800, "height": 600}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "segmentation": [[x1, y1, x2, y2, x3, y3, x4, y4]],
      "category_id": 1
    }
  ]
}
```

**Пример вывода (с оценкой точности):**

```
============================================================
Results for CPU
============================================================

Dataset:
  Images: 4
  Target size: 640x640
  Total detections: 818
  Avg detections/image: 204.5

Inference Time:
  Mean: 530.07 ms
  Median: 524.61 ms
  Throughput: 1.89 FPS

Memory Usage (RAM):
  After load: 832.00 MB
  Peak: 939.15 MB
  Delta: 107.14 MB

Accuracy Metrics:
  F1@0.5: 0.8523
  Precision@0.5: 0.8912
  Recall@0.5: 0.8167
  F1@0.5:0.95: 0.7234
  
  Total Predictions: 818
  Total Ground Truths: 752
  TP@0.5: 672
  FP@0.5: 146
  FN@0.5: 80

============================================================
CPU vs GPU Comparison
============================================================

Speed:
  CPU mean time: 530.07 ms
  GPU mean time: 237.69 ms
  Speedup: 2.23x

Recommendation:
  GPU is 2.2x faster - strongly recommended for production
```

## 2. Экспорт модели в ONNX

### export_east_to_onnx.py

Экспортирует PyTorch модель EAST в формат ONNX для deployment.

**Использование:**

```bash
# Экспорт с параметрами по умолчанию (использует ~/.manuscript/east/east_quad_23_05.pth)
python scripts/export_east_to_onnx.py

# Указать свои веса и выходной файл
python scripts/export_east_to_onnx.py --weights path/to/weights.pth --output model.onnx

# Использовать другой размер входа
python scripts/export_east_to_onnx.py --input-size 1280 --output east_1280.onnx
```

**Параметры:**
- `--weights`: Путь к весам PyTorch (.pth файл). По умолчанию использует `~/.manuscript/east/east_quad_23_05.pth`
- `--output`: Путь для сохранения ONNX модели (default: `east_model.onnx`)
- `--input-size`: Размер входного изображения в пикселях (default: 1280)
- `--opset`: Версия ONNX opset (default: 14)
- `--no-simplify`: Отключить оптимизацию графа через onnx-simplifier

## 3. Тестирование ONNX инференса

### test_east_onnx_inference.py

Тестирует ONNX модель и сравнивает с PyTorch.

```bash
# Простой тест инференса
python scripts/test_east_onnx_inference.py \
    --model east_model.onnx \
    --image path/to/test/image.jpg

# Сравнение с PyTorch моделью
python scripts/test_east_onnx_inference.py \
    --model east_model.onnx \
    --image path/to/test/image.jpg \
    --compare
```

### Параметры

- `--model`: Путь к ONNX модели (обязательный)
- `--image`: Путь к тестовому изображению (обязательный)
- `--compare`: Сравнить с PyTorch моделью
- `--weights`: Путь к PyTorch весам для сравнения
- `--output`: Путь для сохранения визуализации (default: `onnx_detection_result.jpg`)
- `--target-size`: Размер входа (должен совпадать с размером при экспорте)
- `--score-thresh`: Порог уверенности для детекций (default: 0.6)
- `--use-cuda`: Использовать CUDA provider для ONNX Runtime

### Пример полного workflow

```bash
# 1. Экспорт модели
python scripts/export_east_to_onnx.py --output east_640.onnx --input-size 640

# 2. Тест инференса
python scripts/test_east_onnx_inference.py \
    --model east_640.onnx \
    --image example/test_image.jpg \
    --target-size 640 \
    --output result.jpg

# 3. Сравнение с PyTorch
python scripts/test_east_onnx_inference.py \
    --model east_640.onnx \
    --image example/test_image.jpg \
    --target-size 640 \
    --compare
```

## Что делают скрипты

### export_east_to_onnx.py

1. Загружает PyTorch модель EAST
2. Оборачивает её в EASTWrapper (конвертирует dict output в tuple для ONNX)
3. Экспортирует в ONNX с поддержкой dynamic axes
4. Проверяет валидность ONNX модели
5. Опционально упрощает граф через onnx-simplifier

**Выходы ONNX модели:**
- `score_map`: [B, 1, H/4, W/4] - карта уверенности детекций
- `geo_map`: [B, 8, H/4, W/4] - карта геометрии (координаты четырехугольников)

### test_east_onnx_inference.py

1. Загружает ONNX модель через onnxruntime
2. Предобрабатывает входное изображение
3. Запускает инференс и замеряет время
4. Опционально сравнивает с PyTorch моделью
5. Декодирует детекции из карт
6. Сохраняет визуализацию результата

## Производительность

ONNX модель обычно работает быстрее PyTorch на CPU:
- PyTorch CPU: ~200-500ms на изображение 1280x1280
- ONNX CPU: ~100-300ms на изображение 1280x1280
- Speedup: 1.5-2x

Для GPU инференса рекомендуется:
```bash
pip install onnxruntime-gpu
python scripts/test_east_onnx_inference.py --model model.onnx --image img.jpg --use-cuda
```

## Интеграция

После экспорта ONNX модель можно использовать:
- В production сервисах (FastAPI, Flask)
- На мобильных устройствах (через ONNX Runtime Mobile)

## 4. Функции оценки точности детектора

Все функции для оценки точности находятся в `src/manuscript/detectors/_east/utils.py` и могут быть использованы в любых скриптах.

### Доступные функции

#### box_iou(box1, box2)
Вычисляет IoU (Intersection over Union) между двумя прямоугольными боксами.

```python
from manuscript.detectors._east.utils import box_iou

box1 = (10, 10, 50, 50)  # (x_min, y_min, x_max, y_max)
box2 = (30, 30, 70, 70)
iou = box_iou(box1, box2)
print(f"IoU: {iou:.4f}")
```

#### match_boxes(pred_boxes, gt_boxes, iou_threshold=0.5)
Сопоставляет предсказанные боксы с ground truth боксами и возвращает TP, FP, FN.

```python
from manuscript.detectors._east.utils import match_boxes

pred_boxes = [(10, 10, 50, 50), (60, 60, 100, 100)]
gt_boxes = [(12, 12, 48, 48), (300, 300, 350, 350)]
tp, fp, fn = match_boxes(pred_boxes, gt_boxes, iou_threshold=0.5)
print(f"TP: {tp}, FP: {fp}, FN: {fn}")
```

#### compute_f1_score(true_positives, false_positives, false_negatives)
Вычисляет F1 score, precision и recall по значениям TP, FP, FN.

```python
from manuscript.detectors._east.utils import compute_f1_score

f1, precision, recall = compute_f1_score(tp=80, fp=10, fn=10)
print(f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
```

#### evaluate_detection(pred_boxes, gt_boxes, iou_threshold=0.5)
Оценивает качество детекции для одного изображения.

```python
from manuscript.detectors._east.utils import evaluate_detection

pred_boxes = [(10, 10, 50, 50), (60, 60, 100, 100)]
gt_boxes = [(12, 12, 48, 48), (61, 61, 99, 99)]
metrics = evaluate_detection(pred_boxes, gt_boxes, iou_threshold=0.5)

print(f"F1: {metrics['f1']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
```

#### evaluate_detection_multi_iou(pred_boxes, gt_boxes, iou_thresholds=None)
Оценивает детекцию с несколькими порогами IoU (0.5, 0.55, ..., 0.95).

```python
from manuscript.detectors._east.utils import evaluate_detection_multi_iou

pred_boxes = [(10, 10, 50, 50)]
gt_boxes = [(12, 12, 48, 48)]
metrics = evaluate_detection_multi_iou(pred_boxes, gt_boxes)

print(f"F1@0.5: {metrics['f1@0.5']:.4f}")
print(f"F1@0.5:0.95: {metrics['f1@0.5:0.95']:.4f}")  # mAP-style metric
```

#### evaluate_dataset(predictions, ground_truths, iou_thresholds=None, verbose=True)
Оценивает качество детекции на всем датасете.

```python
from manuscript.detectors._east.utils import evaluate_dataset

# predictions и ground_truths - словари {filename: list of boxes}
predictions = {
    "image1.jpg": [(10, 10, 50, 50), (60, 60, 100, 100)],
    "image2.jpg": [(20, 20, 80, 80)],
}

ground_truths = {
    "image1.jpg": [(12, 12, 48, 48), (61, 61, 99, 99)],
    "image2.jpg": [(22, 22, 78, 78), (200, 200, 250, 250)],
}

metrics = evaluate_dataset(predictions, ground_truths, verbose=True)

print(f"F1@0.5: {metrics['f1@0.5']:.4f}")
print(f"F1@0.5:0.95: {metrics['f1@0.5:0.95']:.4f}")
print(f"Precision@0.5: {metrics['precision@0.5']:.4f}")
print(f"Recall@0.5: {metrics['recall@0.5']:.4f}")
print(f"Total predictions: {metrics['num_predictions']}")
print(f"Total ground truths: {metrics['num_ground_truths']}")
```

### Пример использования

Запустите пример использования всех функций:

```bash
python scripts/example_evaluate_detector.py
```

Этот скрипт демонстрирует:
1. Вычисление IoU между боксами
2. Оценку детекции на одном изображении
3. Оценку с несколькими порогами IoU
4. Оценку на датасете

### Формат боксов

Все функции работают с прямоугольными боксами в формате:
```python
box = (x_min, y_min, x_max, y_max)
```

Где:
- `x_min`, `y_min` - координаты верхнего левого угла
- `x_max`, `y_max` - координаты нижнего правого угла

### Метрики

- **F1@0.5** - F1 score при IoU пороге 0.5 (стандартная метрика для детекции объектов)
- **F1@0.5:0.95** - средний F1 score для IoU порогов от 0.5 до 0.95 с шагом 0.05 (аналог mAP из COCO)
- **Precision** - точность (TP / (TP + FP))
- **Recall** - полнота (TP / (TP + FN))
- **TP** - истинно положительные (true positives)
- **FP** - ложно положительные (false positives)
- **FN** - ложно отрицательные (false negatives)

- В браузере (через ONNX.js)
- На edge устройствах

## Troubleshooting

**Ошибка "onnxruntime not installed":**
```bash
pip install onnxruntime
```

**Разные результаты PyTorch vs ONNX:**
- Это нормально если разница < 1e-3
- Может быть вызвано разными версиями операторов
- Проверьте что используете одинаковый input_size

**Модель слишком большая:**
- Используйте onnx-simplifier для оптимизации
- Попробуйте квантизацию (int8)
- Используйте меньший backbone (но нужно переобучить)
