# Performance Benchmarks

## Аппаратная конфигурация

| Компонент | Спецификация                  |
| --------- | ----------------------------- |
| **CPU**   | i9-14900KF              |
| **GPU**   | NVIDIA GeForce RTX 4080 SUPER |

---

## EAST Detector (ONNX Runtime)

| Датасет                    | Кол-во | F1-0.5 / F1-0.5:0.9 | GPU Sec/img | GPU FPS | GPU Mem | CPU Sec/img | CPU FPS | CPU Mem |
| -------------------------- | ------ | --------------------------- | ----------- | ------- | ------- | ----------- | ------- | ------- |
| Archives020525             | 270    | 0.905 / 0.604       | 0.322       | 3.11    | 934     | 0.696       | 1.44    | 1451    |
| School                     | 372    | 0.915 / 0.597       | 0.310       | 3.23    | 900     | 0.671       | 1.49    | 1472    |
| ICDAR2015                  | 200    | 0.536 / 0.280       | 0.049       | 20.24   | 909     | 0.335       | 2.98    | 1436    |
| IAM                        | 308    | 0.986 / 0.754       | 0.183       | 5.47    | 930     | 0.509       | 1.96    | 1458    |
| TotalText                  | 300    | 0.413 / 0.184       | 0.057       | 17.41   | 918     | 0.353       | 2.84    | 1447    |
| DDI_100                    | 311    | 0.709 / 0.312       | 0.217       | 4.61    | 931     | 0.554       | 1.81    | 1441    |

_Примечание: Mem в МБ._

**Детали для Archives020525:**

| Метрика                 | CPU        | GPU (CUDA)  |
| ----------------------- | ---------- | ----------- |
| **F1@0.5**              | 0.9047     | 0.9047      |
| **F1@0.5:0.95**         | 0.6040     | 0.6039      |
| **Среднее время**       | 665.37 ms  | 359.33 ms   |
| **Медиана**             | 600.59 ms  | 267.64 ms   |
| **Throughput**          | 1.50 FPS   | 2.78 FPS    |
| **Память (RAM)**        | 730.90 MB  | 346.80 MB   |
| **Время на 1 картинку** | ~665 ms    | ~359 ms     |

**Параметры:**
- Размер входа: 1280×1280 px
- Total detections: ~48,800
- Avg detections/image: 180.7

---

## Запуск бенчмарка

```bash
# Бенчмарк на Archives020525
python scripts/east_infer_speed_test.py \
    --folder "C:\shared\data02065\Archives020525\test_images" \
    --annotations "C:\shared\data02065\Archives020525\test.json"

# Бенчмарк на School (когда будут данные)
python scripts/east_infer_speed_test.py \
    --folder "path/to/school/images" \
    --annotations "path/to/school/annotations.json"
```

