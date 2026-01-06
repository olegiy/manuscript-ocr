# Руководство по развертыванию и запуску Manuscript OCR

Данный документ содержит исчерпывающие инструкции по настройке, развертыванию и эксплуатации библиотеки Manuscript OCR для DevOps-специалистов и разработчиков.

---

## 1. Системные требования

### Минимальные требования (CPU)
- **ОС:** Linux (Ubuntu 20.04+), Windows 10/11, macOS
- **Процессор:** 4 ядра, поддержка AVX2
- **ОЗУ:** 8 ГБ
- **Диск:** 5 ГБ свободного места (включая веса моделей)

### Рекомендуемые требования (GPU)
- **GPU:** NVIDIA GPU с поддержкой CUDA 11.8+
- **Видеопамять (VRAM):** 6 ГБ+
- **ОЗУ:** 16 ГБ

### Предварительно установленное ПО
- **Python:** 3.8 — 3.11
- **Pip:** 21.0+
- **Git**
- **Docker & Docker Compose** (для контейнеризации)
- **NVIDIA Container Toolkit** (для использования GPU в Docker)

---

## 2. Подготовка и установка

### Клонирование репозитория
```bash
git clone https://github.com/olegiy/manuscript-ocr.git
cd manuscript-ocr
```

### Создание виртуального окружения
```bash
python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### Установка зависимостей

#### Для разработки (CPU)
```bash
pip install -r requirements.txt
pip install -e .
```

#### Для эксплуатации (GPU)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install .
```

---

## 3. Конфигурация

Проект использует конфигурационные файлы для настройки моделей распознавания (TRBA).

- **Путь к конфигу:** `src/manuscript/recognizers/_trba/configs/config.json`
- **Переменные окружения (опционально):**
    - `CUDA_VISIBLE_DEVICES`: индекс GPU (например, `0`).
    - `OMP_NUM_THREADS`: количество потоков для OpenMP (оптимизация CPU).

Пример настройки `config.json` для инференса:
Убедитесь, что пути в `resume_path` (если используется кастомная модель) указаны верно. По умолчанию пайплайн загружает предобученные веса автоматически.

---

## 4. Сборка и запуск

### Режим разработки (Development)
В режиме разработки рекомендуется использовать установку `-e` (editable), чтобы изменения в коде применялись без переустановки.

```bash
# Запуск тестов для проверки корректности установки
pytest tests/
```

### Режим эксплуатации (Production)
Используйте библиотеку как установленный пакет:

```python
from manuscript import Pipeline
pipeline = Pipeline()
result = pipeline.predict("path/to/image.jpg")
```

---

## 5. Контейнеризация (Docker)

Для изоляции зависимостей и упрощения развертывания используйте Docker.

### Сборка образа
```bash
docker build -t manuscript-ocr:latest .
```

### Запуск через Docker Compose
```bash
docker-compose up -d
```

### Health Check (Проверка работоспособности)
Для проверки успешного запуска контейнера и готовности моделей выполните:
```bash
# Узнайте имя запущенного контейнера (обычно manuscript-ocr-1-manuscript-ocr-1)
docker ps

# Запустите тестовый пайплайн внутри контейнера
docker exec -it <ИМЯ_КОНТЕЙНЕРА> python src/example_pipeline.py
```
Если скрипт отрабатывает без ошибок и выводит результат распознавания, система готова к работе.

---

## 6. Устранение типичных ошибок

| Ошибка | Причина | Решение |
| :--- | :--- | :--- |
| `ImportError: libGL.so.1` | Отсутствуют системные библиотеки OpenCV | Установите `libgl1-mesa-glx` (в Ubuntu) или используйте `opencv-python-headless`. |
| `RuntimeError: CUDA out of memory` | Недостаточно видеопамяти | Уменьшите `batch_size` в конфиге или используйте CPU. |
| `nvidia-container-cli: initialization error` | Проблема проброса GPU в Docker/WSL | Закомментируйте блок `deploy` в `docker-compose.yml` для запуска на CPU. |
| `ModuleNotFoundError` | Проект не установлен | Выполните `pip install -e .` в корневой директории. |
| `Shape mismatch` в EAST | Несоответствие размеров входного изображения | Убедитесь, что размеры кратны 32 (библиотека обычно делает это автоматически). |

---

## 7. Контакты и поддержка
- **Author Email:** sherstpasha99@gmail.com
- **Repository:** [https://github.com/olegiy/manuscript-ocr](https://github.com/olegiy/manuscript-ocr)
