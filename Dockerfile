# Dockerfile для Manuscript OCR

# Используем официальный образ Python с поддержкой CUDA для GPU инференса
# Если GPU не требуется, можно использовать python:3.10-slim
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Установка системных зависимостей для OpenCV и других библиотек
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Установка рабочей директории
WORKDIR /app

# Копирование файлов зависимостей
COPY requirements.txt .

# Установка Python зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода проекта
COPY . .

# Установка проекта в режиме редактирования (опционально) или как пакета
RUN pip install -e .

# Команда по умолчанию (может быть переопределена)
# В данном случае запускаем пример инференса для проверки
CMD ["python", "src/example_pipeline.py"]
