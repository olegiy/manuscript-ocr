from manuscript import Pipeline
from manuscript.utils.visualization import visualize_page


image_path = r"C:\Users\USER\Desktop\IMG_9056.JPG"

# Создание OCR-пайплайна с моделями по умолчанию
pipeline = Pipeline()

# Обработка изображения и получение результата
result = pipeline.predict(image_path)

print(result)
