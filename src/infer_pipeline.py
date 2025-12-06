from manuscript import Pipeline
from manuscript.utils.visualization import visualize_page


image_path = r"C:\shared\Archives020525\test_images\540.jpg"

# Создание OCR-пайплайна с моделями по умолчанию
pipeline = Pipeline()

# Обработка изображения и получение результата
result = pipeline.predict(image_path)

print(result)

text = pipeline.get_text(result["page"])
print(text)
