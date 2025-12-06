from manuscript import Pipeline
from manuscript.utils.visualization import visualize_page
from manuscript.detectors import EAST
from manuscript.recognizers import TRBA

image_path = r"C:\shared\Archives020525\test_images\540.jpg"

# Создание OCR-пайплайна с моделями по умолчанию
pipeline = Pipeline(detector=EAST(device="cuda"), recognizer=TRBA(device="cuda"))

# Обработка изображения и получение результата
result = pipeline.predict(image_path)

print(result)

text = pipeline.get_text(result["page"])
print(text)
