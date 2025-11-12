from manuscript import Pipeline
from manuscript.recognizers import TRBA


recognizer = TRBA(model_path = r"C:\Users\USER\Desktop\OCR_MODELS\exp_2\best_acc_weights.pth", 
config_path = r"C:\Users\USER\Desktop\OCR_MODELS\exp_2\config.json")

# Инициализация с моделями по умолчанию
pipeline = Pipeline(recognizer=recognizer)

# Обработка изображения
result, img = pipeline.predict(r"C:\Users\USER\Desktop\егэ\image_2025-11-11_19-18-19.png", vis=True)

# Извлечение текста
text = pipeline.get_text(result)
print(text)

img.show()