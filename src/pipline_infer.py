from manuscript import Pipeline
from manuscript.recognizers import TRBA


recognizer = TRBA(model_path = r"C:\Users\USER\Desktop\OCR_MODELS\exp_2\best_acc_weights.pth", 
config_path = r"C:\Users\USER\Desktop\OCR_MODELS\exp_2\config.json")

# Инициализация с моделями по умолчанию
pipeline = Pipeline(recognizer=recognizer)

# Обработка изображения
result, img = pipeline.predict(r"C:\Users\USER\Desktop\IMG_9056.JPG", vis=True)

# Извлечение текста
text = pipeline.get_text(result)
print(text)

img.show()