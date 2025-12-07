import time
from manuscript import Pipeline
from manuscript.detectors import EAST
from manuscript.recognizers import TRBA
from manuscript.utils.visualization import visualize_page

image_path = r"C:\Users\pasha\OneDrive\Рабочий стол\Dataset of handwritten school essays in Russian\Dataset of handwritten school essays in Russian\handwritten_essay\train\0\0.png"
pipeline = Pipeline(
    detector=EAST(),
    recognizer=TRBA(weights="trba_lite_g1"),
)

start = time.perf_counter()
result = pipeline.predict(image_path)
elapsed = time.perf_counter() - start

print(pipeline.get_text(result["page"]))

# visualize_page(image_path, result["page"], show_order=True).show()
print(f"{elapsed:.4f} сек")


text = pipeline.get_text(result["page"])
result["page"] = pipeline.correct_with_llm(
    result["page"], api_url="https://demo.ai.sfu-kras.ru/v1"
)
print(result["page"])
