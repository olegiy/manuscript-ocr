import time
from manuscript import Pipeline
from manuscript.detectors import EAST
from manuscript.recognizers import TRBA
from manuscript.utils.visualization import visualize_page

image_path = r"example/ocr_example_image.jpg"
pipeline = Pipeline(detector=EAST(device="coreml"), recognizer=TRBA(weights="trba_base_g1", device="coreml"))

start = time.perf_counter()
result = pipeline.predict(image_path)
elapsed = time.perf_counter() - start

print(pipeline.get_text(result["page"]))

visualize_page(image_path, result["page"], show_order=True).show()
print(f"{elapsed:.4f} сек")