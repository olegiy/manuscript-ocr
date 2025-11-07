import time
from manuscript.recognizers import TRBA


# === Инициализация модели ===
recognizer = TRBA(model_path = r"C:\Users\USER\Desktop\OCR_MODELS\exp_2\best_acc_weights.pth", 
config_path = r"C:\Users\USER\Desktop\OCR_MODELS\exp_2\config.json")

# === Список изображений ===
images = [
    r"C:\Users\USER\Desktop\archive_25_09\dataset\handwritten\val\img\images_group_7_1261_4240.png",
]*16

# === Измеряем время ===
start_time = time.perf_counter()
import numpy as np

res = recognizer.predict(
    images=images, batch_size=64, mode="greedy"  # 'greedy' or 'beam'
)
end_time = time.perf_counter()

# === Вывод результатов ===
total_time = end_time - start_time
avg_time = total_time / len(images)
fps = 1.0 / avg_time if avg_time > 0 else float("inf")

print("\n=== Результаты распознавания ===")
for result in res:
    text = result["text"]
    score = result["confidence"]
    print(f"Recognized: {text}, confidence: {score:.4f}")

print(f"\nProcessed {len(images)} images in {total_time:.3f} sec")
print(f"Average per image: {avg_time:.3f} sec ({fps:.1f} FPS)")


res = recognizer.predict(
    images=images, batch_size=64, mode="beam"  # 'greedy' or 'beam'
)
end_time = time.perf_counter()

# === Вывод результатов ===
total_time = end_time - start_time
avg_time = total_time / len(images)
fps = 1.0 / avg_time if avg_time > 0 else float("inf")

print("\n=== Результаты распознавания ===")
for result in res:
    text = result["text"]
    score = result["confidence"]
    print(f"Recognized: {text}, confidence: {score:.4f}")

print(f"\nProcessed {len(images)} images in {total_time:.3f} sec")
print(f"Average per image: {avg_time:.3f} sec ({fps:.1f} FPS)")
