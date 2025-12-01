from manuscript.detectors import EAST
from manuscript.utils import visualize_page, read_image

# Детекция
detector = EAST(weights="east_50_g1")

result = detector(r"C:\Users\USER\Desktop\IMG_9056.JPG")

# Визуализация с разными цветами для строк
img = read_image(r"C:\Users\USER\Desktop\IMG_9056.JPG")
vis = visualize_page(
    img, 
    result["page"], 
    show_order=True,  # Включить разные цвета для строк + номера
)
vis.show()

# Или без масштабирования
vis_full = visualize_page(img, result["page"], show_order=True, max_size=None)