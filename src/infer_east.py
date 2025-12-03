from manuscript.detectors import EAST
from manuscript.utils import visualize_page


image_path = r"C:\Users\USER\Desktop\data02065\Archives020525\test_images\418.jpg"

result = EAST(weights="east_50_g1").predict(image_path)

vis = visualize_page(
    image_path, 
    result["page"], 
    show_order=True,
)

vis.show()