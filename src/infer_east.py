from manuscript.detectors import EAST
from manuscript.utils import visualize_page


image_path = r"C:\Users\pasha\OneDrive\Рабочий стол\h1001_w1001_087aa126e878513ee8791757665f7f3c.jpg"

result = EAST(weights="east_50_g1").predict(image_path)

vis = visualize_page(
    image_path,
    result["page"],
    show_order=True,
)

vis.show()
