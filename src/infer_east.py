from manuscript.detectors import EAST
from manuscript.utils import visualize_page


image_path = r"/Users/user/manuscript-ocr/example/ocr_example_image.jpg"

result = EAST(weights="east_50_g1", device='coreml').predict(image_path)

vis = visualize_page(
    image_path,
    result["page"],
    show_order=True,
)

vis.show()
