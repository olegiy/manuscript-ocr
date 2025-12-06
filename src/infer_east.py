from manuscript.detectors import EAST
from manuscript.utils import visualize_page


image_path = r"C:\shared\Archives020525\test_images\540.jpg"

result = EAST(weights="east_50_g1", axis_aligned_output=False).predict(image_path)

vis = visualize_page(
    image_path,
    result["page"],
    show_order=True,
)

vis.show()
