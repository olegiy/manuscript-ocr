from manuscript.detectors import EAST

# Model initialization
model = EAST(device="cuda")

# Path to the image
img_path = r"C:\Users\pasha\OneDrive\Рабочий стол\Фрагмент_ревизской_сказки_села_Курное_Волынской_губернии,_1857.jpg"

# Inference with visualization
result = model.predict(img_path, vis=True, sort_reading_order=True, profile=True)
page = result["page"]
img = result["vis_image"]

# Show the result
img.show()
