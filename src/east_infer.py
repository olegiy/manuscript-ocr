from manuscript.detectors import EAST

# Model initialization
model = EAST(device="cuda")

# Path to the image
img_path = r"C:\Users\USER\Desktop\photo_2025-10-15_19-34-17.jpg"

# Inference with visualization
result = model.predict(img_path, vis=True, sort_reading_order=True, profile=True)
page = result["page"]
img = result["vis_image"]

# Show the result
img.show()
