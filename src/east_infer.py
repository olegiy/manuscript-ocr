from manuscript.detectors import EAST

# Model initialization
model = EAST(weights_path=r"east_model.onnx", device="cuda")

# Path to the image
img_path = r"C:\Users\USER\Desktop\IMG_9056.JPG"

# Inference with visualization
result = model.predict(img_path, vis=True, sort_reading_order=True, profile=True)
page = result["page"]
img = result["vis_image"]

# Show the result
img.show()
