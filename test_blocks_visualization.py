"""Test script to verify block visualization in visualize_page function."""
from manuscript.detectors import EAST
from manuscript.utils import visualize_page

# Test image path
image_path = r"C:\Users\USER\Desktop\data02065\Archives020525\test_images\418.jpg"

# Run detector
print("Running EAST detector...")
result = EAST(weights="east_50_g1").predict(image_path)

# Check structure
print(f"Number of blocks detected: {len(result['page'].blocks)}")
for i, block in enumerate(result['page'].blocks):
    print(f"  Block {i}: {len(block.lines)} lines")

# Visualize with block visualization
print("\nGenerating visualization with show_order=True...")
vis = visualize_page(
    image_path, 
    result["page"], 
    show_order=True,
)

print("Displaying visualization...")
vis.show()
print("Done!")
