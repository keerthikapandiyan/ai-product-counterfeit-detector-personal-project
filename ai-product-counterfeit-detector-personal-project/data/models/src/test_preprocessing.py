from pathlib import Path

from preprocessing import load_preprocessed


# Path to your genuine test image
image_path = Path("data/train/genuine/genuine_001.jpg")

# Load and preprocess the image
image = load_preprocessed(image_path)

# Display results
print("Image loaded successfully!")
print("Image size:", image.size)
print("Image mode:", image.mode)