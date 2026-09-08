from pathlib import Path
from PIL import Image
import numpy as np

IMAGE_SIZE = 224


def load_dataset(data_dir):
    data_dir = Path(data_dir)

    images = []
    labels = []

    classes = {
        "genuine": 0,
        "counterfeit": 1
    }

    for class_name, label in classes.items():

        class_dir = data_dir / class_name

        for image_path in class_dir.glob("*"):

            try:
                image = Image.open(image_path).convert("RGB")
                image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

                image_array = np.array(image) / 255.0

                images.append(image_array)
                labels.append(label)

            except Exception as e:
                print(f"Could not load {image_path}: {e}")

    return np.array(images), np.array(labels)