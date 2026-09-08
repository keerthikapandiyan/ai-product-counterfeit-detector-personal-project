from pathlib import Path
import sys

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


IMAGE_SIZE = 224

MODEL_PATH = Path("models/counterfeit_detector.keras")


def predict_image(image_path):
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Resize
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    # Convert to NumPy array
    image_array = np.array(image) / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Load model
    model = load_model(MODEL_PATH)

    # Make prediction
    prediction = model.predict(image_array, verbose=0)[0][0]

    if prediction >= 0.5:
        result = "Counterfeit"
        confidence = prediction * 100
    else:
        result = "Genuine"
        confidence = (1 - prediction) * 100

    print("Prediction:", result)
    print("Confidence:", f"{confidence:.2f}%")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Please provide an image path.")
        print("Example:")
        print("py src/predict.py data/train/genuine/genuine_001.jpg")
        sys.exit()

    image_path = sys.argv[1]

    predict_image(image_path)