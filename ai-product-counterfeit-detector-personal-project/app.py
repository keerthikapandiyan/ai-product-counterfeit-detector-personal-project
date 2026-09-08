from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model


IMAGE_SIZE = 224
MODEL_PATH = Path(__file__).parent / "models" / "counterfeit_detector.keras"


@st.cache_resource
def get_model():
	return load_model(MODEL_PATH)


st.set_page_config(
	page_title="AI Counterfeit Detector",
	page_icon="🔎",
	layout="centered",
)

st.title("AI Product Counterfeit Detector")
st.write("Upload a product image to check whether it appears genuine or counterfeit.")

if not MODEL_PATH.is_file():
	st.error(f"Model file not found: {MODEL_PATH}")
	st.info("Run `py src/train.py` first to create the trained model.")
	st.stop()

uploaded_file = st.file_uploader(
	"Choose a product image",
	type=["jpg", "jpeg", "png"],
)

if uploaded_file is None:
	st.info("Choose a JPG or PNG image to begin.")
else:
	image = Image.open(uploaded_file).convert("RGB")
	st.image(image, caption="Uploaded product", width="stretch")

	resized_image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
	image_array = np.array(resized_image, dtype=np.float32) / 255.0
	image_array = np.expand_dims(image_array, axis=0)

	prediction = get_model().predict(image_array, verbose=0)[0][0]

	if prediction >= 0.5:
		result = "Counterfeit"
		confidence = prediction * 100
		st.error(f"{result} | Confidence: {confidence:.2f}%")
	else:
		result = "Genuine"
		confidence = (1 - prediction) * 100
		st.success(f"{result} | Confidence: {confidence:.2f}%")
