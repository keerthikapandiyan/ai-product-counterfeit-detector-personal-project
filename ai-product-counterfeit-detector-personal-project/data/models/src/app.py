import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

IMAGE_SIZE = 224
MODEL_PATH = "models/counterfeit_detector.keras"

model = load_model(MODEL_PATH)

st.title("🛍️ AI Product Counterfeit Detector")

st.write("Upload a product image to check the prediction.")

uploaded_file = st.file_uploader(
    "Choose a product image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Product", width="stretch")

    resized_image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    image_array = np.array(resized_image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array, verbose=0)[0][0]

    if prediction >= 0.5:
        result = "❌ Counterfeit"
        confidence = prediction * 100
    else:
        result = "✅ Genuine"
        confidence = (1 - prediction) * 100

    st.subheader("Prediction")
    st.write(result)
    st.write(f"Confidence: **{confidence:.2f}%**")
    