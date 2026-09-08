from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMAGE_SIZE = 224


def preprocess_bgr(image: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Resize an image and improve lighting consistency."""

    if image is None:
        raise ValueError("Image could not be read. Use JPG or PNG format.")

    image = cv2.resize(
        image,
        (size, size),
        interpolation=cv2.INTER_AREA
    )

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab_image)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    improved_l_channel = clahe.apply(l_channel)

    improved_image = cv2.merge(
        (improved_l_channel, a_channel, b_channel)
    )

    return cv2.cvtColor(
        improved_image,
        cv2.COLOR_LAB2RGB
    )


def load_preprocessed(image_path: str | Path) -> Image.Image:
    """Read an image file and return a cleaned RGB image."""

    image = cv2.imread(str(image_path))

    return Image.fromarray(
        preprocess_bgr(image)
    )


def bytes_to_preprocessed(image_bytes: bytes) -> Image.Image:
    """Read uploaded image bytes and return a cleaned RGB image."""

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    return Image.fromarray(
        preprocess_bgr(image)
    )