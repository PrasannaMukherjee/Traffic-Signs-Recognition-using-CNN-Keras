import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

CLASS_NAMES = [
    "Speed limit (20km/h)",
    "Speed limit (30km/h)",
    "Speed limit (50km/h)",
    "Speed limit (60km/h)",
    "Speed limit (70km/h)",
    "Speed limit (80km/h)",
    "End of speed limit (80km/h)",
    "Speed limit (100km/h)",
    "Speed limit (120km/h)",
    "No passing",
    "No passing for >3.5t",
    "Right-of-way at intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles >3.5t prohibited",
    "No entry",
    "General caution",
    "Dangerous curve left",
    "Dangerous curve right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing >3.5t",
]

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.h5"


@st.cache_resource
def load_model(model_path: str):
    """Load and cache the trained model."""
    return tf.keras.models.load_model(model_path)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Decode, resize, and format uploaded bytes like the training pipeline."""
    np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    cv_image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)

    if cv_image is None:
        raise ValueError("Unable to decode image. Please upload a valid JPG/PNG file.")

    cv_image = cv2.resize(cv_image, (30, 30), interpolation=cv2.INTER_NEAREST)
    return np.expand_dims(cv_image.astype(np.float32), axis=0)


def main():
    st.set_page_config(page_title="Traffic Sign Recognition", page_icon="🚦")
    st.title("Traffic Sign Recognition")
    st.write(
        "Upload a photo of a traffic sign to classify it using the trained CNN "
        "model. Supported formats: JPG, JPEG, PNG."
    )

    model_path = st.sidebar.text_input("Model path", value=str(DEFAULT_MODEL_PATH))
    model_file = Path(model_path)
    model = None

    if model_file.is_file():
        with st.spinner("Loading model..."):
            model = load_model(str(model_file))
    else:
        st.sidebar.error(f"Model file not found at {model_file}")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file and model:
        file_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        batch = preprocess_image(file_bytes)

        with st.spinner("Running inference..."):
            prediction = model.predict(batch)

        predicted_class = int(np.argmax(prediction, axis=1)[0])
        confidence = float(np.max(prediction))
        label = CLASS_NAMES[predicted_class] if predicted_class < len(CLASS_NAMES) else f"Class {predicted_class}"

        st.success("Prediction complete!")
        st.metric("Predicted class", label, f"{confidence:.2%}")

        with st.expander("Raw probabilities"):
            st.write(prediction.tolist())
    elif not uploaded_file:
        st.info("Upload an image to start the prediction.")


if __name__ == "__main__":
    main()

