import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the trained model
def load_model():
    model = tf.keras.models.load_model("vgg19_from_scratch.keras")
    return model

model = load_model()

# Detect the model input size automatically (ignores None for batch dim)
_, height, width, channels = model.input_shape

# Set Streamlit page settings
st.set_page_config(page_title="Mask Detection", layout="centered")
st.title("😷 Mask Detection CNN")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img_size = (width, height)  # automatically matches model input
    image = image.convert("RGB")  # Ensure 3 channels
    img = image.resize(img_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, H, W, 3)

    # Debug: show shapes
    st.write("Model expects:", model.input_shape)
    st.write("Input image shape:", img_array.shape)

    # Predict
    prediction = model.predict(img_array)[0][0]
    label = "😷 Mask" if prediction >= 0.5 else "❌ No Mask"
    confidence = prediction if prediction >= 0.5 else 1 - prediction

    # Show result
    st.subheader("Prediction:")
    st.markdown(f"{label} (Confidence: {confidence:.2%})")