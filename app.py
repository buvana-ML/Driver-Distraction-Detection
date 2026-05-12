import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load model
model = load_model("driver_model.h5", compile=False)

# Class labels
labels = [
    "Safe Driving",
    "Texting Right",
    "Talking Phone Right",
    "Texting Left",
    "Talking Phone Left",
    "Operating Radio",
    "Drinking",
    "Reaching Behind",
    "Hair/Makeup",
    "Talking to Passenger"
]

# Title
st.title("🚗 Driver Distraction Detection System")

st.write("Upload an image of a driver, and the AI model will predict the activity.")

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image", width="stretch")

    # Preprocess
    img_resized = cv2.resize(img, (224,224))
    img_resized = img_resized / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    # Prediction
    pred = model.predict(img_resized)
    class_id = np.argmax(pred)
    confidence = np.max(pred)

    # Output
    st.subheader("🔍 Prediction")
    st.write(f"**Activity:** {labels[class_id]}")
    st.write(f"**Confidence:** {round(confidence, 2)}")

    # Safety message
    if labels[class_id] == "Safe Driving":
        st.success("✅ Driver is Safe")
        st.info("Model is less confident because safe driving has fewer distinctive features.")
    else:
        st.error("⚠️ Driver is Distracted")