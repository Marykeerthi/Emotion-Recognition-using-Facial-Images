
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Load the trained model
model = tf.keras.models.load_model("happy_not_happy_cnn.h5")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Constants
IMG_SIZE = (48, 48)

# Streamlit UI
st.title("😄 Happy or Not Happy Classifier")
st.write("Upload an image or take a picture to see if the person is happy or not.")

# Choose input method
option = st.radio("Choose input method:", ["Upload Image", "Use Webcam"])

# Handle input
image = None
if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

elif option == "Use Webcam":
    camera_image = st.camera_input("Take a picture")
    if camera_image:
        image = Image.open(camera_image).convert("RGB")

# Predict and display
if image:
    st.image(image, caption="Input Image", use_column_width=True)
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = "😊 Happy" if prediction > 0.5 else "😐 Not Happy"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.markdown(f"### Prediction: {label}")
    st.markdown(f"**Confidence:** {confidence:.2%}")
