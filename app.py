import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("dog_cat_model.h5", compile=False)

st.title("Dog vs Cat Prediction")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", width=500)

    img = img.resize((224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    try:
    prediction = model.predict(img_array)

    st.write("Raw Prediction:", prediction)

    pred_value = prediction[0][0]

    if pred_value > 0.5:
        st.success("Prediction: Cat 🐱")
    else:
        st.success("Prediction: Dog 🐶")

except Exception as e:
    st.error(f"Error: {e}")
