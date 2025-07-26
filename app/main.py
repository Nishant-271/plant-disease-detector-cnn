import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Set page config
st.set_page_config(
    page_title="🌿 Plant Disease Detector",
    page_icon="🌱",
    layout="centered"
)

# Custom CSS Styling
st.markdown("""
    <style>
        .main-title {
            font-size: 2.7rem;
            font-weight: bold;
            text-align: center;
            padding: 1rem;
            margin-bottom: 1rem;
            border-radius: 1rem;
            background: linear-gradient(to right, #e6ffe6, #ccffcc);
            color: #1a1a40;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
        .result-box {
            font-size: 1.5rem;
            font-weight: bold;
            padding: 1rem;
            border-radius: 0.8rem;
            background: linear-gradient(to right, #d4fc79, #96e6a1);
            color: #1a1a40;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            text-align: center;
            margin-top: 1rem;
        }
        .upload-area {
            font-size: 1.1rem;
            margin-bottom: 1rem;
        }
        .stButton>button {
            background: linear-gradient(to right, #7de2fc, #b9fbc0);
            border: none;
            color: #1a1a40;
            font-size: 1.2rem;
            padding: 0.6rem 1.2rem;
            border-radius: 0.7rem;
            font-weight: bold;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            transition: 0.3s;
        }
        .stButton>button:hover {
            background: linear-gradient(to right, #a1ffce, #faffd1);
            color: #004d40;
            transform: scale(1.05);
        }
    </style>
""", unsafe_allow_html=True)

# Heading
st.markdown('<div class="main-title">🌿 Plant Disease Classifier</div>', unsafe_allow_html=True)

# Set working dir and load model
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
model = tf.keras.models.load_model(model_path)
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# Image preprocessor
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# Prediction function
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Upload section
st.markdown('<div class="upload-area">📤 Upload a plant leaf image to classify:</div>', unsafe_allow_html=True)
uploaded_image = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image.resize((150, 150)), caption="Uploaded Image", use_column_width=False)

    with col2:
        if st.button('🔍 Classify'):
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.markdown(f'<div class="result-box">🌱 Prediction: {prediction}</div>', unsafe_allow_html=True)
