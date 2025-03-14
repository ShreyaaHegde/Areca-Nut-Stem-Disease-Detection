import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st

# Load pre-trained model
model_path = "saved_models/IntelImageClass_ResNet20v1_model.017.keras"
model = load_model(model_path)

# Class names
class_names = [
    'stem cracking', 'Stem_bleeding', 
    'healthy_foot', 'Healthy_Trunk',
]

def predict_image(image):
    """Preprocess and predict the disease for the given image."""
    image_resized = cv2.resize(image, (150, 150))
    image_normalized = image_resized.astype('float32') / 255
    image_expanded = np.expand_dims(image_normalized, axis=0)
    
    # Get predictions
    predictions = model.predict(image_expanded)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)
    return class_names[predicted_class], confidence, predictions

# Streamlit interface
st.title("Arecanut Disease Classification")

# Description
st.write("""
This application uses a pre-trained deep learning model to classify arecanut diseases. 
Upload an image of an arecanut plant part to identify the disease or confirm its healthy state.
""")

# Upload image
uploaded_file = st.file_uploader("Upload an image of the arecanut plant", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and preprocess the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, channels="BGR", caption="Uploaded Image")
    
    # Predict using the pre-trained model
    predicted_disease, confidence, predictions = predict_image(image)
    st.write(f"**Predicted Disease:** {predicted_disease}")
    st.write(f"**Confidence:** {confidence:.2f}")
    
    # Display class probabilities
    st.write("**Class Probabilities:**")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {predictions[0][i]:.2f}")
