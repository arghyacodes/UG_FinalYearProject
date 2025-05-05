# pip install -r requirements.txt


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image

# Load the pre-trained model
model = load_model('trained_model1.keras')

# List of categories
data_cat = ['Azadirachta indica', 'Justicia adhatoda', 'Mentha arvensis', 
            'Moringa oleifera', 'Weed']

# Set image dimensions for prediction
img_height = 128
img_width = 128

# Streamlit app layout
st.title("Medicinal Leaf Classification")
st.write("Upload an image of a medicinal leaf, and the model will classify it.")

# Media picker for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    # Resize the image to match model input size
    image = image.resize((img_width, img_height))
    
    # Convert the image to a numpy array
    img_arr = np.array(image)
    
    # Check if the image is RGB, if not convert it to RGB
    if img_arr.ndim != 3 or img_arr.shape[2] != 3:
        img_arr = np.stack([img_arr] * 3, axis=-1)
    
    # Preprocess the image (expand dimensions to match model input)
    img_bat = tf.expand_dims(img_arr, 0)  # Add batch dimension
    
    # Predict with the model
    predict = model.predict(img_bat)
    
    # Get prediction scores
    score = tf.nn.softmax(predict)
    
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Display the prediction result
    st.write('Medicinal leaf in this image is: ' + data_cat[np.argmax(score)])
    st.write('With accuracy of: ' + str(np.max(score) * 100) + '%')