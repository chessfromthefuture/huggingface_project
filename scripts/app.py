import streamlit as st
from transformers import pipeline
from PIL import Image
import io

# Load the Hugging Face image classification model
classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

# Streamlit UI
st.title("Image Classifier with Hugging Face 🤗")
st.write("Upload an image, and the model will predict its content!")

# Upload file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run classification
    st.write("Classifying...")
    results = classifier(image)

    # Display results
    for result in results:
        st.write(f"**{result['label']}**: {result['score']:.4f}")
