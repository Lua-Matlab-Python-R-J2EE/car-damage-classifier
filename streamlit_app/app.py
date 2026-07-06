import sys
from pathlib import Path

# Setup paths cleanly for local and cloud imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
print(ROOT)
print(sys.path)
print("--------------------------------------")

import streamlit as st
from PIL import Image
from core.model_utils import classify_car_damage
print("--------------------------------------")

# Title
st.title("Car Damage Classifier")

# Get user input
uploaded_file = st.file_uploader(
    label="Upload the image file:",
    accept_multiple_files=False,
    type=["jpg", "jpeg", "png"]
)

# Check if file was actually uploaded
if uploaded_file is not None:
    # Print image attributes
    # print(f"NAME: {uploaded_file.name}")
    # print(f"TYPE: {uploaded_file.type}")
    # print(f"SIZE: {uploaded_file.size}")

    # Open the image file using PIL
    image = Image.open(uploaded_file)

    # NOT NEEDED to convert to RGB mode
    # bec transformation function does it

    # Display the image using the modern layout standard
    st.image(image, caption="User uploaded file", use_container_width=True)

    # Continue
    st.success("...user image ready for processing...")

    # Now pass prepared image to model with a loading indicator
    with st.spinner("Processing..."):
        prediction = classify_car_damage(image)
        
    st.success(f"Predicted Class is: {prediction}")
