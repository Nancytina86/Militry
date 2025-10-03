import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from PIL import Image

# Load YOLO model
model = YOLO("C:/Users/hp/Downloads/epoch199.pt")  # <-- change if on Colab

# Initialize OCR reader
reader = easyocr.Reader(['en'])

st.title("🖼️ Object Detection + OCR App")
st.write("Upload an image to detect objects and extract text.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # Run YOLO detection
    results = model(img_array)

    # Draw detections
    annotated = results[0].plot()  # YOLO returns numpy array with boxes

    # OCR on the image
    ocr_results = reader.readtext(img_array)

    # Show image with detections
    st.image(annotated, caption="Detected Objects", use_column_width=True)

    # Show OCR results
    st.subheader("📜 OCR Extracted Text")
    if ocr_results:
        for (bbox, text, prob) in ocr_results:
            st.write(f"**{text}** (confidence: {prob:.2f})")
    else:
        st.write("No text detected.")
