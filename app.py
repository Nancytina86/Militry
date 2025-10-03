import os
import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.title("🖼️ Object Detection + OCR App")

# Path to your YOLO model
MODEL_PATH = r"C:/Users/nancy/OneDrive/Pictures/epoch199.pt"
model = YOLO(MODEL_PATH)

# Optional: local EasyOCR model directory (set this if you've pre-downloaded)
LOCAL_EASYOCR_MODELS = r"C:\easyocr_models"  # <-- set to your folder or None

# Try to initialize EasyOCR reader, otherwise prepare fallback
reader = None
use_easyocr = False
try:
    import easyocr
    # If you have a local model dir, pass it; otherwise omit to allow downloads
    if LOCAL_EASYOCR_MODELS and os.path.exists(LOCAL_EASYOCR_MODELS):
        reader = easyocr.Reader(['en'], model_storage_directory=LOCAL_EASYOCR_MODELS)
    else:
        # This may attempt to download; if your network is blocked it will raise URLError
        reader = easyocr.Reader(['en'])
    use_easyocr = True
    st.write("Using EasyOCR for text extraction.")
except Exception as e:
    st.warning(f"EasyOCR unavailable: {e}")
    st.info("Falling back to Tesseract OCR (pytesseract). Install Tesseract binary and `pytesseract` python package.")
    try:
        import pytesseract
        from pytesseract import Output
        use_easyocr = False
        # If Tesseract executable not in PATH, set pytesseract.pytesseract.tesseract_cmd:
        # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    except Exception as e2:
        st.error("Neither EasyOCR nor pytesseract is available. Install one of them and restart the app.")
        st.stop()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is None:
    st.write("Upload an image to detect objects and extract text.")
else:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # YOLO detection
    try:
        results = model(img_array)
        annotated = results[0].plot()
        st.image(annotated, caption="Detected Objects", use_column_width=True)
    except Exception as e:
        st.error(f"YOLO error: {e}")
        st.image(image, caption="Original image (YOLO failed)")

    st.subheader("📜 OCR Extracted Text")
    if use_easyocr and reader is not None:
        try:
            ocr_results = reader.readtext(img_array)
            if ocr_results:
                for (bbox, text, prob) in ocr_results:
                    st.write(f"**{text}** (confidence: {prob:.2f})")
            else:
                st.write("No text detected by EasyOCR.")
        except Exception as e:
            st.error(f"EasyOCR reading failed: {e}")
    else:
        # pytesseract fallback
        import pytesseract
        try:
            text = pytesseract.image_to_string(image)
            if text.strip():
                st.text(text)
            else:
                st.write("No text detected by pytesseract.")
        except Exception as e:
            st.error(f"pytesseract failed: {e}")
