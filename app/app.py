import streamlit as st
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="Synthetic Image Detection", layout="wide")

# App Title
st.title("Synthetic Image Detection")
st.write("An app to detect synthetic (AI-generated) images.")

# Sidebar
st.sidebar.header("Upload and Settings")
uploaded_image = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
detection_threshold = st.sidebar.slider("Detection Threshold", 0.0, 1.0, 0.5)
st.sidebar.write("Choose a threshold value to adjust sensitivity of the detector.")

# Function to simulate detection (Replace with actual detection function)
def detect_synthetic_image(image_np, threshold):
    import random
    probability_score = random.uniform(0, 1)
    detection_result = probability_score >= threshold
    return detection_result, probability_score

# Display Image and Result
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process image and display result
    image_np = np.array(image)
    detection_result, probability_score = detect_synthetic_image(image_np, detection_threshold)

    st.subheader("Detection Result")
    st.write("**Synthetic Image:**" if detection_result else "**Real Image:**")
    st.write(f"**Confidence Score:** {probability_score:.2f}")

    # Optional: Display Annotation
    if detection_result:
        annotated_image = image_np.copy()
        cv2.rectangle(annotated_image, (50, 50), (200, 200), (255, 0, 0), 2)  # Placeholder
        st.image(annotated_image, caption="Annotated Image", use_column_width=True)

# Footer
st.markdown("---")
st.markdown("### About")
st.markdown("This app uses AI-based models to detect synthetic (AI-generated) images. Adjust the threshold to control sensitivity.")
