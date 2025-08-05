import sys
import os
import streamlit as st
from PIL import Image
import tempfile
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference import predict_image, CLASS_NAMES

def show():
    st.title("🦋 Butterfly Species Classifier")
    st.write("Upload an image of a butterfly and let the model predict its species.")

    uploaded_file = st.file_uploader("Upload Butterfly Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            img.save(tmp.name, format="JPEG")
            label, probs = predict_image(tmp.name)

        st.success(f"Prediction: **{label}**")

        st.subheader("Confidence Scores")
        fig, ax = plt.subplots(figsize=(8, 5))
        top_k = 5  
        indices = np.argsort(probs)[::-1][:top_k]
        ax.barh([CLASS_NAMES[i] for i in indices], probs[indices], color='skyblue')
        ax.set_xlabel("Probability")
        ax.set_title("Top Predictions")
        plt.gca().invert_yaxis()
        st.pyplot(fig)
