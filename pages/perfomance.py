import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import os
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

def show():
    st.title("📈 Model Performance Dashboard")
    st.write("Visualize model training metrics and evaluation results.")
   
    HISTORY_PATH = "models/history.json"  

    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r") as f:
            history = json.load(f)
    else:
        st.error("History JSON file not found. Please run training first.")
        st.stop()

    st.subheader("Training History")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(history['accuracy'], label='Train')
    ax[0].plot(history['val_accuracy'], label='Validation')
    ax[0].set_title('Accuracy')
    ax[0].legend()

    ax[1].plot(history['loss'], label='Train Loss')
    ax[1].plot(history['val_loss'], label='Validation Loss')
    ax[1].set_title('Loss')
    ax[1].legend()

    st.pyplot(fig)

