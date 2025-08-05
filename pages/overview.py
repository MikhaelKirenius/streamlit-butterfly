import streamlit as st


def show():
    st.title("🦋 Butterfly Species Classification - Project Overview")
    st.write("""
    This application demonstrates a deep learning model to classify **75 butterfly species**
    using **Transfer Learning (MobileNetV2)**.

    ---
    ### 📂 Dataset
    - **Source:** Kaggle - Butterfly Image Classification
    - **Structure:**
        - `train/`: Training images with labels
        - `test/`: Testing images (unlabeled)
        - `Training_set.csv` & `Testing_set.csv` for metadata
    - **Classes:** 75 species

    ---
    ### 🧠 Workflow
    1. **Data Loading & Augmentation**
        - Load CSV metadata (`pandas`)
        - Use `ImageDataGenerator` for preprocessing & augmentation
    2. **Model Building**
        - Base Model: `MobileNetV2` (pretrained on ImageNet)
        - Custom Layers:
            - `GlobalAveragePooling2D`
            - `Dense(128, ReLU)`
            - `Dropout(0.5)`
            - `Dense(75, Softmax)`
    3. **Training**
        - Optimizer: `Adam`
        - Loss: `categorical_crossentropy`
        - Callbacks: `EarlyStopping`, `ModelCheckpoint`
        - Save Model in **Keras format**
    4. **Evaluation & Visualization**
        - Accuracy & Loss curves
        - Model performance metrics
        - Sample predictions on test images
    """)

    st.markdown("### ✅ Key Features")
    st.write("""
    - Transfer Learning with MobileNetV2
    - Modularized pipeline (Data Loader, Training, Preprocessing)
    - **Streamlit App** for inference
    - **Multi-page UI** for better navigation
    """)

    st.markdown("### 📈 Model Summary")
    st.write("""
    - **Final Train Accuracy:** ~81.6%
    - **Final Validation Accuracy:** ~86.2%
    - **Model Size:** ~11.1 MB
    """)

    st.info("💡 Navigate to the **Model Performance** page to see training curves and metrics.")
    

