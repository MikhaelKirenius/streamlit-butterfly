from PIL import Image
import numpy as np

def preprocess_image(img_path, target_size=(224, 224)):
    """Load and preprocess image for model prediction"""
    img = Image.open(img_path).convert('RGB').resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array
