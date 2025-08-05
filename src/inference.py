import os
import numpy as np
from tensorflow.keras.models import load_model
from src.preprocess import preprocess_image

MODEL_PATH = os.path.join("models", "butterfly_model.keras")

CLASS_NAMES = ['SOUTHERN DOGFACE', 'ADONIS', 'BROWN SIPROETA', 'MONARCH',
       'GREEN CELLED CATTLEHEART', 'CAIRNS BIRDWING',
       'EASTERN DAPPLE WHITE', 'RED POSTMAN', 'MANGROVE SKIPPER',
       'BLACK HAIRSTREAK', 'CABBAGE WHITE', 'RED ADMIRAL', 'PAINTED LADY',
       'PAPER KITE', 'SOOTYWING', 'PINE WHITE', 'PEACOCK',
       'CHECQUERED SKIPPER', 'JULIA', 'COMMON WOOD-NYMPH', 'BLUE MORPHO',
       'CLOUDED SULPHUR', 'STRAITED QUEEN', 'ORANGE OAKLEAF',
       'PURPLISH COPPER', 'ATALA', 'IPHICLUS SISTER', 'DANAID EGGFLY',
       'LARGE MARBLE', 'PIPEVINE SWALLOW', 'BLUE SPOTTED CROW',
       'RED CRACKER', 'QUESTION MARK', 'CRIMSON PATCH', 'BANDED PEACOCK',
       'SCARCE SWALLOW', 'COPPER TAIL', 'GREAT JAY', 'INDRA SWALLOW',
       'VICEROY', 'MALACHITE', 'APPOLLO', 'TWO BARRED FLASHER',
       'MOURNING CLOAK', 'TROPICAL LEAFWING', 'POPINJAY', 'ORANGE TIP',
       'GOLD BANDED', 'BECKERS WHITE', 'RED SPOTTED PURPLE',
       'MILBERTS TORTOISESHELL', 'SILVER SPOT SKIPPER', 'AMERICAN SNOOT',
       'AN 88', 'ULYSES', 'COMMON BANDED AWL', 'CRECENT', 'METALMARK',
       'SLEEPY ORANGE', 'PURPLE HAIRSTREAK', 'ELBOWED PIERROT',
       'GREAT EGGFLY', 'ORCHARD SWALLOW', 'ZEBRA LONG WING', 'WOOD SATYR',
       'MESTRA', 'EASTERN PINE ELFIN', 'EASTERN COMA',
       'YELLOW SWALLOW TAIL', 'CLEOPATRA', 'GREY HAIRSTREAK',
       'BANDED ORANGE HELICONIAN', 'AFRICAN GIANT SWALLOWTAIL',
       'CHESTNUT', 'CLODIUS PARNASSIAN']

_model = None
def load_trained_model():
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH)
    return _model

def predict_image(img_path):
    model = load_trained_model()
    img_array = preprocess_image(img_path)

    print(f"Predicting image with shape: {img_array.shape}")
    
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    
    return CLASS_NAMES[class_idx], predictions[0]
