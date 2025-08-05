import os
from src.data_loader import load_data, create_generators
from src.train import build_model, train_model

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
SAVE_DIR = "models"

def main():
    TRAIN_CSV = os.path.join("data", "Training_set.csv")
    TEST_CSV = os.path.join("data", "Testing_set.csv")
    TRAIN_DIR = os.path.join("data", "train")
    TEST_DIR = os.path.join("data", "test")

    print("[INFO] Loading data...")
    train_df, test_df = load_data(TRAIN_CSV, TEST_CSV)

    print("[INFO] Creating generators...")
    train_gen, val_gen, test_gen, class_names = create_generators(
        train_df, test_df, TRAIN_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE
    )

    print(f"[INFO] Found {len(class_names)} classes.")

    print("[INFO] Building model...")
    model = build_model(input_shape=IMG_SIZE + (3,), num_classes=len(class_names))

    print("[INFO] Training model...")
    model, history = train_model(
        model,
        train_gen,
        val_gen,
        class_names,
        epochs=EPOCHS,
        save_dir=SAVE_DIR
    )

    print("[INFO] ✅ Model training completed and saved!")

if __name__ == "__main__":
    main()
