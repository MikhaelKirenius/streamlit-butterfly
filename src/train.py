import os
import json
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

def build_model(input_shape=(224, 224, 3), num_classes=75, lr=0.0001):
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_generator, val_generator, class_names, epochs=20, save_dir='models'):
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'butterfly_model.keras')

    earlystop_cb = EarlyStopping(patience=5, restore_best_weights=True)
    checkpoint_cb = ModelCheckpoint(model_path, save_best_only=True)

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=[earlystop_cb, checkpoint_cb]
    )

    class_file = os.path.join(save_dir, "classes.txt")
    with open(class_file, "w") as f:
        for cls in class_names:
            f.write(cls + "\n")

    history_file = os.path.join(save_dir, "history.json")
    with open(history_file, "w") as f:
        json.dump(history.history, f)

    print(f"✅ Model saved to {model_path}")
    print(f"✅ Classes saved to {class_file}")
    print(f"✅ Training history saved to {history_file}")

    return model, history
