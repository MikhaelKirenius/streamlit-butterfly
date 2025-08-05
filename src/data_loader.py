import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(train_csv, test_csv):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    return train_df, test_df

def create_generators(train_df, test_df, train_dir, test_dir, img_size=(224, 224), batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        directory=train_dir,
        x_col='filename',
        y_col='label',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        seed=42
    )

    val_generator = train_datagen.flow_from_dataframe(
        train_df,
        directory=train_dir,
        x_col='filename',
        y_col='label',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        seed=42
    )

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_dataframe(
        test_df,
        directory=test_dir,
        x_col='filename',
        target_size=img_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )

    class_names = list(train_generator.class_indices.keys())

    return train_generator, val_generator, test_generator, class_names
