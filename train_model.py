"""Train a simple helmet / no-helmet classifier.

Expected dataset format:

dataset/
  helmet/
  no_helmet/
"""

import os

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator


DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "helmet_classifier.h5")


def train() -> None:
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
    )

    train_data = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(128, 128),
        batch_size=32,
        class_mode="binary",
        subset="training",
    )

    val_data = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(128, 128),
        batch_size=32,
        class_mode="binary",
        subset="validation",
    )

    model = models.Sequential([
        layers.Input(shape=(128, 128, 3)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(train_data, validation_data=val_data, epochs=12)
    model.save(MODEL_PATH)


if __name__ == "__main__":
    train()
