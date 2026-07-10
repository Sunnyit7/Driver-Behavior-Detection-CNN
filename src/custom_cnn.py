"""
Custom CNN Model
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
from config import IMAGE_SIZE, NUM_CLASSES

def build_custom_cnn():
    """
    Builds and returns a custom CNN model.

    Returns:
        tensorflow.keras.Model
    """
    model = Sequential(name="Custom_CNN")
    model.add(
    Conv2D(
        filters=32,
        kernel_size=(3,3),
        activation="relu",
        input_shape=(
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3
            )
        )
    )
    model.add(
    MaxPooling2D(
        pool_size=(2,2)
        )
    )
    model.add(
    Conv2D(
        filters=64,
        kernel_size=(3,3),
        activation="relu"
            )
        )

    model.add(
        MaxPooling2D(
            pool_size=(2,2)
        )
        )
    model.add(
    Conv2D(
        filters=128,
        kernel_size=(3,3),
        activation="relu"
        )
    )

    model.add(
        MaxPooling2D(
            pool_size=(2,2)
        )
    )
    model.add(
    Flatten()
    )
    model.add(
    Dense(
        units=256,
        activation="relu"
        )
    )
    model.add(
    Dropout(0.5)
    )
    model.add(
    Dense(
        units=NUM_CLASSES,
        activation="softmax"
        )
    )
    return model
    
if __name__ == "__main__":

    model = build_custom_cnn()

    model.summary()