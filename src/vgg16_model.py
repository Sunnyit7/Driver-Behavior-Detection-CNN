
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Flatten,
    Dense,
    Dropout
)

from tensorflow.keras.applications import VGG16

from config import (
    IMAGE_SIZE,
    NUM_CLASSES
)

def build_vgg16():
    base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
        )
    )
    for layer in base_model.layers:
        layer.trainable = False
    model = Sequential(name="VGG16_TransferLearning")

    model.add(base_model)

    model.add(Flatten())

    model.add(
        Dense(
            256,
            activation="relu"
        )
    )

    model.add(
        Dropout(0.5)
    )

    model.add(
        Dense(
            NUM_CLASSES,
            activation="softmax"
        )
    )

    return model
    
    


if __name__ == "__main__":

    model = build_vgg16()

    model.summary()   