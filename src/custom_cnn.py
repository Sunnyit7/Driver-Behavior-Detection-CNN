import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

def build_custom_cnn():

    model = Sequential()

    model.add(
        layers.Conv2D(
            32,
            (3,3),
            activation="relu",
            input_shape=(224,224,3)
        )
    )

    model.add(layers.MaxPooling2D())

    model.add(layers.Conv2D(64,(3,3),activation="relu"))

    model.add(layers.MaxPooling2D())

    model.add(layers.Conv2D(128,(3,3),activation="relu"))

    model.add(layers.MaxPooling2D())

    model.add(layers.Flatten())

    model.add(layers.Dense(256,activation="relu"))

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(5,activation="softmax"))

    return model