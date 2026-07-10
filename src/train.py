
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from dataset import load_dataset
from model_factory import get_model

from config import (
    EPOCHS,
    LEARNING_RATE
    
)
MODEL_NAME = "resnet50"
def main():

    print("=" * 50)
    print("Loading Dataset...")
    print("=" * 50)

    train_generator, validation_generator, test_generator = load_dataset()
    print("\n" + "=" * 50)
    
    print(f"Building {MODEL_NAME.upper()} Model.....")
    print("=" * 50)

    
    model = get_model(MODEL_NAME)
    model.summary()
    print("\n" + "=" * 50)
    print("Compiling Model...")
    print("=" * 50)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    checkpoint = ModelCheckpoint(
    filepath=f"../saved_models/{MODEL_NAME}_best.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=3,
    verbose=1
    )
    early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
    )

    print("\n" + "=" * 50)
    print(f"Training Model {MODEL_NAME.upper()}...")
    print("=" * 50)

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=[
        checkpoint,
        early_stop,
        reduce_lr
        ]
    )
    print("\nSaving Final Model...")

    model.save(f"../saved_models/{MODEL_NAME}_final.keras")

    print("Final model saved successfully!")



    print("Model compiled successfully!")

if __name__ == "__main__":
    main()