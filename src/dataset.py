
from pathlib import Path
import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import (
    IMAGE_SIZE,
    BATCH_SIZE,
    CLASS_NAMES
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw dataset path
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "Revitsone-5classes"

def verify_images():
    """
    Verify that all images in the dataset are readable.

    Returns:
        bad_images (list): List of corrupted image paths.
    """

    bad_images = []

    for class_name in CLASS_NAMES:

        class_path = DATASET_PATH / class_name

        # Skip if folder doesn't exist
        if not class_path.exists():
            print(f"Warning: {class_name} folder not found.")
            continue

        for image_name in os.listdir(class_path):

            image_path = class_path / image_name

            try:
                with Image.open(image_path) as img:
                    img.verify()

            except Exception:
                bad_images.append(image_path)

    return bad_images




def create_dataframe():
    """
    Create a DataFrame containing image file paths and labels.

    Returns:
        pd.DataFrame: DataFrame with columns:
            - filepath
            - label
    """

    data = []

    for class_name in CLASS_NAMES:

        class_path = DATASET_PATH / class_name

        if not class_path.exists():
            continue

        for image_name in os.listdir(class_path):

            image_path = class_path / image_name

            data.append({
                "filepath": str(image_path),
                "label": class_name
            })

    df = pd.DataFrame(data)

    return df


    

    



def split_dataset(df):
    """
    Split the dataset into train, validation and test sets.

    Args:
        df (pd.DataFrame): Complete dataset.

    Returns:
        train_df, validation_df, test_df
    """

    # First split: 70% train, 30% temporary
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["label"]
    )

    # Second split: 15% validation, 15% test
    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["label"]
    )

    return train_df, validation_df, test_df

def create_generators(train_df,validation_df,test_df):
    train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=20,

    zoom_range=0.2,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True
    )


    test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
    )
    train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="filepath",
    y_col="label",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
    )
    validation_generator = test_datagen.flow_from_dataframe(
    dataframe=validation_df,
    x_col="filepath",
    y_col="label",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
    )
    test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="filepath",
    y_col="label",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
    )
    return train_generator, validation_generator, test_generator

def load_dataset():
    """
    Complete dataset pipeline.

    Returns:
        train_generator,
        validation_generator,
        test_generator
    """

    # Verify images
    bad_images = verify_images()

    if bad_images:
        print(f"Warning: {len(bad_images)} corrupted images found.")

    # Create dataframe
    df = create_dataframe()

    # Split dataset
    train_df, validation_df, test_df = split_dataset(df)

    # Create generators
    train_generator, validation_generator, test_generator = create_generators(
        train_df,
        validation_df,
        test_df
    )

    return train_generator, validation_generator, test_generator


def main():

    train_generator, validation_generator, test_generator = load_dataset()

    print("\nDataset Pipeline Executed Successfully!")
    print("-" * 40)

    print("Training Samples   :", train_generator.samples)
    print("Validation Samples :", validation_generator.samples)
    print("Testing Samples    :", test_generator.samples)
if __name__ == "__main__":
    main()