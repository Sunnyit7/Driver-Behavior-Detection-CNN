"""
Flask Web Application

Deploys the best-performing VGG16 model for
driver behavior classification.
"""
import uuid

import os
import numpy as np
from werkzeug.utils import secure_filename

from flask import (
    Flask,
    render_template,
    request,
    url_for
)

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array


app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(
    "static",
    "uploads"
)

app.config[
    "UPLOAD_FOLDER"
] = UPLOAD_FOLDER

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)
MODEL_PATH = os.path.join(
    "saved_models",
    "vgg16_best.keras"
)

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "other_activities",
    "safe_driving",
    "talking_phone",
    "texting_phone",
    "turning"
]
ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg"
}
print("Loading VGG16 model...")

model = load_model(
    MODEL_PATH
)

print("VGG16 model loaded successfully!")
def allowed_file(filename):
    """
    Checks whether the uploaded file has
    a supported image extension.
    """

    return (
        "." in filename
        and
        filename.rsplit(
            ".",
            1
        )[1].lower()
        in ALLOWED_EXTENSIONS
    )

def preprocess_image(image_path):
    """
    Loads and preprocesses an image for VGG16 prediction.
    """

    image = load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    image_array = img_to_array(
        image
    )

    image_array = (
        image_array / 255.0
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array

@app.route("/")
def home():
    return render_template("index.html")

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # Check whether an image was sent
    if "image" not in request.files:

        return render_template(
            "index.html",
            error="No image was uploaded."
        )

    image_file = request.files[
        "image"
    ]

    # Check whether a file was selected
    if image_file.filename == "":

        return render_template(
            "index.html",
            error="Please select an image."
        )
    if not allowed_file(
    image_file.filename
    ):

        return render_template(
        "index.html",
        error=(
            "Invalid file type. "
            "Please upload a JPG, JPEG, or PNG image."
        )
    )

    # Create a safe filename
    original_filename = secure_filename(
        image_file.filename
    )

    file_extension = original_filename.rsplit(
        ".",
        1
    )[1].lower()

    filename = (
        f"{uuid.uuid4().hex}."
        f"{file_extension}"
    )

    # Create the complete image path
    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    # Save uploaded image
    image_file.save(
        image_path
    )

    # Preprocess image
    processed_image = preprocess_image(
        image_path
    )

    # Generate class probabilities
    predictions = model.predict(
        processed_image,
        verbose=0
    )

    # Get index of highest probability
    predicted_index = np.argmax(
        predictions[0]
    )

    # Convert class index into class name
    predicted_class = CLASS_NAMES[
        predicted_index
    ]
    display_class = predicted_class.replace(
    "_",
    " "
        ).title()

    # Get confidence percentage
    confidence = (
        predictions[0][predicted_index]
        * 100
    )

    return render_template(
        "index.html",
        prediction=display_class,
        confidence=confidence,
        image_path=url_for(
        "static",
        filename=f"uploads/{filename}"
            )
        )


if __name__ == "__main__":
    app.run(
        debug=True
    )