from custom_cnn import build_custom_cnn
from vgg16_model import build_vgg16
from resnet50_model import build_resnet50

def get_model(model_name):
    """
    Returns the requested model.

    Parameters
    ----------
    model_name : str

    Returns
    -------
    keras.Model
    """

    model_name = model_name.lower()

    if model_name == "custom_CNN":
        return build_custom_cnn()

    elif model_name == "vgg16":
        return build_vgg16()
    elif model_name == "resnet50":
        return build_resnet50()

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )


if __name__ == "__main__":

    model = get_model("resnet50")

    model.summary()