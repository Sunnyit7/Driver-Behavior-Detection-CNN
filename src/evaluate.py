import os
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.metrics import precision_recall_fscore_support

from tensorflow.keras.models import load_model

from dataset import load_dataset
from config import CLASS_NAMES

MODEL_NAME = "vgg16"
def main():
    print("=" * 50)
    print("Loading Dataset...")
    print("=" * 50)
    print("=" * 50)
    print("Loading Dataset...")
    print("=" * 50)

    _, _, test_generator = load_dataset()
    print("\nLoading Model...")

    model = load_model(
        f"saved_models/{MODEL_NAME}_best.keras"
    )

    print("Model Loaded Successfully!")
    OUTPUT_DIR = f"outputs/{MODEL_NAME}"

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )
    print("\n" + "=" * 50)
    print("Evaluating Model...")
    print("=" * 50)

    test_loss, test_accuracy = model.evaluate(
        test_generator,
        verbose=1
    )
    print(f"\nTest Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_accuracy*100:.2f}%")
    
    test_generator.reset()
    print("\n" + "=" * 50)
    print("Generating Predictions...")
    print("=" * 50)

    predictions = model.predict(
        test_generator,
        verbose=1
    )
    y_pred = np.argmax(
    predictions,
    axis=1
    )
    y_true = test_generator.classes
    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
        )
    print("\n" + "=" * 50)
    print("Classification Report")
    print("=" * 50)

    print(report)
    with open(
    os.path.join(
        OUTPUT_DIR,
        "classification_report.txt"
    ),
    "w"
    ) as f:

        f.write(report)

    cm = confusion_matrix(
    y_true,
    y_pred
    )
    disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
    )

    fig, ax = plt.subplots(figsize=(8, 8))

    disp.plot(
        cmap="Blues",
        ax=ax,
        colorbar=False
    )

    plt.title(f"{MODEL_NAME} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.png"
    ),
    dpi=300
    )

    plt.close()
    precision, recall, f1, _ = precision_recall_fscore_support(
    y_true,
    y_pred,
    average="weighted"
    )
    print("\nOverall Metrics")
    print("-" * 40)
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }


    OUTPUT_DIR = f"outputs/{MODEL_NAME}"

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )
    with open(
    os.path.join(
        OUTPUT_DIR,
        "metrics.json"
    ),
    "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )
    
    

if __name__ == "__main__":
    main()