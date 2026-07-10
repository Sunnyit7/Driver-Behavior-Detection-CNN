import os
import json
import pandas as pd
import matplotlib.pyplot as plt

MODELS = [
    "custom_cnn",
    "vgg16"
]
OUTPUT_DIR = os.path.join(
    "outputs",
    "comparison"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

def main():

    print("=" * 50)
    print("Comparing Model Performance")
    print("=" * 50)

    results = []
    for model_name in MODELS:

        metrics_path = os.path.join(
            "outputs",
            model_name,
            "metrics.json"
        )

        print(
            f"\nLoading metrics for: "
            f"{model_name}"
        )

        with open(
            metrics_path,
            "r"
        ) as f:

            metrics = json.load(f)

        results.append(
            {
            "Model": model_name,
            "Test Loss": metrics["test_loss"],
            "Test Accuracy": metrics["test_accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1 Score": metrics["f1_score"]
            }
        )
    results_df = pd.DataFrame(
    results
    )
    print("\n" + "=" * 50)
    print("Model Comparison")
    print("=" * 50)

    print(
        results_df.to_string(
            index=False
        )
    )
    csv_path = os.path.join(
    OUTPUT_DIR,
    "model_comparison.csv"
    )

    results_df.to_csv(
        csv_path,
        index=False
    )

    print(
        f"\nComparison table saved to: "
        f"{csv_path}"
    )
    plot_df = results_df.set_index(
    "Model"
    )

    plot_df[
        [
        "Test Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
        ]
    ].plot(
    kind="bar",
    figsize=(10, 6)
    )

    plt.title(
        "Model Performance Comparison"
    )

    plt.ylabel(
        "Score"
    )

    plt.xlabel(
        "Model"
    )

    plt.ylim(
        0,
        1
    )

    plt.xticks(
        rotation=0
    )

    plt.legend(
        loc="lower right"
    )

    plt.tight_layout()
    graph_path = os.path.join(
    OUTPUT_DIR,
    "model_comparison.png"
    )

    plt.savefig(
        graph_path,
        dpi=300
    )

    plt.close()

    print(
        f"Comparison graph saved to: "
        f"{graph_path}"
    )
    best_model_index = results_df[
    "F1 Score"
    ].idxmax()

    best_model = results_df.loc[
        best_model_index,
        "Model"
    ]

    best_f1_score = results_df.loc[
        best_model_index,
        "F1 Score"
    ]

    print("\n" + "=" * 50)
    print("Best Performing Model")
    print("=" * 50)

    print(
        f"Model    : {best_model}"
    )

    print(
        f"F1 Score : "
        f"{best_f1_score * 100:.2f}%"
    )
    
if __name__ == "__main__":
    main()        