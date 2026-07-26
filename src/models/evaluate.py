"""
evaluate.py

Evaluate the trained baseline model and save evaluation results.

Author: Esther
Project: TruthLens AI
"""

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.config.paths import (
    MODELS_DIR,
    REPORTS_DIR,
    TRAINING_DATASET,
)
from src.data.loader import load_dataset


EVALUATION_DIR = REPORTS_DIR / "evaluation"
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)


def load_model():
    """
    Load trained model and TF-IDF vectorizer.
    """

    model = joblib.load(
        MODELS_DIR / "baseline_model.pkl"
    )

    vectorizer = joblib.load(
        MODELS_DIR / "tfidf_vectorizer.pkl"
    )

    return model, vectorizer


def prepare_test_data():
    """
    Prepare the test split.
    """

    df = load_dataset(TRAINING_DATASET)

    df = df.dropna(subset=["text", "label"]).copy()

    x = df["text"].astype(str)

    y = df["label"]

    _, x_test, _, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    return x_test, y_test


def evaluate():
    """
    Evaluate the baseline model.
    """

    model, vectorizer = load_model()

    x_test, y_test = prepare_test_data()

    x_test = vectorizer.transform(x_test)

    predictions = model.predict(x_test)

    metrics = {
        "accuracy": round(
            accuracy_score(y_test, predictions),
            4,
        ),
        "precision": round(
            precision_score(y_test, predictions),
            4,
        ),
        "recall": round(
            recall_score(y_test, predictions),
            4,
        ),
        "f1_score": round(
            f1_score(y_test, predictions),
            4,
        ),
    }

    print("\nEvaluation Metrics\n")

    for key, value in metrics.items():
        print(f"{key:<12}: {value}")

    with open(
        EVALUATION_DIR / "metrics.json",
        "w",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    report = classification_report(
        y_test,
        predictions,
    )

    with open(
        EVALUATION_DIR / "classification_report.txt",
        "w",
    ) as file:

        file.write(report)

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
    )

    display.plot()

    plt.tight_layout()

    plt.savefig(
        EVALUATION_DIR / "confusion_matrix.png"
    )

    plt.close()

    print("\nEvaluation files saved.")

    print(EVALUATION_DIR)


if __name__ == "__main__":
    evaluate()