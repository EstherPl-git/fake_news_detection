"""
train_baseline.py

Train a TF-IDF + Logistic Regression baseline model.

Author: Esther
Project: TruthLens AI
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split

from src.config.paths import (
    MODELS_DIR,
    TRAINING_DATASET,
)
from src.data.loader import load_dataset


def prepare_features(df: pd.DataFrame):
    """
    Prepare text features and labels.
    """

    if "text" not in df.columns:
        raise ValueError("Column 'text' not found.")

    if "label" not in df.columns:
        raise ValueError("Column 'label' not found.")

    # Keep only rows with valid text and labels
    df = df.dropna(subset=["text", "label"]).copy()

    # Ensure text is string
    df["text"] = df["text"].astype(str)

    x = df["text"]
    y = df["label"]

    return x, y


def train_model():

    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    df = load_dataset(TRAINING_DATASET)

    x, y = prepare_features(df)

    print("\nCreating train-test split...")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Vectorizing text...")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000,
    )

    x_train = vectorizer.fit_transform(x_train)

    x_test = vectorizer.transform(x_test)

    print("Training Logistic Regression...")

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print("\nAccuracy")

    print(f"{accuracy:.4f}")

    print("\nClassification Report\n")

    print(
        classification_report(
            y_test,
            predictions,
        )
    )

    model_path = MODELS_DIR / "baseline_model.pkl"

    vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"

    joblib.dump(
        model,
        model_path,
    )

    joblib.dump(
        vectorizer,
        vectorizer_path,
    )

    print("\nModel saved successfully.")

    print(model_path)

    print(vectorizer_path)


if __name__ == "__main__":
    train_model()