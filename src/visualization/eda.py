"""
eda.py

Exploratory Data Analysis (EDA) for Fake News Detection datasets.

This module generates basic visualizations to understand the dataset
before preprocessing and model training.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config.paths import REPORTS_DIR, WELFAKE_DATASET
from src.data.loader import load_dataset


EDA_OUTPUT_DIR = REPORTS_DIR / "figures"
EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_label_distribution(df: pd.DataFrame) -> None:
    """
    Plot and save the label distribution.
    """

    if "label" not in df.columns:
        print("Label column not found.")
        return

    counts = df["label"].value_counts().sort_index()

    plt.figure(figsize=(6, 5))
    counts.plot(kind="bar")

    plt.title("Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")

    plt.tight_layout()

    output_file = EDA_OUTPUT_DIR / "label_distribution.png"
    plt.savefig(output_file)

    plt.close()

    print(f"Saved: {output_file}")


def plot_missing_values(df: pd.DataFrame) -> None:
    """
    Plot missing values for each column.
    """

    missing = df.isnull().sum()

    plt.figure(figsize=(8, 5))
    missing.plot(kind="bar")

    plt.title("Missing Values")
    plt.xlabel("Columns")
    plt.ylabel("Missing Count")

    plt.tight_layout()

    output_file = EDA_OUTPUT_DIR / "missing_values.png"
    plt.savefig(output_file)

    plt.close()

    print(f"Saved: {output_file}")


def plot_article_length(df: pd.DataFrame) -> None:
    """
    Plot distribution of article lengths.
    """

    if "text" not in df.columns:
        print("Text column not found.")
        return

    lengths = df["text"].fillna("").astype(str).str.split().str.len()

    plt.figure(figsize=(8, 5))
    plt.hist(lengths, bins=50)

    plt.title("Article Length Distribution")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")

    plt.tight_layout()

    output_file = EDA_OUTPUT_DIR / "article_length_distribution.png"
    plt.savefig(output_file)

    plt.close()

    print(f"Saved: {output_file}")


def generate_eda(df: pd.DataFrame) -> None:
    """
    Generate all EDA visualizations.
    """

    print("\nGenerating EDA Visualizations...\n")

    plot_label_distribution(df)

    plot_missing_values(df)

    plot_article_length(df)

    print("\nEDA completed successfully.")


if __name__ == "__main__":

    dataframe = load_dataset(WELFAKE_DATASET)

    generate_eda(dataframe)