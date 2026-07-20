"""
audit.py

Dataset auditing module.

Performs quality analysis before preprocessing
or model training.

Author: Esther
Project: TruthLens AI
"""

from pathlib import Path

import pandas as pd

from src.config.paths import WELFAKE_DATASET
from src.data.loader import load_dataset


def print_section(title: str) -> None:
    """
    Print a formatted section heading.
    """

    print("\n" + "=" * 70)
    print(title.upper())
    print("=" * 70)


def basic_information(df: pd.DataFrame) -> None:
    """
    Display basic dataset information.
    """

    print_section("Basic Information")

    rows, columns = df.shape

    print(f"Rows           : {rows:,}")
    print(f"Columns        : {columns}")

    memory = df.memory_usage(deep=True).sum() / (1024 ** 2)

    print(f"Memory Usage   : {memory:.2f} MB")

    print("\nColumn Names")

    for column in df.columns:
        print(f"• {column}")

    print("\nData Types")

    print(df.dtypes)

def missing_value_analysis(df: pd.DataFrame) -> None:
    """
    Analyze missing values.
    """

    print_section("Missing Value Analysis")

    missing = df.isnull().sum()

    percentages = (missing / len(df)) * 100

    report = pd.DataFrame({
        "Missing": missing,
        "Percentage": percentages.round(2)
    })

    print(report)

def duplicate_analysis(df: pd.DataFrame) -> None:
    """
    Analyze duplicate data.
    """

    print_section("Duplicate Analysis")

    duplicate_rows = df.duplicated().sum()

    print(f"Duplicate Rows : {duplicate_rows}")

    if "title" in df.columns:

        duplicate_titles = df["title"].duplicated().sum()

        print(f"Duplicate Titles : {duplicate_titles}")

    if "text" in df.columns:

        duplicate_articles = df["text"].duplicated().sum()

        print(f"Duplicate Articles : {duplicate_articles}")

def class_distribution(df: pd.DataFrame) -> None:
    """
    Display class balance.
    """

    if "label" not in df.columns:
        return

    print_section("Class Distribution")

    counts = df["label"].value_counts()

    percentages = (
        df["label"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    report = pd.DataFrame({
        "Count": counts,
        "Percentage": percentages
    })

    print(report)

def text_statistics(df: pd.DataFrame) -> None:
    """
    Analyze article lengths.
    """

    if "text" not in df.columns:
        return

    print_section("Text Statistics")

    lengths = df["text"].astype(str).str.split().str.len()

    print(f"Average Words : {lengths.mean():.2f}")
    print(f"Median Words  : {lengths.median():.2f}")
    print(f"Minimum Words : {lengths.min()}")
    print(f"Maximum Words : {lengths.max()}")

def dataset_summary(df: pd.DataFrame) -> None:
    """
    Final dataset recommendation.
    """

    print_section("Summary")

    missing = df.isnull().sum().sum()

    duplicates = df.duplicated().sum()

    if missing == 0:
        print("✓ Missing Values : Excellent")
    else:
        print(f"⚠ Missing Values : {missing}")

    if duplicates == 0:
        print("✓ Duplicate Rows : Excellent")
    else:
        print(f"⚠ Duplicate Rows : {duplicates}")

    print("\nRecommendation")

    print("Dataset is suitable for preprocessing.")


def audit_dataset(dataset_path: Path) -> None:
    """
    Run the complete audit.
    """

    df = load_dataset(dataset_path)

    basic_information(df)

    missing_value_analysis(df)

    duplicate_analysis(df)

    class_distribution(df)

    text_statistics(df)

    dataset_summary(df)


if __name__ == "__main__":

    audit_dataset(WELFAKE_DATASET)