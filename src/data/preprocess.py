"""
preprocess.py

Data preprocessing module for the Fake News Detection project.

This module cleans the raw dataset and saves a processed version
for model training.
"""

from pathlib import Path

import pandas as pd

from src.config.paths import (
    PROCESSED_DATA_DIR,
    WELFAKE_DATASET,
)
from src.data.loader import load_dataset


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows.
    """
    return df.drop_duplicates().reset_index(drop=True)


def remove_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows that have missing text or label values.
    """

    required_columns = ["text", "label"]

    if "title" in df.columns:
        required_columns.append("title")

    df = df.dropna(subset=required_columns)

    return df.reset_index(drop=True)


def clean_text(text: str) -> str:
    """
    Basic text cleaning.
    """

    text = str(text)

    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")

    text = " ".join(text.split())

    return text


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply text cleaning to title and text columns.
    """

    if "title" in df.columns:
        df["title"] = df["title"].astype(str).apply(clean_text)

    if "text" in df.columns:
        df["text"] = df["text"].astype(str).apply(clean_text)

    # Remove rows where cleaned text is empty
    df = df[df["text"].str.strip() != ""]

    # Remove rows where cleaned title is empty
    df = df[df["title"].str.strip() != ""]

    return df.reset_index(drop=True)


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    print("Removing missing values...")
    df = remove_missing_values(df)

    print("Removing duplicate rows...")
    df = remove_duplicates(df)

    print("Cleaning text...")
    df = clean_dataset(df)

    return df


def save_processed_dataset(
    df: pd.DataFrame,
    filename: str = "processed_dataset.csv",
) -> Path:
    """
    Save processed dataset.
    """

    output_path = PROCESSED_DATA_DIR / filename

    df.to_csv(output_path, index=False)

    return output_path


def main() -> None:
    """
    Run preprocessing pipeline.
    """

    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    df = load_dataset(WELFAKE_DATASET)

    print(f"Original Shape : {df.shape}")

    df = preprocess_dataset(df)

    print(f"Processed Shape : {df.shape}")

    output_path = save_processed_dataset(df)

    print("\nProcessed dataset saved successfully.")

    print(output_path)


if __name__ == "__main__":
    main()