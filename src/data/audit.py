"""
Dataset Audit Tool

Purpose:
    Analyze datasets before preprocessing or training.
    This helps identify issues such as missing values,
    incorrect columns, duplicates, and class imbalance.

Author: Esther
Project: Fake News Detection
"""

from pathlib import Path

import pandas as pd


def load_dataset(file_path: Path) -> pd.DataFrame:
    """
    Load a CSV dataset.

    Parameters
    ----------
    file_path : Path
        Location of the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset does not exist.
    """

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    return pd.read_csv(file_path)


def show_basic_info(dataset: pd.DataFrame, dataset_name: str) -> None:
    """
    Display basic dataset information.
    """

    print("=" * 60)
    print(f"DATASET : {dataset_name}")
    print("=" * 60)

    print(f"Rows          : {dataset.shape[0]}")
    print(f"Columns       : {dataset.shape[1]}")
    print(f"Memory Usage  : {dataset.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print("\nColumns:")
    print(dataset.columns.tolist())

    print("\nData Types:")
    print(dataset.dtypes)


def main() -> None:
    """
    Main entry point.
    """

    isot_fake = Path("data/raw/isot/Fake.csv")

    dataset = load_dataset(isot_fake)

    show_basic_info(dataset, "ISOT Fake")


if __name__ == "__main__":
    main()