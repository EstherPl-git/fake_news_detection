"""
loader.py

Utility functions for safely loading datasets.

This module centralizes all CSV loading logic so the rest of the
project never calls pandas.read_csv() directly.

Author: Esther
Project: TruthLens AI
"""

from pathlib import Path
from typing import Optional

import pandas as pd


def validate_dataset_path(file_path: Path) -> None:
    """
    Validate that a dataset path exists and is a CSV file.

    Parameters
    ----------
    file_path : Path
        Path to the dataset.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.

    ValueError
        If the file is not a CSV.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{file_path}"
        )

    if file_path.suffix.lower() != ".csv":
        raise ValueError(
            f"Expected a CSV file.\nReceived: {file_path.name}"
        )


def load_dataset(
    file_path: Path,
    encoding: str = "utf-8",
    low_memory: bool = False,
) -> pd.DataFrame:
    """
    Load a CSV dataset.

    Parameters
    ----------
    file_path : Path
        Dataset location.

    encoding : str
        File encoding.

    low_memory : bool
        Pandas memory optimization flag.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.

    Raises
    ------
    FileNotFoundError
        Dataset missing.

    ValueError
        Invalid file.

    RuntimeError
        Any pandas loading error.
    """

    validate_dataset_path(file_path)

    try:
        dataframe = pd.read_csv(
            file_path,
            encoding=encoding,
            low_memory=low_memory,
        )

    except UnicodeDecodeError as exc:
        raise RuntimeError(
            "Encoding error while reading dataset."
        ) from exc

    except pd.errors.EmptyDataError as exc:
        raise RuntimeError(
            "Dataset is empty."
        ) from exc

    except pd.errors.ParserError as exc:
        raise RuntimeError(
            "CSV parsing failed."
        ) from exc

    except Exception as exc:
        raise RuntimeError(
            f"Unexpected error while loading dataset:\n{exc}"
        ) from exc

    return dataframe


def dataset_shape(dataframe: pd.DataFrame) -> tuple[int, int]:
    """
    Return dataset dimensions.

    Returns
    -------
    tuple
        (rows, columns)
    """

    return dataframe.shape


def preview_dataset(
    dataframe: pd.DataFrame,
    rows: int = 5,
) -> pd.DataFrame:
    """
    Return the first few rows.

    Parameters
    ----------
    dataframe : DataFrame

    rows : int

    Returns
    -------
    DataFrame
    """

    return dataframe.head(rows)


def memory_usage_mb(
    dataframe: pd.DataFrame,
) -> float:
    """
    Return memory usage in MB.
    """

    memory = dataframe.memory_usage(
        deep=True
    ).sum()

    return round(memory / (1024 ** 2), 2)


def column_names(
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Return all column names.
    """

    return dataframe.columns.tolist()


def dataset_info(
    dataframe: pd.DataFrame,
) -> dict:
    """
    Return a quick summary of the dataset.

    Returns
    -------
    dict
    """

    rows, columns = dataset_shape(dataframe)

    return {
        "rows": rows,
        "columns": columns,
        "memory_mb": memory_usage_mb(dataframe),
        "column_names": column_names(dataframe),
    }


if __name__ == "__main__":

    from src.config.paths import WELFAKE_DATASET

    df = load_dataset(WELFAKE_DATASET)

    print("=" * 60)
    print("DATASET LOADED SUCCESSFULLY")
    print("=" * 60)

    print(dataset_info(df))

    print("\nPreview\n")

    print(preview_dataset(df))