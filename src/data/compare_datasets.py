"""
compare_datasets.py

Compare the ISOT and WELFake datasets and recommend
the best dataset for model training.

Author: Esther
Project: TruthLens AI
"""

from pathlib import Path

import pandas as pd

from src.config.paths import (
    ISOT_FAKE_DATASET,
    ISOT_TRUE_DATASET,
    WELFAKE_DATASET,
)
from src.data.loader import load_dataset


def prepare_isot_dataset() -> pd.DataFrame:
    """
    Load and combine the ISOT fake and true datasets.
    """

    fake_df = load_dataset(ISOT_FAKE_DATASET)
    true_df = load_dataset(ISOT_TRUE_DATASET)

    fake_df["label"] = 1
    true_df["label"] = 0

    isot_df = pd.concat(
        [fake_df, true_df],
        ignore_index=True
    )

    return isot_df


def dataset_statistics(
    dataset_name: str,
    dataframe: pd.DataFrame,
) -> dict:
    """
    Generate statistics for a dataset.
    """

    text_column = "text" if "text" in dataframe.columns else None

    duplicate_articles = (
        dataframe[text_column].duplicated().sum()
        if text_column
        else 0
    )

    average_words = (
        dataframe[text_column]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
        .mean()
        if text_column
        else 0
    )

    statistics = {
        "Dataset": dataset_name,
        "Rows": len(dataframe),
        "Columns": dataframe.shape[1],
        "Missing Values": dataframe.isnull().sum().sum(),
        "Duplicate Rows": dataframe.duplicated().sum(),
        "Duplicate Articles": duplicate_articles,
        "Memory (MB)": round(
            dataframe.memory_usage(deep=True).sum() / (1024 ** 2),
            2,
        ),
        "Average Words": round(average_words, 2),
    }

    if "label" in dataframe.columns:
        statistics["Class Distribution"] = (
            dataframe["label"]
            .value_counts(normalize=True)
            .round(3)
            .to_dict()
        )

    return statistics


def print_report(report: dict) -> None:
    """
    Print dataset statistics.
    """

    print("\n" + "=" * 70)
    print(report["Dataset"])
    print("=" * 70)

    for key, value in report.items():

        if key == "Dataset":
            continue

        print(f"{key:<22}: {value}")


def main() -> None:

    print("\nLoading datasets...\n")

    isot_dataset = prepare_isot_dataset()

    welfake_dataset = load_dataset(WELFAKE_DATASET)

    isot_report = dataset_statistics(
        "ISOT",
        isot_dataset,
    )

    welfake_report = dataset_statistics(
        "WELFake",
        welfake_dataset,
    )

    print_report(isot_report)

    print_report(welfake_report)

    print("\n" + "=" * 70)
    print("FINAL RECOMMENDATION")
    print("=" * 70)

    if len(welfake_dataset) >= len(isot_dataset):

        print(
            "Selected Dataset : WELFake"
        )

        print(
            "Reason : Larger dataset with greater diversity."
        )

    else:

        print(
            "Selected Dataset : ISOT"
        )

        print(
            "Reason : Better suited based on comparison."
        )


if __name__ == "__main__":
    main()