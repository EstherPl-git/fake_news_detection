"""
paths.py

Centralized project paths.

This module stores every important directory and file path used
throughout the Fake News Detection project.

Author: Esther
"""

from pathlib import Path


# ==========================================================
# Project Root
# ==========================================================

# fake-news-detection/
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ==========================================================
# Data Directories
# ==========================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

REPORTS_DIR = DATA_DIR / "reports"


# ==========================================================
# Raw Dataset Paths
# ==========================================================

ISOT_DIR = RAW_DATA_DIR / "isot"

WELFAKE_DIR = RAW_DATA_DIR / "welfake"

ISOT_FAKE_DATASET = ISOT_DIR / "Fake.csv"

ISOT_TRUE_DATASET = ISOT_DIR / "True.csv"

WELFAKE_DATASET = WELFAKE_DIR / "WELFake_Dataset.csv"

TRAINING_DATASET = PROCESSED_DATA_DIR / "processed_dataset.csv"


# ==========================================================
# Model Directory
# ==========================================================

MODELS_DIR = PROJECT_ROOT / "models"


# ==========================================================
# API Directory
# ==========================================================

API_DIR = PROJECT_ROOT / "api"


# ==========================================================
# Tests Directory
# ==========================================================

TESTS_DIR = PROJECT_ROOT / "tests"


# ==========================================================
# Ensure Required Directories Exist
# ==========================================================

DIRECTORIES = [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    MODELS_DIR,
    API_DIR,
    TESTS_DIR,
]

for directory in DIRECTORIES:
    directory.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Helper Functions
# ==========================================================

def print_project_paths() -> None:
    """
    Print all important project paths.
    Useful for debugging.
    """

    print("=" * 60)
    print("PROJECT PATHS")
    print("=" * 60)

    print(f"Project Root      : {PROJECT_ROOT}")
    print(f"Raw Data          : {RAW_DATA_DIR}")
    print(f"Processed Data    : {PROCESSED_DATA_DIR}")
    print(f"Reports           : {REPORTS_DIR}")

    print(f"ISOT Fake         : {ISOT_FAKE_DATASET}")
    print(f"ISOT True         : {ISOT_TRUE_DATASET}")

    print(f"WELFake           : {WELFAKE_DATASET}")
    print(f"Training Dataset  : {TRAINING_DATASET}")

    print(f"Models            : {MODELS_DIR}")
    print(f"API               : {API_DIR}")
    print(f"Tests             : {TESTS_DIR}")


# ==========================================================
# Run as Script
# ==========================================================

if __name__ == "__main__":
    print_project_paths()