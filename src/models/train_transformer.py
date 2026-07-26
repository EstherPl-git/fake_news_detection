"""
train_transformer.py

Fine-tune DistilBERT for Fake News Detection.

Author: Esther
Project: TruthLens AI
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.config.paths import (
    MODELS_DIR,
    REPORTS_DIR,
    TRAINING_DATASET,
)

from src.data.loader import load_dataset


# ==========================================================
# Configuration
# ==========================================================

MODEL_NAME = "distilbert-base-uncased"

MODEL_SAVE_PATH = MODELS_DIR / "bert_fake_news"

EVALUATION_DIR = REPORTS_DIR / "evaluation"

MAX_LENGTH = 256

BATCH_SIZE = 16

LEARNING_RATE = 2e-5

NUM_EPOCHS = 2

RANDOM_STATE = 42


# ==========================================================
# Dataset
# ==========================================================


def load_training_dataset() -> pd.DataFrame:
    """
    Load processed dataset.
    """

    dataframe = load_dataset(TRAINING_DATASET)

    dataframe = dataframe.dropna(
        subset=["text", "label"]
    ).copy()

    dataframe["text"] = dataframe["text"].astype(str)

    dataframe["label"] = dataframe["label"].astype(int)

    return dataframe


# ==========================================================
# Split Dataset
# ==========================================================


def create_data_split(
    dataframe: pd.DataFrame,
) -> tuple[Dataset, Dataset]:

    train_texts, valid_texts, train_labels, valid_labels = train_test_split(
        dataframe["text"],
        dataframe["label"],
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=dataframe["label"],
    )

    train_dataset = Dataset.from_dict(
        {
            "text": train_texts.tolist(),
            "label": train_labels.tolist(),
        }
    )

    validation_dataset = Dataset.from_dict(
        {
            "text": valid_texts.tolist(),
            "label": valid_labels.tolist(),
        }
    )

    return train_dataset, validation_dataset


# ==========================================================
# Metrics
# ==========================================================


def compute_metrics(eval_prediction):

    logits, labels = eval_prediction

    predictions = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
    )

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ==========================================================
# Save Metrics
# ==========================================================


def save_metrics(metrics: dict) -> None:

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = EVALUATION_DIR / "bert_metrics.json"

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    print()

    print(f"Metrics saved to: {output_file}")


# ==========================================================
# Build Trainer
# ==========================================================


def build_trainer() -> tuple[Trainer, AutoTokenizer]:

    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    dataframe = load_training_dataset()

    print(f"Dataset Size : {len(dataframe):,}")

    train_dataset, validation_dataset = create_data_split(
        dataframe
    )

    print()

    if torch.cuda.is_available():
        print(f"GPU : {torch.cuda.get_device_name(0)}")
    else:
        print("GPU : CPU")

    print()

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    def tokenize(batch):

        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

    tokenized_train = train_dataset.map(
    tokenize,
    batched=True,
    remove_columns=["text"],
    )

    tokenized_validation = validation_dataset.map(
    tokenize,
    batched=True,
    remove_columns=["text"],
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
    )

    print()

    print("Loading model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    training_arguments = TrainingArguments(

        output_dir=str(MODEL_SAVE_PATH),

        learning_rate=LEARNING_RATE,

        per_device_train_batch_size=BATCH_SIZE,

        per_device_eval_batch_size=BATCH_SIZE,

        num_train_epochs=NUM_EPOCHS,

        eval_strategy="epoch",

        save_strategy="epoch",

        logging_strategy="steps",

        logging_steps=100,

        load_best_model_at_end=True,

        metric_for_best_model="f1",

        greater_is_better=True,

        save_total_limit=2,

        report_to="none",

        seed=RANDOM_STATE,

        fp16=torch.cuda.is_available(),

        remove_unused_columns=False,
    )

    trainer = Trainer(

        model=model,

        args=training_arguments,

        train_dataset=tokenized_train,

        eval_dataset=tokenized_validation,

        processing_class=tokenizer,

        data_collator=data_collator,

        compute_metrics=compute_metrics,
    )

    return trainer, tokenizer

# ==========================================================
# Train Model
# ==========================================================


def train_model() -> Trainer:
    """
    Fine-tune DistilBERT.
    """

    trainer, tokenizer = build_trainer()

    print()
    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print()

    trainer.train()

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    MODEL_SAVE_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("Saving model...")

    trainer.save_model(MODEL_SAVE_PATH)

    tokenizer.save_pretrained(MODEL_SAVE_PATH)

    print(f"Model saved to: {MODEL_SAVE_PATH}")

    print()
    print("Running final evaluation...")

    metrics = trainer.evaluate()

    print()

    print("=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    for key, value in metrics.items():

        if isinstance(value, float):

            print(f"{key:<20}: {value:.4f}")

        else:

            print(f"{key:<20}: {value}")

    save_metrics(metrics)

    print()

    print("=" * 60)
    print("TRAINING PIPELINE FINISHED")
    print("=" * 60)

    return trainer


# ==========================================================
# Main
# ==========================================================


def main():

    train_model()


if __name__ == "__main__":

    main()