"""
predict.py

Inference pipeline for the Fake News Detection project.

Loads the fine-tuned DistilBERT model and tokenizer,
then predicts whether a news article is fake or real.

Author: Esther
Project: TruthLens AI
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from src.config.paths import MODELS_DIR


# ==========================================================
# Configuration
# ==========================================================

MODEL_PATH = MODELS_DIR / "bert_fake_news"

MAX_LENGTH = 256


# ==========================================================
# Label Mapping
# ==========================================================

LABEL_MAPPING = {
    0: "Fake",
    1: "Real",
}


# ==========================================================
# Device
# ==========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================================
# Model Predictor
# ==========================================================


class FakeNewsPredictor:
    """
    Loads the trained DistilBERT model and performs predictions.
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
    ) -> None:

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model directory not found: {self.model_path}"
            )

        print("Loading tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
        )

        print("Loading model...")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            local_files_only=True,
        )

        self.model.to(DEVICE)

        self.model.eval()

        print(f"Model loaded successfully.")
        print(f"Device: {DEVICE}")


    # ======================================================
    # Prediction
    # ======================================================

    def predict(
        self,
        text: str,
    ) -> dict:
        """
        Predict whether the supplied news text is fake or real.
        """

        if not isinstance(text, str):
            raise TypeError(
                "Input text must be a string."
            )

        text = text.strip()

        if not text:
            raise ValueError(
                "Input text cannot be empty."
            )

        # --------------------------------------------------
        # Tokenization
        # --------------------------------------------------

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        # --------------------------------------------------
        # Model Prediction
        # --------------------------------------------------

        with torch.no_grad():

            outputs = self.model(
                **inputs
            )

        # --------------------------------------------------
        # Convert logits to probabilities
        # --------------------------------------------------

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        )

        probabilities = probabilities[
            0
        ].cpu().numpy()

        # --------------------------------------------------
        # Select predicted class
        # --------------------------------------------------

        predicted_class = int(
            probabilities.argmax()
        )

        predicted_label = LABEL_MAPPING.get(
            predicted_class,
            str(predicted_class),
        )

        confidence = float(
            probabilities[predicted_class]
        )

        return {
            "prediction": predicted_label,
            "confidence": confidence,
            "probabilities": {
                "fake": float(probabilities[0]),
                "real": float(probabilities[1]),
            },
        }


# ==========================================================
# Create Predictor
# ==========================================================


predictor = FakeNewsPredictor()


# ==========================================================
# Convenience Function
# ==========================================================


def predict_news(text: str) -> dict:
    """
    Predict fake or real news using the trained model.
    """

    return predictor.predict(text)


# ==========================================================
# Manual Test
# ==========================================================


if __name__ == "__main__":

    sample_text = (
        "The government announced a new policy today "
        "after a meeting with senior officials."
    )

    result = predict_news(sample_text)

    print()
    print("=" * 60)
    print("PREDICTION")
    print("=" * 60)

    print(f"Prediction : {result['prediction']}")
    print(
        f"Confidence : {result['confidence']:.4f}"
    )

    print()
    print("Probabilities:")

    print(
        f"Fake : {result['probabilities']['fake']:.4f}"
    )

    print(
        f"Real : {result['probabilities']['real']:.4f}"
    )