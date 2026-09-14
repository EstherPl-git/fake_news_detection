"""
predict.py

Inference pipeline for the Fake News Detection project.

Loads the quantized ONNX DistilBERT model and tokenizer,
then predicts whether a news article is fake or real.

Author: Esther
Project: TruthLens AI
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from src.config.paths import MODELS_DIR


# ==========================================================
# Configuration
# ==========================================================

MODEL_ID = os.getenv(
    "MODEL_ID",
    "estherp/truthlens-distilbert-onnx",
)

LOCAL_MODEL_PATH = MODELS_DIR / "truthlens-onnx"

MAX_LENGTH = 256


# ==========================================================
# Label Mapping
# ==========================================================

LABEL_MAPPING = {
    0: "Real",
    1: "Fake",
}


# ==========================================================
# Model Predictor
# ==========================================================


class FakeNewsPredictor:
    """
    Loads the quantized ONNX model and performs predictions.
    """

    def __init__(
        self,
        model_path: Path | None = None,
    ) -> None:

        # --------------------------------------------------
        # Determine model source
        # --------------------------------------------------

        if model_path is not None:

            self.model_source = Path(model_path)

            if not self.model_source.exists():
                raise FileNotFoundError(
                    f"Model directory not found: {self.model_source}"
                )

            print(
                f"Loading model from local path: "
                f"{self.model_source}"
            )

        else:

            self.model_source = MODEL_ID

            print(
                f"Loading model from Hugging Face: "
                f"{self.model_source}"
            )

        # --------------------------------------------------
        # Load tokenizer
        # --------------------------------------------------

        print("Loading tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_source,
            local_files_only=model_path is not None,
        )

        # --------------------------------------------------
        # Load ONNX model
        # --------------------------------------------------

        print("Loading ONNX model...")

        if model_path is not None:
            model_file = (
                self.model_source / "model.int8.onnx"
            )
        else:
            from huggingface_hub import hf_hub_download

            model_file = hf_hub_download(
                repo_id=self.model_source,
                filename="model.int8.onnx",
            )

        self.session = ort.InferenceSession(
            str(model_file),
            providers=["CPUExecutionProvider"],
        )

        # --------------------------------------------------
        # Get supported ONNX input names
        # --------------------------------------------------

        self.input_names = {
            input_info.name
            for input_info in self.session.get_inputs()
        }

        print("ONNX model loaded successfully.")
        print("Execution provider: CPUExecutionProvider")

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
            return_tensors="np",
            truncation=True,
            max_length=MAX_LENGTH,
        )

        # --------------------------------------------------
        # Prepare ONNX inputs
        # --------------------------------------------------

        ort_inputs = {
            key: value
            for key, value in inputs.items()
            if key in self.input_names
        }

        # --------------------------------------------------
        # Model Prediction
        # --------------------------------------------------

        outputs = self.session.run(
            None,
            ort_inputs,
        )

        logits = outputs[0][0]

        # --------------------------------------------------
        # Convert logits to probabilities
        # --------------------------------------------------

        logits = logits - np.max(logits)

        exp_logits = np.exp(logits)

        probabilities = (
            exp_logits / exp_logits.sum()
        )

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
                "fake": float(probabilities[1]),
                "real": float(probabilities[0]),
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

    print(
        f"Prediction : {result['prediction']}"
    )

    print(
        f"Confidence : "
        f"{result['confidence']:.4f}"
    )

    print()
    print("Probabilities:")

    print(
        f"Fake : "
        f"{result['probabilities']['fake']:.4f}"
    )

    print(
        f"Real : "
        f"{result['probabilities']['real']:.4f}"
    )