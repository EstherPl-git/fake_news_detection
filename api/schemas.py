"""
schemas.py

Pydantic schemas for the TruthLens AI API.

Author: Esther
"""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """
    Request body for fake news prediction.
    """

    text: str = Field(
        ...,
        min_length=1,
        description="News article text to classify.",
    )


class PredictionResponse(BaseModel):
    """
    Response returned by the prediction endpoint.
    """

    prediction: str = Field(
        ...,
        description="Predicted class: Fake or Real.",
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence of the prediction.",
    )

    probabilities: dict[str, float] = Field(
        ...,
        description="Probability for each class.",
    )