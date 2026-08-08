"""
routes.py

API routes for the TruthLens AI fake news classifier.

Author: Esther
"""

from fastapi import APIRouter, HTTPException

from api.schemas import (
    PredictionRequest,
    PredictionResponse,
)

from src.models.predict import predict_news


router = APIRouter()


@router.get(
    "/health",
    tags=["Health"],
)
def health_check() -> dict[str, str]:
    """
    Check whether the API is running.
    """

    return {
        "status": "healthy",
    }


@router.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
)
def predict(
    request: PredictionRequest,
) -> PredictionResponse:
    """
    Predict whether a news article is fake or real.
    """

    try:

        result = predict_news(
            request.text
        )

        return PredictionResponse(
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except TypeError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )