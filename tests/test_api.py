"""
test_api.py

Tests for the TruthLens AI FastAPI application.

Author: Esther
"""

from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


# ==========================================================
# Health Endpoint
# ==========================================================


def test_health_endpoint():
    """
    Health endpoint should return HTTP 200.
    """

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "healthy"
    }


# ==========================================================
# Prediction Endpoint
# ==========================================================


def test_prediction_endpoint():
    """
    Prediction endpoint should successfully classify text.
    """

    response = client.post(
        "/predict",
        json={
            "text": (
                "Scientists announced a new discovery "
                "after years of research."
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in {
        "Fake",
        "Real",
    }

    assert 0.0 <= data["confidence"] <= 1.0

    assert "fake" in data["probabilities"]
    assert "real" in data["probabilities"]


# ==========================================================
# Invalid Input
# ==========================================================


def test_prediction_rejects_empty_text():
    """
    Empty text should be rejected by Pydantic validation.
    """

    response = client.post(
        "/predict",
        json={
            "text": ""
        },
    )

    assert response.status_code == 422


# ==========================================================
# Missing Input
# ==========================================================


def test_prediction_rejects_missing_text():
    """
    Missing text field should be rejected.
    """

    response = client.post(
        "/predict",
        json={},
    )

    assert response.status_code == 422