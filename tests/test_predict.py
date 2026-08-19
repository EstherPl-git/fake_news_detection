"""
test_predict.py

Tests for the TruthLens AI prediction pipeline.

Author: Esther
"""

from src.models.predict import predict_news


def test_prediction_returns_valid_label():
    """
    Prediction should return either Fake or Real.
    """

    result = predict_news(
        "Scientists announced a new discovery after years of research."
    )

    assert result["prediction"] in {"Fake", "Real"}


def test_prediction_returns_valid_confidence():
    """
    Confidence should be between 0 and 1.
    """

    result = predict_news(
        "The government announced a new policy today."
    )

    assert 0.0 <= result["confidence"] <= 1.0


def test_prediction_returns_probabilities():
    """
    Prediction should contain probabilities for both classes.
    """

    result = predict_news(
        "Researchers published the results of a new scientific study."
    )

    probabilities = result["probabilities"]

    assert "fake" in probabilities
    assert "real" in probabilities

    assert 0.0 <= probabilities["fake"] <= 1.0
    assert 0.0 <= probabilities["real"] <= 1.0


def test_probabilities_sum_to_one():
    """
    Fake and real probabilities should sum approximately to 1.
    """

    result = predict_news(
        "A new technology was introduced by researchers."
    )

    probabilities = result["probabilities"]

    total = (
        probabilities["fake"]
        + probabilities["real"]
    )

    assert abs(total - 1.0) < 1e-5