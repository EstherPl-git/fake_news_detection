"""
main.py

FastAPI application for TruthLens AI.

Author: Esther
"""

from fastapi import FastAPI

from api.routes import router


app = FastAPI(
    title="TruthLens AI",
    description=(
        "Fake News Detection API powered by "
        "fine-tuned DistilBERT."
    ),
    version="1.0.0",
)


app.include_router(router)


@app.get(
    "/",
    tags=["Health"],
)
def root() -> dict[str, str]:
    """
    Root endpoint.
    """

    return {
        "message": "TruthLens AI API is running."
    }