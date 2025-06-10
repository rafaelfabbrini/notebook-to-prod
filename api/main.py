"""
FastAPI application exposing the property-valuation model for online inference.

Key points
----------
* API-key header security (`X-API-Key`).
* Logging of request, prediction, and model version.

Endpoints
---------
/health
    Quick liveness probe.
/info
    Return the latest registered model name and version.
/predict
    Predict property price (requires `X-API-Key`).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.security.api_key import APIKeyHeader

from core.config import settings
from core.pipeline import ModelPipeline
from core.schemas import PipelineInput, PipelineOutput

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("api")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(
    api_key: Annotated[str | None, Depends(api_key_header)],
) -> str:
    """
    Validate the `X-API-Key` header.

    Args:
        api_key: Value provided by the client in the `X-API-Key` header. The value
            can be ``None`` when the header is absent.

    Returns:
        The validated API key (identical to *api_key*).

    Raises:
        HTTPException: If the header is missing or the key does not match
            :pydataattr:`core.config.settings.API_KEY`.
    """
    if api_key and api_key == settings.API_KEY:
        return api_key
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key.",
    )


@lru_cache(maxsize=1)
def get_model_pipeline() -> ModelPipeline:
    """
    Load a single :class:`core.pipeline.ModelPipeline` instance.

    The function is wrapped by :pyfunc:`functools.lru_cache` so the model is
    instantiated only once per interpreter process, drastically reducing cold
    start time for subsequent requests.

    Returns:
        A fully initialised :class:`core.pipeline.ModelPipeline`.
    """
    logger.info("Initialising ModelPipeline…")
    pipeline = ModelPipeline()
    logger.info("ModelPipeline initialised.")
    return pipeline


app = FastAPI(
    title="Property Valuation API",
    version="0.1.0",
    description="Inference service for the notebook-to-prod project.",
)


@app.get("/health", tags=["Utility"])
def health() -> dict[str, str]:
    """
    Return a simple liveness probe.

    Returns:
        A dictionary containing a fixed ``"status": "ok"`` flag and the current
        UTC timestamp.
    """
    return {"status": "ok", "timestamp": datetime.now(tz=timezone.utc).isoformat()}


@app.get("/info", tags=["Utility"])
def info(
    pipeline: Annotated[ModelPipeline, Depends(get_model_pipeline)],
) -> dict[str, str]:
    """
    Expose high-level model metadata.

    Args:
        pipeline: Dependency-injected :class:`core.pipeline.ModelPipeline`
            instance.

    Returns:
        A dictionary with the latest registered model name and version.
    """
    return {
        "model_name": pipeline.model_store._model_name,
        "model_version": pipeline.model_store._get_latest_model_version(),
    }


@app.post(
    "/predict",
    response_model=PipelineOutput,
    dependencies=[Depends(require_api_key)],
    tags=["Prediction"],
)
def predict(
    input_data: PipelineInput,
    pipeline: Annotated[ModelPipeline, Depends(get_model_pipeline)],
    request: Request,
) -> PipelineOutput:
    """
    Predict property price using the latest trained model.

    This endpoint is protected by an API-key header (`X-API-Key`). Successful
    calls are logged together with the request ID (if the caller provided the
    `X-Request-ID` header) and the model version.

    Args:
        input_data: Structured feature data adhering to
            :class:`core.schemas.PipelineInput`.
        pipeline: Dependency-injected :class:`core.pipeline.ModelPipeline`
            instance.
        request: Raw FastAPI :class:`fastapi.Request`, used only for logging the
            `X-Request-ID` header.

    Returns:
        A :class:`core.schemas.PipelineOutput` containing the predicted price.
    """
    prediction = pipeline.predict(input_data)
    target = PipelineOutput.get_target()

    logger.info(
        f"request_id=%s | model_version=%s | {target}=%.2f",
        request.headers.get("X-Request-ID", "n/a"),
        pipeline.model_store._get_latest_model_version(),
        prediction,
    )

    return prediction
