import time

import numpy as np
import pandas as pd
from fastapi import APIRouter, Request

from app.schemas import (
    FreightBatchRequest,
    FreightPredictionRequest,
    FreightPredictionResponse,
)

router = APIRouter(prefix="/predict/freight", tags=["Freight Prediction"])


@router.post("", response_model=FreightPredictionResponse)
def predict_freight(request: FreightPredictionRequest, req: Request):
    """Predict freight cost for a single invoice."""
    start = time.time()

    model = req.app.state.freight_model
    version = req.app.state.registry.get_current_version("freight")

    input_df = pd.DataFrame({"Dollars": [request.dollars]})
    prediction = float(np.round(model.predict(input_df)[0], 2))

    latency_ms = (time.time() - start) * 1000
    req.app.state.prediction_logger.log(
        model_name="freight",
        model_version=version,
        input_data={"dollars": request.dollars},
        prediction={"predicted_freight": prediction},
        latency_ms=latency_ms,
    )

    return FreightPredictionResponse(
        dollars=request.dollars,
        predicted_freight=prediction,
        model_version=version,
    )


@router.post("/batch", response_model=list[FreightPredictionResponse])
def predict_freight_batch(request: FreightBatchRequest, req: Request):
    """Predict freight cost for a batch of invoices (max 100)."""
    start = time.time()

    model = req.app.state.freight_model
    version = req.app.state.registry.get_current_version("freight")

    dollars_list = [item.dollars for item in request.items]
    input_df = pd.DataFrame({"Dollars": dollars_list})
    predictions = np.round(model.predict(input_df), 2)

    latency_ms = (time.time() - start) * 1000
    req.app.state.prediction_logger.log(
        model_name="freight",
        model_version=version,
        input_data={"dollars": dollars_list},
        prediction={"predicted_freight": predictions.tolist()},
        latency_ms=latency_ms,
    )

    return [
        FreightPredictionResponse(
            dollars=d, predicted_freight=float(p), model_version=version
        )
        for d, p in zip(dollars_list, predictions)
    ]
