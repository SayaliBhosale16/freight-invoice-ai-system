import time

import numpy as np
import pandas as pd
from fastapi import APIRouter, Request

from app.schemas import (
    InvoiceRiskBatchRequest,
    InvoiceRiskRequest,
    InvoiceRiskResponse,
)

router = APIRouter(prefix="/predict/invoice-risk", tags=["Invoice Risk Flagging"])

FEATURE_COLUMNS = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "days_po_to_invoice",
    "days_to_pay",
    "total_brands",
    "total_item_quantity",
    "total_item_dollars",
    "avg_receiving_time",
]


def _predict_single(model, scaler, data: dict) -> tuple[int, float]:
    """Run prediction with scaling. Returns (flag, confidence)."""
    input_df = pd.DataFrame([data])
    input_df.columns = FEATURE_COLUMNS
    scaled = scaler.transform(input_df)
    flag = int(model.predict(scaled)[0])
    proba = model.predict_proba(scaled)[0]
    confidence = float(np.max(proba))
    return flag, confidence


@router.post("", response_model=InvoiceRiskResponse)
def predict_invoice_risk(request: InvoiceRiskRequest, req: Request):
    """Predict invoice risk flag for a single invoice."""
    start = time.time()

    model = req.app.state.invoice_model
    scaler = req.app.state.invoice_scaler
    version = req.app.state.registry.get_current_version("invoice")

    input_data = {
        "invoice_quantity": request.invoice_quantity,
        "invoice_dollars": request.invoice_dollars,
        "Freight": request.freight,
        "days_po_to_invoice": request.days_po_to_invoice,
        "days_to_pay": request.days_to_pay,
        "total_brands": request.total_brands,
        "total_item_quantity": request.total_item_quantity,
        "total_item_dollars": request.total_item_dollars,
        "avg_receiving_time": request.avg_receiving_time,
    }

    flag, confidence = _predict_single(model, scaler, input_data)

    latency_ms = (time.time() - start) * 1000
    req.app.state.prediction_logger.log(
        model_name="invoice",
        model_version=version,
        input_data=input_data,
        prediction={"risk_flag": flag, "confidence": confidence},
        latency_ms=latency_ms,
    )

    return InvoiceRiskResponse(
        risk_flag=flag,
        risk_label="risky" if flag == 1 else "normal",
        confidence=round(confidence, 4),
        model_version=version,
    )


@router.post("/batch", response_model=list[InvoiceRiskResponse])
def predict_invoice_risk_batch(request: InvoiceRiskBatchRequest, req: Request):
    """Predict invoice risk for a batch of invoices (max 100)."""
    start = time.time()

    model = req.app.state.invoice_model
    scaler = req.app.state.invoice_scaler
    version = req.app.state.registry.get_current_version("invoice")

    results = []
    input_records = []
    for item in request.items:
        input_data = {
            "invoice_quantity": item.invoice_quantity,
            "invoice_dollars": item.invoice_dollars,
            "Freight": item.freight,
            "days_po_to_invoice": item.days_po_to_invoice,
            "days_to_pay": item.days_to_pay,
            "total_brands": item.total_brands,
            "total_item_quantity": item.total_item_quantity,
            "total_item_dollars": item.total_item_dollars,
            "avg_receiving_time": item.avg_receiving_time,
        }
        input_records.append(input_data)
        flag, confidence = _predict_single(model, scaler, input_data)
        results.append(
            InvoiceRiskResponse(
                risk_flag=flag,
                risk_label="risky" if flag == 1 else "normal",
                confidence=round(confidence, 4),
                model_version=version,
            )
        )

    latency_ms = (time.time() - start) * 1000
    req.app.state.prediction_logger.log(
        model_name="invoice",
        model_version=version,
        input_data={"batch": input_records},
        prediction={"results": [r.model_dump() for r in results]},
        latency_ms=latency_ms,
    )

    return results
