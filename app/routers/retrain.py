import logging

from fastapi import APIRouter, HTTPException, Request

from app.config import settings
from app.schemas import RetrainResponse

router = APIRouter(prefix="/retrain", tags=["Retraining"])
logger = logging.getLogger(__name__)


@router.post("/freight", response_model=RetrainResponse)
def retrain_freight(req: Request):
    """Retrain the freight cost prediction model and optionally promote it."""
    from training.freight.train import train_and_evaluate

    registry = req.app.state.registry
    old_version = registry.get_current_version("freight") or "none"
    old_metrics = registry.get_current_metrics("freight")

    try:
        model, new_metrics = train_and_evaluate(settings.db_path)
    except Exception as e:
        logger.error(f"Freight retraining failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")

    new_version = registry.save_version(
        model_name="freight",
        model=model,
        metrics=new_metrics,
        algorithm=new_metrics.get("model_name", ""),
    )

    # Decide whether to promote
    promoted = False
    if not old_metrics or old_version == "none":
        promoted = True
    else:
        old_mae = old_metrics.get("mae", float("inf"))
        new_mae = new_metrics.get("mae", float("inf"))
        improvement = (old_mae - new_mae) / old_mae if old_mae > 0 else 0
        if improvement >= settings.retrain_min_improvement:
            promoted = True

    if promoted:
        registry.promote("freight", new_version)
        req.app.state.freight_model = registry.load_model("freight")
        logger.info(f"Freight model promoted to {new_version}")

    return RetrainResponse(
        status="completed",
        model_name="freight",
        old_version=old_version,
        new_version=new_version,
        old_metrics=old_metrics,
        new_metrics={k: v for k, v in new_metrics.items() if k != "classification_report"},
        promoted=promoted,
    )


@router.post("/invoice", response_model=RetrainResponse)
def retrain_invoice(req: Request):
    """Retrain the invoice risk flagging model and optionally promote it."""
    from training.invoice.train import train_and_evaluate

    registry = req.app.state.registry
    old_version = registry.get_current_version("invoice") or "none"
    old_metrics = registry.get_current_metrics("invoice")

    try:
        model, scaler, new_metrics = train_and_evaluate(settings.db_path)
    except Exception as e:
        logger.error(f"Invoice retraining failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")

    new_version = registry.save_version(
        model_name="invoice",
        model=model,
        metrics=new_metrics,
        scaler=scaler,
        algorithm="RandomForestClassifier",
    )

    # Decide whether to promote
    promoted = False
    if not old_metrics or old_version == "none":
        promoted = True
    else:
        old_f1 = old_metrics.get("f1_score", 0)
        new_f1 = new_metrics.get("f1_score", 0)
        improvement = (new_f1 - old_f1) / old_f1 if old_f1 > 0 else 1
        if improvement >= settings.retrain_min_improvement:
            promoted = True

    if promoted:
        registry.promote("invoice", new_version)
        loaded = registry.load_model("invoice")
        req.app.state.invoice_model = loaded[0]
        req.app.state.invoice_scaler = loaded[1]
        logger.info(f"Invoice model promoted to {new_version}")

    return RetrainResponse(
        status="completed",
        model_name="invoice",
        old_version=old_version,
        new_version=new_version,
        old_metrics=old_metrics,
        new_metrics={k: v for k, v in new_metrics.items() if k != "classification_report"},
        promoted=promoted,
    )
