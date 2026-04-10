import logging
import json
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import settings
from app.routers import dashboard, freight, invoice, retrain
from app.schemas import HealthResponse, ModelInfo
from app.services.model_registry import ModelRegistry
from app.services.prediction_logger import PredictionLogger

# --- Structured JSON logging ---

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        return json.dumps(log_entry)


def setup_logging():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logging.root.handlers = [handler]
    logging.root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))


# --- App lifecycle ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and services on startup."""
    setup_logging()
    logger = logging.getLogger(__name__)

    # Initialize model registry
    registry = ModelRegistry(settings.models_dir, settings.model_registry_path)
    app.state.registry = registry

    # Load freight model
    try:
        app.state.freight_model = registry.load_model("freight")
        logger.info(f"Freight model loaded: {registry.get_current_version('freight')}")
    except FileNotFoundError:
        logger.warning("No freight model found in registry")
        app.state.freight_model = None

    # Load invoice model + scaler
    try:
        loaded = registry.load_model("invoice")
        app.state.invoice_model = loaded[0]
        app.state.invoice_scaler = loaded[1]
        logger.info(f"Invoice model loaded: {registry.get_current_version('invoice')}")
    except FileNotFoundError:
        logger.warning("No invoice model found in registry")
        app.state.invoice_model = None
        app.state.invoice_scaler = None

    # Initialize prediction logger
    app.state.prediction_logger = PredictionLogger(settings.predictions_db_path)

    yield


# --- App factory ---

app = FastAPI(
    title="Freight Analytics Engine",
    description="Dual-model ML API for freight cost prediction and invoice risk flagging",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(freight.router)
app.include_router(invoice.router)
app.include_router(retrain.router)
app.include_router(dashboard.router)


# --- Exception handlers ---

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.getLogger(__name__).error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# --- Health & info endpoints ---

@app.get("/health", response_model=HealthResponse)
def health_check(request: Request):
    """Health check with model status and prediction count."""
    registry = request.app.state.registry
    pred_logger = request.app.state.prediction_logger

    models = {}
    for name in ["freight", "invoice"]:
        version = registry.get_current_version(name)
        metrics = registry.get_current_metrics(name)
        loaded = getattr(request.app.state, f"{name}_model", None) is not None
        models[name] = ModelInfo(
            version=version or "not loaded",
            loaded=loaded,
            metrics=metrics,
        )

    return HealthResponse(
        status="healthy",
        models=models,
        prediction_count_24h=pred_logger.get_count_since(24),
    )


@app.get("/models/info")
def models_info(request: Request):
    """Full model registry information."""
    return request.app.state.registry.get_all_info()
