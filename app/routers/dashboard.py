import sqlite3
from datetime import datetime, timedelta

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.config import settings

router = APIRouter(tags=["Dashboard"])
templates = Jinja2Templates(directory="app/templates")


def _get_prediction_stats(db_path: str) -> dict:
    """Gather prediction statistics for the dashboard."""
    stats = {
        "total": 0,
        "last_24h": 0,
        "freight_count": 0,
        "invoice_count": 0,
        "avg_latency_ms": 0.0,
        "recent": [],
        "hourly_counts": [],
    }

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        # Total predictions
        stats["total"] = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

        # Last 24h
        cutoff_24h = (datetime.now() - timedelta(hours=24)).isoformat()
        stats["last_24h"] = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE timestamp > ?", (cutoff_24h,)
        ).fetchone()[0]

        # Per-model counts
        stats["freight_count"] = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE model_name = 'freight'"
        ).fetchone()[0]
        stats["invoice_count"] = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE model_name = 'invoice'"
        ).fetchone()[0]

        # Average latency
        row = conn.execute("SELECT AVG(latency_ms) FROM predictions").fetchone()
        stats["avg_latency_ms"] = round(row[0], 2) if row[0] else 0.0

        # Recent predictions (last 10)
        rows = conn.execute(
            """SELECT timestamp, model_name, model_version, input_data, prediction, latency_ms
               FROM predictions ORDER BY timestamp DESC LIMIT 10"""
        ).fetchall()
        stats["recent"] = [dict(r) for r in rows]

        # Hourly counts for chart (last 24 hours)
        for i in range(23, -1, -1):
            start = (datetime.now() - timedelta(hours=i + 1)).isoformat()
            end = (datetime.now() - timedelta(hours=i)).isoformat()
            count = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE timestamp > ? AND timestamp <= ?",
                (start, end),
            ).fetchone()[0]
            hour_label = (datetime.now() - timedelta(hours=i)).strftime("%H:00")
            stats["hourly_counts"].append({"hour": hour_label, "count": count})

        conn.close()
    except Exception:
        pass

    return stats


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    """Render the monitoring dashboard."""
    registry = request.app.state.registry

    # Model info
    models = {}
    for name in ["freight", "invoice"]:
        version = registry.get_current_version(name)
        metrics = registry.get_current_metrics(name)
        all_info = registry.get_all_info().get(name, {})
        version_count = len(all_info.get("versions", {}))
        loaded = getattr(request.app.state, f"{name}_model", None) is not None
        models[name] = {
            "version": version or "not loaded",
            "loaded": loaded,
            "metrics": metrics,
            "version_count": version_count,
        }

    # Prediction stats
    pred_stats = _get_prediction_stats(settings.predictions_db_path)

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "models": models,
            "stats": pred_stats,
        },
    )
