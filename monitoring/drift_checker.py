"""
Basic data drift detection script.

Compares the distribution of recent prediction inputs against
training-time reference statistics stored in model metadata.

Usage:
    python -m monitoring.drift_checker --model freight --limit 100
    python -m monitoring.drift_checker --model invoice --limit 50
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, ".")

from app.services.model_registry import ModelRegistry
from app.services.prediction_logger import PredictionLogger


DRIFT_THRESHOLD = 2.0  # flag if mean shifts > 2 std deviations


def compute_stats(values: list[float]) -> dict:
    """Compute mean and std for a list of values."""
    arr = np.array(values, dtype=float)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "count": len(arr)}


def check_drift(
    model_name: str,
    models_dir: str = "models",
    registry_path: str = "models/registry.json",
    predictions_db: str = "data/predictions.db",
    limit: int = 100,
) -> dict:
    """Check for input drift on recent predictions vs training reference."""
    registry = ModelRegistry(models_dir, registry_path)
    logger = PredictionLogger(predictions_db)

    version = registry.get_current_version(model_name)
    if not version:
        return {"error": f"No active version for '{model_name}'"}

    # Load training reference stats from metadata
    version_path = registry.get_version_path(model_name, version)
    metadata_path = version_path / "metadata.json"
    if not metadata_path.exists():
        return {"error": f"No metadata found at {metadata_path}"}

    with open(metadata_path) as f:
        metadata = json.load(f)

    ref_stats = metadata.get("input_stats")
    if not ref_stats:
        return {
            "warning": "No input_stats in metadata. Run training with stats collection to enable drift detection.",
            "model": model_name,
            "version": version,
        }

    # Get recent prediction inputs
    recent_inputs = logger.get_recent_inputs(model_name, limit=limit)
    if not recent_inputs:
        return {"warning": "No recent predictions found", "model": model_name}

    # Compare distributions
    drift_report = {
        "model": model_name,
        "version": version,
        "recent_count": len(recent_inputs),
        "features": {},
        "drifted": False,
    }

    for feature, ref in ref_stats.items():
        values = [inp.get(feature) for inp in recent_inputs if feature in inp]
        if not values:
            drift_report["features"][feature] = {"status": "no_data"}
            continue

        current = compute_stats(values)
        ref_mean = ref["mean"]
        ref_std = ref["std"]

        if ref_std > 0:
            z_score = abs(current["mean"] - ref_mean) / ref_std
        else:
            z_score = 0.0 if current["mean"] == ref_mean else float("inf")

        is_drifted = z_score > DRIFT_THRESHOLD
        if is_drifted:
            drift_report["drifted"] = True

        drift_report["features"][feature] = {
            "reference_mean": round(ref_mean, 4),
            "current_mean": round(current["mean"], 4),
            "reference_std": round(ref_std, 4),
            "z_score": round(z_score, 4),
            "status": "DRIFT" if is_drifted else "OK",
        }

    return drift_report


def main():
    parser = argparse.ArgumentParser(description="Check for input data drift")
    parser.add_argument("--model", required=True, choices=["freight", "invoice"])
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    report = check_drift(model_name=args.model, limit=args.limit)
    print(json.dumps(report, indent=2))

    if report.get("drifted"):
        print("\n*** DRIFT DETECTED — consider retraining the model ***")
        sys.exit(1)


if __name__ == "__main__":
    main()
