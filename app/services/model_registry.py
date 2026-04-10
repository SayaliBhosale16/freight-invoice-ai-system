import json
import logging
from datetime import datetime
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)


class ModelRegistry:
    """JSON-based model registry for versioning and promoting models."""

    def __init__(self, models_dir: str, registry_path: str):
        self.models_dir = Path(models_dir)
        self.registry_path = Path(registry_path)
        self._registry = self._load_registry()

    def _load_registry(self) -> dict:
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                return json.load(f)
        return {}

    def _save_registry(self):
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w") as f:
            json.dump(self._registry, f, indent=2, default=str)

    def get_current_version(self, model_name: str) -> str | None:
        entry = self._registry.get(model_name)
        if entry:
            return entry.get("current_version")
        return None

    def get_current_metrics(self, model_name: str) -> dict:
        version = self.get_current_version(model_name)
        if not version:
            return {}
        return self._registry[model_name]["versions"][version].get("metrics", {})

    def get_version_path(self, model_name: str, version: str) -> Path:
        return self.models_dir / model_name / version

    def load_model(self, model_name: str):
        """Load the current version of a model. Returns model (and scaler for invoice)."""
        version = self.get_current_version(model_name)
        if not version:
            raise FileNotFoundError(f"No registered version for model '{model_name}'")

        version_path = self.get_version_path(model_name, version)
        model = joblib.load(version_path / "model.pkl")

        scaler_path = version_path / "scaler.pkl"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            return model, scaler

        return model

    def save_version(
        self,
        model_name: str,
        model,
        metrics: dict,
        scaler=None,
        algorithm: str = "",
    ) -> str:
        """Save a new model version and return the version string."""
        # Determine next version number
        if model_name in self._registry:
            existing = self._registry[model_name].get("versions", {})
            version_num = len(existing) + 1
        else:
            version_num = 1

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"v{version_num}_{timestamp}"

        version_path = self.get_version_path(model_name, version)
        version_path.mkdir(parents=True, exist_ok=True)

        # Save artifacts
        joblib.dump(model, version_path / "model.pkl")
        if scaler is not None:
            joblib.dump(scaler, version_path / "scaler.pkl")

        # Save metadata
        metadata = {
            "created_at": datetime.now().isoformat(),
            "metrics": {k: v for k, v in metrics.items() if k != "classification_report"},
            "algorithm": algorithm,
        }
        with open(version_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        # Update registry
        if model_name not in self._registry:
            self._registry[model_name] = {"current_version": None, "versions": {}}

        self._registry[model_name]["versions"][version] = metadata
        self._save_registry()

        logger.info(f"Saved {model_name} version {version}")
        return version

    def promote(self, model_name: str, version: str):
        """Set a version as the current active model."""
        if model_name not in self._registry:
            raise ValueError(f"Model '{model_name}' not in registry")
        if version not in self._registry[model_name]["versions"]:
            raise ValueError(f"Version '{version}' not found for '{model_name}'")

        self._registry[model_name]["current_version"] = version
        self._save_registry()
        logger.info(f"Promoted {model_name} to {version}")

    def get_all_info(self) -> dict:
        """Return registry info for all models."""
        return self._registry
