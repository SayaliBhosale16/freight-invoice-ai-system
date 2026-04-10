"""Bootstrap models into the versioned registry with full metrics."""
import logging
import sys

sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO)

from app.services.model_registry import ModelRegistry
from training.freight.train import train_and_evaluate as train_freight
from training.invoice.train import train_and_evaluate as train_invoice


def main():
    registry = ModelRegistry("models", "models/registry.json")

    # Train and register freight model
    print("Training freight model...")
    model, metrics = train_freight("data/inventory.db")
    version = registry.save_version(
        model_name="freight",
        model=model,
        metrics=metrics,
        algorithm=metrics.get("model_name", ""),
    )
    registry.promote("freight", version)
    print(f"Freight model: {version}")
    print(f"  MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.1f}%")

    # Train and register invoice model
    print("\nTraining invoice model...")
    model, scaler, metrics = train_invoice("data/inventory.db")
    version = registry.save_version(
        model_name="invoice",
        model=model,
        metrics=metrics,
        scaler=scaler,
        algorithm="RandomForestClassifier",
    )
    registry.promote("invoice", version)
    print(f"Invoice model: {version}")
    print(f"  F1={metrics['f1_score']:.2f}, ROC-AUC={metrics['roc_auc']:.2f}, Precision={metrics['precision']:.2f}")

    print("\nRegistry bootstrapped successfully!")


if __name__ == "__main__":
    main()
