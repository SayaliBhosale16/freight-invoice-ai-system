import logging
from pathlib import Path

import joblib

from training.invoice.data_preprocessing import (
    apply_label,
    load_invoice_data,
    scale_features,
    split_data,
)
from training.invoice.model_evaluation import evaluate_classifier, train_random_forest

logger = logging.getLogger(__name__)

FEATURES = [
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

TARGET = "flag_invoice"


def train_and_evaluate(db_path: str, scaler_path: str = "models/scaler.pkl") -> tuple:
    """Train invoice risk model, evaluate, and return (model, scaler, metrics).

    Returns:
        tuple: (trained_model, scaler, metrics_dict) where metrics_dict has keys:
            accuracy, f1_score, precision, recall, classification_report
    """
    df = load_invoice_data(db_path)
    df = apply_label(df)

    X_train, X_test, y_train, y_test = split_data(df, FEATURES, TARGET)
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_test, scaler_path
    )

    grid_search = train_random_forest(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_

    metrics = evaluate_classifier(best_model, X_test_scaled, y_test)

    logger.info(
        f"Invoice model: Accuracy={metrics['accuracy']:.4f}, "
        f"F1={metrics['f1_score']:.4f}, Precision={metrics['precision']:.4f}"
    )
    logger.info(f"Best params: {grid_search.best_params_}")

    return best_model, scaler, metrics


def main():
    """Standalone training entrypoint."""
    logging.basicConfig(level=logging.INFO)
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    model, scaler, metrics = train_and_evaluate(
        db_path="data/inventory.db",
        scaler_path=str(model_dir / "scaler.pkl"),
    )

    joblib.dump(model, model_dir / "predict_flag_invoice.pkl")
    print(f"Model saved. Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")


if __name__ == "__main__":
    main()
