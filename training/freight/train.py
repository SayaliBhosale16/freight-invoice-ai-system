import logging

import joblib
from pathlib import Path

from training.freight.data_preprocessing import (
    load_vendor_invoice_data,
    prepare_features,
    split_data,
)
from training.freight.model_evaluation import (
    evaluate_model,
    train_decision_tree,
    train_linear_regression,
    train_random_forest,
)

logger = logging.getLogger(__name__)


def train_and_evaluate(db_path: str) -> tuple:
    """Train all freight models, evaluate, and return (best_model, best_metrics).

    Returns:
        tuple: (trained_model, metrics_dict) where metrics_dict has keys:
            model_name, mae, mse, r2
    """
    df = load_vendor_invoice_data(db_path)
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train all three models
    models = {
        "Linear Regression": train_linear_regression(X_train, y_train),
        "Decision Tree": train_decision_tree(X_train, y_train),
        "Random Forest": train_random_forest(X_train, y_train),
    }

    # Evaluate each
    results = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)
        logger.info(f"{name}: MAE={metrics['mae']:.2f}, R²={metrics['r2']:.2f}%")

    # Select best by lowest MAE
    best_result = min(results, key=lambda x: x["mae"])
    best_model = models[best_result["model_name"]]

    logger.info(f"Best model: {best_result['model_name']}")
    return best_model, best_result


def main():
    """Standalone training entrypoint."""
    logging.basicConfig(level=logging.INFO)
    db_path = "data/inventory.db"
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    model, metrics = train_and_evaluate(db_path)

    model_path = model_dir / "predict_freight_model.pkl"
    joblib.dump(model, model_path)
    print(f"Best model '{metrics['model_name']}' saved to {model_path}")
    print(f"Metrics: MAE={metrics['mae']:.2f}, MSE={metrics['mse']:.2f}, R²={metrics['r2']:.2f}%")


if __name__ == "__main__":
    main()
