from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV


def train_random_forest(X_train, y_train) -> GridSearchCV:
    """Train a Random Forest classifier with GridSearchCV hyperparameter tuning."""
    model = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced")

    param_grid = {
        "n_estimators": [200, 300, 500],
        "max_depth": [None, 8, 10, 15],
        "min_samples_split": [2, 3, 5],
        "min_samples_leaf": [1, 2],
        "criterion": ["gini", "entropy"],
    }

    scorer = make_scorer(f1_score)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scorer,
        cv=5,
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)
    return grid_search


def evaluate_classifier(model, X_test, y_test) -> dict:
    """Evaluate classifier and return structured metrics dict."""
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, proba)
    report = classification_report(y_test, preds)

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "classification_report": report,
    }
