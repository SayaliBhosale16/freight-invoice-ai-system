import joblib
from pathlib import Path
# from frieght_cost_prediction.data_preprocessing import load_vendor_invoice_data, prepare_features, split_data
# This (relative import):
from data_preprocessing import load_vendor_invoice_data, prepare_features, split_data
from model_evaluation import train_linear_regression, train_decision_tree, train_random_forest, evaluate_model

def main():
    # Load and preprocess data
    db_path = 'data/inventory.db'
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)

    df = load_vendor_invoice_data(db_path)

    # Prepare features and target variable
    X, y = prepare_features(df)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train models
    model1 = train_linear_regression(X_train, y_train)
    model2 = train_decision_tree(X_train, y_train)
    model3 = train_random_forest(X_train, y_train)
    
    # Evaluate models
    results = []
    results.append(evaluate_model(model1, X_test, y_test, "Linear Regression"))
    results.append(evaluate_model(model2, X_test, y_test, "Decision Tree"))
    results.append(evaluate_model(model3, X_test, y_test, "Random Forest"))
    
    best_model = min(results, key=lambda x: x['mae'])
    best_model_name = best_model['model_name']

    best_model = {
        "Linear Regression": model1,
        "Decision Tree": model2,
        "Random Forest": model3
    }[best_model_name]

    model_path = model_dir / "predict_freight_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"Best model '{best_model_name}' saved to {model_path}")

    # # Save the best model (for simplicity, we save the random forest model here)
    # joblib.dump(model3, Path('best_model.joblib'))

if __name__ == "__main__":
    main()