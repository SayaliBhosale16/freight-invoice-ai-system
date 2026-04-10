from modeling_evaluation import train_random_forest, evaluate_classifier
import joblib
from data_preprocessing import load_invoice_data, apply_label, split_data, scale_features
import sqlite3

FEATURES = [
    'invoice_quantity',
    'invoice_dollars',
    'Freight',
    'total_item_quantity',
    'total_item_dollars',   
]

TARGET = 'flag_invoice'

def check_database_tables(db_path):
    conn = sqlite3.connect(db_path)
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    conn.close()
    return tables

def main():
    df = load_invoice_data("data/inventory.db")
    df = apply_label(df)

    X_train, X_test, y_train, y_test = split_data(df, FEATURES, TARGET)

    X_train_scaled, X_test_scaled = scale_features(X_train, X_test, "models/scaler.pkl")

    grid_search = train_random_forest(X_train_scaled, y_train)

    evaluate_classifier(grid_search.best_estimator_, X_test_scaled, y_test, "Tuned Random Forest")

    joblib.dump(grid_search.best_estimator_, "models/predict_flag_invoice.pkl")

if __name__ == "__main__":
    print("Checking database tables...")
    print(check_database_tables("data/inventory.db"))

    main()