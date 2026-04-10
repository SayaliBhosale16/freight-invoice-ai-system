import sqlite3

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from app.services.model_registry import ModelRegistry


@pytest.fixture
def test_db(tmp_path):
    """Create a small test SQLite database with vendor_invoice and purchases tables."""
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)

    conn.execute("""
        CREATE TABLE vendor_invoice (
            VendorNumber TEXT, VendorName TEXT, InvoiceDate TEXT,
            PONumber TEXT, PODate TEXT, PayDate TEXT,
            Quantity REAL, Dollars REAL, Freight REAL, Approval TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE purchases (
            InventoryId TEXT, Store TEXT, Brand TEXT, Description TEXT,
            Size TEXT, VendorNumber TEXT, VendorName TEXT,
            PONumber TEXT, PODate TEXT, ReceivingDate TEXT,
            InvoiceDate TEXT, PayDate TEXT,
            PurchasePrice REAL, Quantity REAL, Dollars REAL, Classification TEXT
        )
    """)

    # Insert test invoice data
    invoices = [
        ("V1", "Vendor A", "2024-01-15", "PO001", "2024-01-10", "2024-02-01", 10, 500.0, 25.0, "Y"),
        ("V2", "Vendor B", "2024-01-20", "PO002", "2024-01-12", "2024-02-05", 20, 1000.0, 50.0, "Y"),
        ("V3", "Vendor C", "2024-02-01", "PO003", "2024-01-25", "2024-02-15", 5, 250.0, 12.5, "N"),
        ("V1", "Vendor A", "2024-02-10", "PO004", "2024-02-01", "2024-03-01", 30, 1500.0, 75.0, "Y"),
        ("V2", "Vendor B", "2024-02-15", "PO005", "2024-02-10", "2024-03-05", 15, 750.0, 37.5, "Y"),
    ]
    conn.executemany(
        "INSERT INTO vendor_invoice VALUES (?,?,?,?,?,?,?,?,?,?)", invoices
    )

    # Insert test purchase data
    purchases = [
        ("I1", "S1", "BrandA", "Desc", "750ml", "V1", "Vendor A", "PO001", "2024-01-10", "2024-01-12", "2024-01-15", "2024-02-01", 10.0, 10, 500.0, "C1"),
        ("I2", "S1", "BrandB", "Desc", "750ml", "V2", "Vendor B", "PO002", "2024-01-12", "2024-01-25", "2024-01-20", "2024-02-05", 20.0, 20, 1000.0, "C1"),
        ("I3", "S2", "BrandC", "Desc", "1L", "V3", "Vendor C", "PO003", "2024-01-25", "2024-01-27", "2024-02-01", "2024-02-15", 5.0, 5, 250.0, "C2"),
        ("I4", "S1", "BrandA", "Desc", "750ml", "V1", "Vendor A", "PO004", "2024-02-01", "2024-02-03", "2024-02-10", "2024-03-01", 15.0, 30, 1500.0, "C1"),
        ("I5", "S2", "BrandB", "Desc", "1L", "V2", "Vendor B", "PO005", "2024-02-10", "2024-02-12", "2024-02-15", "2024-03-05", 10.0, 15, 750.0, "C1"),
    ]
    conn.executemany(
        "INSERT INTO purchases VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", purchases
    )

    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def test_models(tmp_path):
    """Create small test models and registry for API testing."""
    models_dir = tmp_path / "models"

    # Create freight model (simple linear regression)
    freight_model = LinearRegression()
    X = np.array([[100], [200], [500], [1000]])
    y = np.array([5, 10, 25, 50])
    freight_model.fit(X, y)

    # Create invoice model (simple random forest) + scaler
    # 9 features: invoice_quantity, invoice_dollars, Freight, days_po_to_invoice,
    #             days_to_pay, total_brands, total_item_quantity, total_item_dollars, avg_receiving_time
    invoice_model = RandomForestClassifier(n_estimators=10, random_state=42)
    X_inv = np.array([
        [10, 500, 25, 5, 15, 1, 100, 500, 3],
        [20, 1000, 50, 3, 10, 2, 200, 1000, 5],
        [5, 250, 12, 8, 20, 1, 50, 260, 12],
        [30, 1500, 75, 4, 12, 3, 300, 1500, 2],
        [15, 3000, 37, 10, 25, 2, 150, 750, 15],
        [8, 400, 20, 6, 18, 1, 80, 410, 11],
    ])
    y_inv = np.array([0, 0, 1, 0, 1, 1])
    scaler = StandardScaler()
    X_inv_scaled = scaler.fit_transform(X_inv)
    invoice_model.fit(X_inv_scaled, y_inv)

    # Save via registry
    registry = ModelRegistry(str(models_dir), str(models_dir / "registry.json"))

    fv = registry.save_version("freight", freight_model, {"mae": 1.0, "mse": 2.0, "r2": 99.0})
    registry.promote("freight", fv)

    iv = registry.save_version(
        "invoice", invoice_model,
        {"accuracy": 0.85, "f1_score": 0.80, "precision": 0.90, "recall": 0.75},
        scaler=scaler,
    )
    registry.promote("invoice", iv)

    return str(models_dir), str(models_dir / "registry.json"), registry


@pytest.fixture
def client(tmp_path, test_models):
    """FastAPI test client with test models loaded."""
    models_dir, registry_path, registry = test_models
    predictions_db = str(tmp_path / "predictions.db")

    # Patch settings before importing app
    from app.config import settings
    settings.models_dir = models_dir
    settings.model_registry_path = registry_path
    settings.predictions_db_path = predictions_db

    from app.main import app
    with TestClient(app) as c:
        yield c
