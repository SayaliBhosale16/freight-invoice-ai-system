import pandas as pd

from training.freight.data_preprocessing import (
    load_vendor_invoice_data,
    prepare_features,
    split_data,
)


def test_load_vendor_invoice_data(test_db):
    df = load_vendor_invoice_data(test_db)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert "Dollars" in df.columns
    assert "Freight" in df.columns


def test_prepare_features(test_db):
    df = load_vendor_invoice_data(test_db)
    X, y = prepare_features(df)
    assert list(X.columns) == ["Dollars"]
    assert y.name == "Freight"
    assert len(X) == len(y) == 5


def test_split_data(test_db):
    df = load_vendor_invoice_data(test_db)
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.4)
    assert len(X_train) == 3
    assert len(X_test) == 2
