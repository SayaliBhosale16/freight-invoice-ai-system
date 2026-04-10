import pandas as pd

from training.invoice.data_preprocessing import (
    create_invoice_risk_label,
    load_invoice_data,
    apply_label,
)


def test_risk_label_dollar_mismatch():
    """Flag when invoice_dollars and total_item_dollars differ by >$5."""
    row = {"invoice_dollars": 100, "total_item_dollars": 110, "avg_receiving_time": 5}
    assert create_invoice_risk_label(row) == 1


def test_risk_label_slow_receiving():
    """Flag when avg_receiving_time > 10 days."""
    row = {"invoice_dollars": 100, "total_item_dollars": 102, "avg_receiving_time": 15}
    assert create_invoice_risk_label(row) == 1


def test_risk_label_normal():
    """No flag when amounts match and receiving is fast."""
    row = {"invoice_dollars": 100, "total_item_dollars": 103, "avg_receiving_time": 5}
    assert create_invoice_risk_label(row) == 0


def test_load_invoice_data(test_db):
    df = load_invoice_data(test_db)
    assert isinstance(df, pd.DataFrame)
    assert "invoice_quantity" in df.columns
    assert "invoice_dollars" in df.columns
    assert "total_item_dollars" in df.columns


def test_apply_label(test_db):
    df = load_invoice_data(test_db)
    df = apply_label(df)
    assert "flag_invoice" in df.columns
    assert set(df["flag_invoice"].unique()).issubset({0, 1})
