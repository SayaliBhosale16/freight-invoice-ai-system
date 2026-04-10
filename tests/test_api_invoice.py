VALID_INVOICE = {
    "invoice_quantity": 10,
    "invoice_dollars": 500,
    "freight": 25,
    "days_po_to_invoice": 5,
    "days_to_pay": 15,
    "total_brands": 1,
    "total_item_quantity": 100,
    "total_item_dollars": 500,
    "avg_receiving_time": 3,
}


def test_predict_invoice_risk_valid(client):
    response = client.post("/predict/invoice-risk", json=VALID_INVOICE)
    assert response.status_code == 200
    data = response.json()
    assert data["risk_flag"] in [0, 1]
    assert data["risk_label"] in ["normal", "risky"]
    assert 0 <= data["confidence"] <= 1
    assert "model_version" in data


def test_predict_invoice_risk_invalid(client):
    response = client.post("/predict/invoice-risk", json={"invoice_quantity": -1})
    assert response.status_code == 422


def test_predict_invoice_risk_batch(client):
    items = [VALID_INVOICE, {**VALID_INVOICE, "invoice_dollars": 3000}]
    response = client.post("/predict/invoice-risk/batch", json={"items": items})
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    for item in data:
        assert item["risk_flag"] in [0, 1]
        assert 0 <= item["confidence"] <= 1


def test_models_info(client):
    response = client.get("/models/info")
    assert response.status_code == 200
    data = response.json()
    assert "freight" in data
    assert "invoice" in data
