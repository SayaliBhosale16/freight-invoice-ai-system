def test_predict_freight_valid(client):
    response = client.post("/predict/freight", json={"dollars": 1000})
    assert response.status_code == 200
    data = response.json()
    assert data["dollars"] == 1000
    assert "predicted_freight" in data
    assert isinstance(data["predicted_freight"], float)
    assert "model_version" in data


def test_predict_freight_invalid_negative(client):
    response = client.post("/predict/freight", json={"dollars": -100})
    assert response.status_code == 422


def test_predict_freight_invalid_zero(client):
    response = client.post("/predict/freight", json={"dollars": 0})
    assert response.status_code == 422


def test_predict_freight_batch(client):
    items = [{"dollars": d} for d in [100, 500, 1000]]
    response = client.post("/predict/freight/batch", json={"items": items})
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3
    for item in data:
        assert "predicted_freight" in item
        assert "model_version" in item


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "freight" in data["models"]
    assert "invoice" in data["models"]
    assert data["models"]["freight"]["loaded"] is True
