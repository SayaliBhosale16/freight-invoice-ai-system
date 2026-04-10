# Freight Analytics Engine

> Production-grade ML system for freight cost prediction and invoice risk flagging — featuring FastAPI serving, model versioning, automated retraining, observability dashboard, and Docker deployment.

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?style=flat-square&logo=fastapi)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat-square&logo=docker)
![Tests](https://img.shields.io/badge/Tests-17%20Passing-green?style=flat-square)

---

## Problem Statement

Finance teams lose thousands monthly to unpredictable freight costs and undetected risky invoices. This system automates both — **predicting shipping costs** before they happen and **flagging invoice anomalies** before they drain revenue.


---

## Dual-Model Design

| Model | Task | Algorithm | Key Metrics | Results |
|-------|------|-----------|-------------|---------|
| Freight Cost Prediction | Regression | Linear Regression (best of 3) | MAE / RMSE / R² | **MAE: 24.11, RMSE: 124.72, R²: 97%** |
| Invoice Risk Flagging | Classification | Random Forest + GridSearchCV | F1 / ROC-AUC / Precision | **F1: 0.82, ROC-AUC: 0.87, Precision: 96%** |

**Freight model** compares Linear Regression, Decision Tree, and Random Forest — selects the one with the lowest MAE. 

**Invoice model** uses SQL-based feature engineering (CTE aggregations across purchase orders) and flags invoices where dollar amounts don't match PO totals or receiving is abnormally slow. Tuned with 5-fold cross-validated GridSearchCV optimizing for F1 score.

---
<img width="1337" height="814" alt="image" src="https://github.com/user-attachments/assets/b3eb4cdf-17dd-4794-a756-96dc904c73a8" />

---

## Project Structure

```
freight-analytics-engine/
├── app/                              # FastAPI application
│   ├── main.py                       # App factory, lifespan, health endpoint
│   ├── config.py                     
│   ├── schemas.py                    # Pydantic request/response models
│   ├── templates/
│   │   └── dashboard.html            # Observability dashboard
│   ├── routers/
│   │   ├── freight.py                # /predict/freight endpoints
│   │   ├── invoice.py                # /predict/invoice-risk endpoints
│   │   ├── retrain.py                # /retrain/{model} endpoints
│   │   └── dashboard.py              # /dashboard UI
│   └── services/
│       ├── model_registry.py         # JSON-based model versioning
│       └── prediction_logger.py      # SQLite prediction logging
├── training/                         # ML training pipelines
│   ├── freight/
│   │   ├── data_preprocessing.py     
│   │   ├── model_evaluation.py      
│   │   └── train.py                  
│   └── invoice/
│       ├── data_preprocessing.py    
│       ├── model_evaluation.py       
│       └── train.py                  
├── models/                           
│   ├── freight/v1_{timestamp}/
│   │   ├── model.pkl
│   │   └── metadata.json
│   ├── invoice/v1_{timestamp}/
│   │   ├── model.pkl, scaler.pkl
│   │   └── metadata.json
│   └── registry.json                 # Tracks current active version per model
├── monitoring/
│   └── drift_checker.py              # Input distribution drift detection
├── tests/                            # 17 pytest tests
│   ├── conftest.py                   # Fixtures: test DB, test models, test client
│   ├── test_freight_preprocessing.py
│   ├── test_invoice_preprocessing.py
│   ├── test_api_freight.py
│   └── test_api_invoice.py
├── notebooks/                        # Exploratory analysis
│   ├── predicting_fright_cost.ipynb
│   └── invoice_flagging.ipynb
├── data/
│   └── inventory.db                  # SQLite database (5 tables, 5.5K invoices)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .github/workflows/ci.yml          # Lint + Test + Docker build
└── bootstrap_registry.py             # One-time model migration script
```

---

## Quick Start

### Local

```bash
# Clone and install
git clone https://github.com/SayaliBhosale16/freight-analytics-engine.git
cd freight-analytics-engine
pip install -r requirements.txt

# Bootstrap existing models into the versioned registry
python bootstrap_registry.py

# Start the API server
uvicorn app.main:app --reload
```

### Docker

```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`.

---

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check with model versions and prediction count |
| `GET` | `/docs` | Interactive Swagger API documentation |
| `GET` | `/dashboard` | Observability dashboard with live testing |
| `GET` | `/models/info` | Full model registry details |
| `POST` | `/predict/freight` | Predict freight cost for a single invoice |
| `POST` | `/predict/freight/batch` | Batch freight predictions (max 100) |
| `POST` | `/predict/invoice-risk` | Flag a single invoice for risk |
| `POST` | `/predict/invoice-risk/batch` | Batch invoice risk flagging (max 100) |
| `POST` | `/retrain/freight` | Retrain freight model, auto-promote if improved |
| `POST` | `/retrain/invoice` | Retrain invoice model, auto-promote if improved |

### Example API Calls

```bash
# Predict freight cost
curl -X POST http://localhost:8000/predict/freight \
  -H "Content-Type: application/json" \
  -d '{"dollars": 1500}'

# Response: {"dollars": 1500.0, "predicted_freight": 12.53, "model_version": "v1_..."}

# Check invoice risk
curl -X POST http://localhost:8000/predict/invoice-risk \
  -H "Content-Type: application/json" \
  -d '{
    "invoice_quantity": 30,
    "invoice_dollars": 3000,
    "freight": 150,
    "total_item_quantity": 200,
    "total_item_dollars": 750
  }'

# Response: {"risk_flag": 1, "risk_label": "risky", "confidence": 0.797, "model_version": "v1_..."}

# Trigger retraining
curl -X POST http://localhost:8000/retrain/freight

# Response: {"status": "completed", "promoted": true, "new_version": "v2_...", ...}
```

---

## MLOps Features

### Model Registry
JSON-based versioning system. Each model version is stored in its own directory with `model.pkl`, optional `scaler.pkl`, and `metadata.json` (metrics, algorithm, timestamp). The registry tracks which version is "current" and only promotes a new model if it outperforms the active one.

### Automated Retraining
Hit `/retrain/{model}` to trigger a full training run. The system:
1. Trains the model on the latest data
2. Evaluates against the test set
3. Compares metrics with the current model
4. Auto-promotes only if improvement exceeds the configured threshold (default: 1%)

### Prediction Logging
Every prediction is logged to `data/predictions.db` with timestamp, model name, version, input data, output, and latency. This powers the dashboard activity chart and enables drift detection.

### Drift Detection
```bash
python -m monitoring.drift_checker --model freight --limit 100
```
Compares recent prediction input distributions against training-time reference statistics. Flags features where the mean has shifted by more than 2 standard deviations.

### Observability Dashboard
A single-page dashboard at `/dashboard` showing:
- Real-time prediction counts and latency metrics
- Model registry with performance metrics and retrain buttons
- Live prediction testers for both models
- Hourly prediction activity chart
- Recent prediction log

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **API** | FastAPI, Pydantic v2, Uvicorn |
| **ML** | Scikit-learn (Linear Regression, Random Forest, GridSearchCV) |
| **Data** | SQLite, Pandas, SQL CTEs for feature engineering |
| **Config** | pydantic-settings, environment variables |
| **Testing** | pytest, httpx (17 tests) |
| **CI/CD** | GitHub Actions (ruff lint + pytest + Docker build) |
| **Deployment** | Docker, docker-compose |
| **Monitoring** | SQLite prediction logging, statistical drift detection |
| **Dashboard** | Jinja2 templates served from FastAPI |

---


