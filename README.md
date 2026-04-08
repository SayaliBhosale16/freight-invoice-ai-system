# 🚛 freight-analytics-engine

> End-to-end ML pipeline for freight cost prediction & invoice risk flagging — SQL feature engineering, dual-model architecture, and Streamlit deployment.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat-square&logo=streamlit)
![SQLite](https://img.shields.io/badge/SQLite-Database-lightblue?style=flat-square&logo=sqlite)
![Status](https://img.shields.io/badge/Status-Production--Ready-green?style=flat-square)

---

## 🧠 Problem Statement

Finance teams lose thousands monthly to unpredictable freight costs and undetected risky invoices. This system automates both — predicting costs before they happen and flagging anomalies before they drain revenue.

---

## ⚙️ Dual-Model Architecture

| Module | Task | Algorithm | Metric |
|--------|------|-----------|--------|
| 01 — Freight Cost Prediction | Regression | Random Forest, Linear Regression, Decision Tree | MAE, RMSE, R² |
| 02 — Invoice Risk Flagging | Classification | Random Forest Classifier | Precision, Recall, F1-Score |

---

## 🗂️ Project Structure

```
freight-analytics-engine/
├── data/
│   └── inventory.db          # SQLite relational database
├── frieght_cost_prediction/
│   ├── data_preprocessing.py # SQL feature engineering
│   ├── model_evaluation.py   # Train & evaluate models
│   └── train.py              # Main training pipeline
├── models/                   # Saved .pkl files
├── notebooks/
│   └── predicting_freight_cost.ipynb
├── app.py                    # Streamlit dashboard
└── README.md
```

---

## 🛠️ Tech Stack

- **Data** — SQLite, Pandas, NumPy, SQLAlchemy
- **ML** — Scikit-learn, XGBoost
- **Visualization** — Seaborn, Matplotlib, Plotly
- **Deployment** — 


---

## 📊 Pipeline Flow

```
SQLite DB → SQL Feature Engineering → EDA → Modeling → Evaluation → .pkl → App
```

---
