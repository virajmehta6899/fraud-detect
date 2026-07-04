# 🔍 Fraud Detection System

End-to-end credit card fraud detection pipeline with
XGBoost + LLM Explainability.

🔗 **Live Demo:** https://fraud-detect-54fz.onrender.com/docs

---

## Architecture
Raw Transactions (284K) → EDA → Feature Engineering
→ SMOTE (handle imbalance) → Model Training
→ XGBoost (best model) → FastAPI Scoring API
→ Groq LLM Explainer → "Why was this flagged?"

## Model Performance

| Model | ROC-AUC | Recall | F1-Score |
|---|---|---|---|
| **XGBoost** | **0.9813** | **0.8776** | **0.4943** |
| Random Forest | 0.9777 | 0.8673 | 0.4521 |
| LightGBM | 0.9391 | 0.8265 | 0.4538 |
| Logistic Regression | 0.9705 | 0.9184 | 0.0955 |

## Tech Stack

| Layer | Technology |
|---|---|
| EDA | Pandas, Matplotlib, Seaborn |
| ML Models | Scikit-learn, XGBoost, LightGBM |
| Imbalance | SMOTE (imbalanced-learn) |
| Explainability | Groq LLM (Llama 3.3 70B) |
| Backend | FastAPI |
| Frontend | Streamlit + Plotly |
| Deployment | Render |

## Key Features

- EDA on 284K transactions with visualization
- SMOTE to handle 0.17% fraud imbalance
- 4 models compared — XGBoost selected
- Real-time scoring API (sub-second latency)
- LLM explainer — plain English explanation
- Risk gauge dashboard — Streamlit + Plotly

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/predict` | Real-time fraud prediction |
| POST | `/api/v1/explain` | Predict + LLM explanation |
| GET | `/api/v1/model-info` | Model metrics |
| GET | `/api/v1/sample-fraud` | Sample test transaction |

## Quick Start

```bash
cd backend
cp .env.example .env  # Add GROQ_API_KEY
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001
```

## Key Design Decisions

**Why XGBoost?**
Highest ROC-AUC (0.9813) with strong recall (87.76%).
In fraud detection, recall matters more than precision —
missing real fraud costs more than false alarms.

**Why SMOTE?**
Dataset has 0.17% fraud rate — 577:1 imbalance.
SMOTE synthesizes minority class samples to balance
training data without losing majority class information.

**Why LLM Explainer?**
Regulatory compliance in BFSI requires explainability.
Banks must justify why a transaction was flagged —
LLM translates model features into plain English.