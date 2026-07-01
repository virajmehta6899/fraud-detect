import pickle
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
from src.explainer import explain_prediction

router = APIRouter()

# Load model and features
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../../models/best_model.pkl')
FEATURES_PATH = os.path.join(os.path.dirname(__file__), '../../../models/selected_features.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '../../../models/scaler.pkl')
METRICS_PATH = os.path.join(os.path.dirname(__file__), '../../../models/model_metrics.csv')

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(FEATURES_PATH, 'rb') as f:
    selected_features = pickle.load(f)

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)


# ── Schemas ───────────────────────────────────────────────────────────────────

class TransactionRequest(BaseModel):
    amount: float
    time: float = 0.0
    v1: float = 0.0
    v2: float = 0.0
    v3: float = 0.0
    v4: float = 0.0
    v5: float = 0.0
    v6: float = 0.0
    v7: float = 0.0
    v8: float = 0.0
    v9: float = 0.0
    v10: float = 0.0
    v11: float = 0.0
    v12: float = 0.0
    v13: float = 0.0
    v14: float = 0.0
    v15: float = 0.0
    v16: float = 0.0
    v17: float = 0.0
    v18: float = 0.0
    v19: float = 0.0
    v20: float = 0.0
    v21: float = 0.0
    v22: float = 0.0
    v23: float = 0.0
    v24: float = 0.0
    v25: float = 0.0
    v26: float = 0.0
    v27: float = 0.0
    v28: float = 0.0


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/predict")
async def predict(transaction: TransactionRequest):
    """Real-time fraud prediction."""
    try:
        # Build feature dict
        raw = {
            "V1": transaction.v1, "V2": transaction.v2,
            "V3": transaction.v3, "V4": transaction.v4,
            "V5": transaction.v5, "V6": transaction.v6,
            "V7": transaction.v7, "V8": transaction.v8,
            "V9": transaction.v9, "V10": transaction.v10,
            "V11": transaction.v11, "V12": transaction.v12,
            "V13": transaction.v13, "V14": transaction.v14,
            "V15": transaction.v15, "V16": transaction.v16,
            "V17": transaction.v17, "V18": transaction.v18,
            "V19": transaction.v19, "V20": transaction.v20,
            "V21": transaction.v21, "V22": transaction.v22,
            "V23": transaction.v23, "V24": transaction.v24,
            "V25": transaction.v25, "V26": transaction.v26,
            "V27": transaction.v27, "V28": transaction.v28,
            "Amount_Scaled": scaler.transform([[transaction.amount]])[0][0],
"Time_Scaled": scaler.transform([[transaction.time]])[0][0],
        }

        # Select only trained features
        input_df = pd.DataFrame([raw])[selected_features]

        # Predict
        prob = model.predict_proba(input_df)[0][1]
        prediction = "FRAUD" if prob >= 0.5 else "LEGITIMATE"
        risk_score = int(prob * 100)

        # Top features by importance
        importances = model.feature_importances_
        feature_importance = sorted(
            zip(selected_features, importances, input_df.values[0]),
            key=lambda x: x[1], reverse=True
        )[:5]

        # TO
        top_features = [
            {"feature": str(f), "importance": float(round(float(imp), 4)), "value": float(round(float(val), 4))}
            for f, imp, val in feature_importance
        ]

        return {
            "prediction": prediction,
            "confidence": float(round(float(prob), 4)),
            "risk_score": int(risk_score),
            "top_features": top_features,
            "amount": float(transaction.amount),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/explain")
async def explain(transaction: TransactionRequest):
    """Predict + generate LLM explanation."""
    try:
        # Get prediction first
        pred_result = await predict(transaction)

        # Generate LLM explanation
        explanation = explain_prediction(
            prediction=pred_result["prediction"],
            confidence=pred_result["confidence"],
            amount=transaction.amount,
            top_features=pred_result["top_features"],
        )

        return {
            **pred_result,
            "explanation": explanation,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@router.get("/model-info")
def model_info():
    """Return model metrics and info."""
    metrics = pd.read_csv(METRICS_PATH)
    best = metrics.iloc[0]
    return {
        "best_model": best["Model"],
        "metrics": {
            "roc_auc": best["ROC-AUC"],
            "f1_score": best["F1-Score"],
            "precision": best["Precision"],
            "recall": best["Recall"],
        },
        "dataset": "Kaggle Credit Card Fraud Detection",
        "total_transactions": 284807,
        "fraud_rate": "0.17%",
        "features_used": len(selected_features),
    }


@router.get("/sample-fraud")
def sample_fraud():
    """Returns a sample fraud transaction for testing."""
    return {
        "amount": 2125.87,
        "v1": -3.0435, "v2": -3.1573, "v3": 1.0884,
        "v4": 2.2886, "v5": 1.3597, "v6": -1.0903,
        "v7": -0.8773, "v8": 0.0502, "v9": -0.5920,
        "v10": -0.6178, "v11": 1.9658, "v12": -1.2324,
        "v13": 0.3826, "v14": -2.3598, "v15": 0.6446,
        "v16": -2.0857, "v17": -5.3656, "v18": -2.6022,
        "v19": 1.1095, "v20": -0.3220, "v21": 0.5280,
        "v22": 0.4008, "v23": -0.2296, "v24": 0.1256,
        "v25": 0.2230, "v26": 0.2453, "v27": 0.0835,
        "v28": 0.0416, "time": 406.0
    }