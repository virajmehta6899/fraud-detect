from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

load_dotenv()

from app.api.routes import router

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection with XGBoost + LLM Explainability",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "XGBoost",
        "roc_auc": 0.9813,
        "recall": 0.8776
    }