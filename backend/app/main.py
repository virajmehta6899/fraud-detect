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

from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
from fastapi.responses import JSONResponse
import time
from collections import defaultdict

class IPRateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests: int = 5, window_seconds: int = 86400):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.ip_records = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        # Only rate limit the core prediction endpoints, ignore metadata and health checks
        if not (request.url.path.endswith("/predict") or request.url.path.endswith("/explain")):
            return await call_next(request)
            
        client_ip = request.client.host
        now = time.time()
        
        # Clean up old requests
        self.ip_records[client_ip] = [
            timestamp for timestamp in self.ip_records[client_ip] 
            if now - timestamp < self.window_seconds
        ]
        
        if len(self.ip_records[client_ip]) >= self.max_requests:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Maximum 5 requests per day allowed from this IP."}
            )
            
        self.ip_records[client_ip].append(now)
        
        return await call_next(request)

app.add_middleware(IPRateLimitMiddleware, max_requests=5, window_seconds=86400)

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