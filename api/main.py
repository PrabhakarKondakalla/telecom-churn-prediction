from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path

# Make src/ importable
sys.path.append(str(Path(__file__).resolve().parent.parent))

from api.schema import CustomerInput, ChurnResponse, HealthResponse
from src.predict import predict, THRESHOLD, FEATURES

app = FastAPI(
    title       = "Telecom Churn Prediction API",
    description = "Scores a customer's churn risk using a trained XGBoost model on Cell2Cell data.",
    version     = "1.0.0",
)

# Allow frontend / CRM tools to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

RECOMMENDATIONS = {
    "high"  : "Immediate action — offer a discount or loyalty reward within 24 hours.",
    "medium": "Monitor closely — send a satisfaction survey and a soft retention offer.",
    "low"   : "No action needed — customer is stable.",
}


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Check if the API and model are loaded and running."""
    return HealthResponse(
        status  = "ok",
        model   = "XGBoost — Cell2Cell Churn",
        version = "1.0.0",
    )


@app.get("/", tags=["System"])
def root():
    return {
        "message" : "Churn Prediction API is running.",
        "docs"    : "/docs",
        "health"  : "/health",
    }


@app.post("/score", response_model=ChurnResponse, tags=["Prediction"])
def score_customer(customer: CustomerInput):
    """
    Score a single customer's churn risk.

    - **churn_probability**: 0.0 to 1.0
    - **risk_tier**: high / medium / low
    - **will_churn**: True if probability >= threshold
    - **recommendation**: suggested retention action
    """
    try:
        result = predict(customer.model_dump())
        result["recommendation"] = RECOMMENDATIONS[result["risk_tier"]]
        return ChurnResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info", tags=["System"])
def model_info():
    """Return model metadata — threshold, number of features."""
    return {
        "threshold"   : THRESHOLD,
        "n_features"  : len(FEATURES),
        "features"    : FEATURES,
    }