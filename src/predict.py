import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.pkl"
META_PATH  = BASE_DIR / "models" / "threshold.json"
SCALER_PATH= BASE_DIR / "models" / "scaler.pkl"

# Load once at import time — not on every request
model     = joblib.load(MODEL_PATH)
scaler    = joblib.load(SCALER_PATH)

with open(META_PATH) as f:
    meta = json.load(f)

THRESHOLD  = meta["threshold"]
FEATURES   = meta["features"]


def get_risk_tier(probability: float) -> str:
    if probability >= 0.65:
        return "high"
    elif probability >= 0.35:
        return "medium"
    else:
        return "low"


def preprocess(data: dict) -> pd.DataFrame:
    """Convert raw customer dict into model-ready DataFrame."""
    df = pd.DataFrame([data])

    # Ensure all expected feature columns are present
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0  # fill missing with 0

    # Keep only the columns the model was trained on, in correct order
    df = df[FEATURES]
    return df


def predict(customer_data: dict) -> dict:
    """
    Takes a customer dict, returns churn probability + risk tier.
    """
    df        = preprocess(customer_data)
    prob      = float(model.predict_proba(df)[0][1])
    risk_tier = get_risk_tier(prob)
    churns    = prob >= THRESHOLD

    return {
        "churn_probability" : round(prob, 4),
        "risk_tier"         : risk_tier,
        "will_churn"        : bool(churns),
        "threshold_used"    : THRESHOLD,
    }