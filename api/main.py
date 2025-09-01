from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd, joblib, os

ART = os.environ.get("MODEL_PATH", "model_artifacts/model.joblib")
bundle = joblib.load(ART)  # { "model": Pipeline, "required_features": [...] }
clf = bundle["model"]
REQUIRED: List[str] = list(bundle["required_features"])

class FeaturePayload(BaseModel):
    features: Dict[str, float] = Field(..., description="Map of feature_name -> value")

app = FastAPI(title="PhishDetector API", version="1.0.0")

@app.get("/healthz")
def healthz():
    return {"ok": True, "required_features": REQUIRED[:5] + (["..."] if len(REQUIRED) > 5 else [])}

@app.post("/predict")
def predict(payload: FeaturePayload):
    missing = [c for c in REQUIRED if c not in payload.features]
    if missing:
        raise HTTPException(status_code=400, detail={"missing_features": missing})
    X = pd.DataFrame([[payload.features[c] for c in REQUIRED]], columns=REQUIRED)
    proba = float(clf.predict_proba(X)[0, 1])
    return {"is_phishing": int(proba >= 0.5), "prob": proba}
