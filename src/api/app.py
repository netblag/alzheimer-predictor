"""
Alzheimer's Predictor — FastAPI Application
============================================
REST API + serves the interactive web dashboard.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import numpy as np, joblib, json, os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="🧠 Alzheimer's Risk Predictor API",
    description="""
## Alzheimer's Disease Risk Assessment API

Predicts dementia risk from MRI-derived and clinical features,
trained on the OASIS longitudinal dataset.

### ⚠️ Medical Disclaimer
For research / educational use only. Not a clinical diagnostic tool.
    """,
    version="1.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../../models/saved")
DASHBOARD  = os.path.join(os.path.dirname(__file__), "dashboard.html")

_model, _scaler, _metadata, _feature_names = None, None, None, None


def load_artifacts():
    global _model, _scaler, _metadata, _feature_names
    try:
        _model    = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
        _scaler   = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        with open(os.path.join(MODELS_DIR, "metadata.json")) as f:
            _metadata = json.load(f)
        _feature_names = _metadata["feature_names"]
        print(f"✅ Model loaded: {_metadata['model_name']}  AUC={_metadata.get('test_auc',0):.4f}")
    except Exception as e:
        print(f"⚠️  Model not loaded: {e}. Run main.py first.")


@app.on_event("startup")
async def startup(): load_artifacts()


# ── Schemas ───────────────────────────────────────────────────────────────────
class PatientData(BaseModel):
    Age:   float = Field(..., ge=18,  le=100, example=72)
    Sex:   int   = Field(..., ge=0,   le=1,   example=1,  description="1=Male, 0=Female")
    EDUC:  int   = Field(..., ge=1,   le=25,  example=14, description="Years of education")
    SES:   int   = Field(..., ge=1,   le=5,   example=2,  description="Socioeconomic status 1(high)–5(low)")
    MMSE:  float = Field(..., ge=0,   le=30,  example=27, description="Mini-Mental State Exam (0–30)")
    eTIV:  float = Field(..., ge=900, le=2200,example=1468)
    nWBV:  float = Field(..., ge=0.5, le=0.95,example=0.728)
    ASF:   float = Field(..., ge=0.8, le=1.6, example=1.20)

    class Config:
        schema_extra = {"example": {
            "Age":72,"Sex":1,"EDUC":14,"SES":2,"MMSE":27,
            "eTIV":1468,"nWBV":0.728,"ASF":1.20
        }}


class PredictionResponse(BaseModel):
    prediction:       int
    probability:      float
    risk_level:       str
    risk_description: str
    confidence:       str
    key_risk_factors: List[str]
    recommendations:  List[str]
    engineered_features: dict
    model_version:    str
    timestamp:        str


# ── Feature engineering (mirrors preprocess.py) ───────────────────────────────
def engineer(d: dict) -> dict:
    d = dict(d)
    d["MMSE_deficit"]       = 30 - d["MMSE"]
    d["brain_atrophy_rate"] = (1 - d["nWBV"]) / (d["Age"] / 100)
    d["cog_reserve"]        = d["EDUC"] * d["nWBV"]
    d["brain_vol_ratio"]    = d["nWBV"] * d["eTIV"] / 1500
    d["age_mmse_risk"]      = (d["Age"] / 100) * d["MMSE_deficit"]
    d["social_risk"]        = d["SES"] * (1 / (d["EDUC"] + 1))
    d["mmse_impaired"]      = int(d["MMSE"] < 24)
    d["elderly"]            = int(d["Age"] > 75)
    d["asf_deviation"]      = abs(d["ASF"] - 1.19)
    d["neuro_risk_score"]   = (
        d["MMSE_deficit"]       * 0.35 +
        d["brain_atrophy_rate"] * 2.0  +
        d["social_risk"]        * 0.5  +
        d["age_mmse_risk"]      * 3.0  +
        (5 - d["SES"])          * 0.2  +
        d["elderly"]            * 1.5
    )
    return d


def get_risk_factors(d: dict) -> List[str]:
    f = []
    if d["Age"] > 75:       f.append(f"Advanced age ({d['Age']:.0f} years)")
    if d["MMSE"] < 24:      f.append(f"Significant cognitive impairment (MMSE={d['MMSE']:.0f})")
    elif d["MMSE"] < 27:    f.append(f"Mild cognitive decline (MMSE={d['MMSE']:.0f})")
    if d["nWBV"] < 0.71:    f.append(f"Reduced brain volume (nWBV={d['nWBV']:.3f})")
    if d["SES"] >= 4:       f.append(f"High socioeconomic risk (SES={d['SES']})")
    if d["EDUC"] < 10:      f.append(f"Low education (cognitive reserve reduced)")
    if d["Sex"] == 1:       f.append("Male sex (higher structural atrophy rate)")
    if d["eTIV"] < 1300:    f.append(f"Small intracranial volume ({d['eTIV']:.0f} mm³)")
    if not f:               f.append("No major individual risk factors identified")
    return f


def get_recommendations(level: str) -> List[str]:
    if level == "HIGH":
        return [
            "🚨 Immediate neurological evaluation recommended",
            "Comprehensive neuropsychological battery testing",
            "MRI brain volumetry and PET amyloid scan",
            "Review medications for anticholinergic burden",
            "Enrol in cognitive stimulation programme",
            "Family counselling and caregiver support services",
        ]
    if level == "MODERATE":
        return [
            "⚠️ Neurology referral within 4 weeks",
            "Repeat cognitive assessment in 6 months",
            "Cardiovascular risk factor optimisation",
            "Mediterranean diet and regular aerobic exercise",
            "Social engagement to maintain cognitive reserve",
        ]
    return [
        "✅ Continue annual cognitive screening",
        "Maintain physically and mentally active lifestyle",
        "Heart-healthy diet rich in omega-3",
        "Manage blood pressure and cholesterol",
    ]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_dashboard():
    if os.path.exists(DASHBOARD):
        return HTMLResponse(open(DASHBOARD, encoding="utf-8").read())
    return HTMLResponse("<h2>Dashboard not found. Run main.py first.</h2>")


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": _model is not None,
            "model": _metadata.get("model_name","?") if _metadata else None,
            "timestamp": datetime.now().isoformat()}


@app.get("/model/info")
async def model_info():
    if not _metadata: raise HTTPException(503, "Model not loaded")
    return _metadata


@app.get("/features")
async def features():
    return {
        "input": {
            "Age":  "Age in years (18–100)",
            "Sex":  "Sex: 1=Male, 0=Female",
            "EDUC": "Years of education (1–25)",
            "SES":  "Socioeconomic status 1(high)–5(low)",
            "MMSE": "Mini-Mental State Exam (0–30, lower=worse)",
            "eTIV": "Estimated total intracranial volume (mm³)",
            "nWBV": "Normalized whole brain volume (0–1 ratio)",
            "ASF":  "Atlas scaling factor",
        },
        "engineered": [
            "MMSE_deficit","brain_atrophy_rate","cog_reserve","brain_vol_ratio",
            "age_mmse_risk","social_risk","mmse_impaired","elderly","asf_deviation",
            "neuro_risk_score",
        ],
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientData):
    if _model is None or _scaler is None:
        raise HTTPException(503, "Model not loaded. Run main.py first.")
    try:
        d  = engineer(patient.model_dump())
        X  = np.array([[d[f] for f in _feature_names]])
        Xs = _scaler.transform(X)

        prob = float(_model.predict_proba(Xs)[0][1])
        pred = int(prob >= 0.5)

        level = "HIGH" if prob >= 0.70 else "MODERATE" if prob >= 0.40 else "LOW"
        desc  = {
            "HIGH":     "High dementia risk. Immediate neurological evaluation recommended.",
            "MODERATE": "Moderate risk. Medical evaluation and monitoring advised.",
            "LOW":      "Low risk based on current parameters. Maintain healthy lifestyle.",
        }[level]

        margin = abs(prob - 0.5)
        conf   = "Very High" if margin >= 0.35 else "High" if margin >= 0.2 else "Moderate" if margin >= 0.1 else "Low"

        eng_feats = {k: round(float(d[k]), 4) for k in [
            "MMSE_deficit","brain_atrophy_rate","cog_reserve",
            "neuro_risk_score","age_mmse_risk","mmse_impaired","elderly"
        ]}

        return PredictionResponse(
            prediction=pred, probability=round(prob, 4),
            risk_level=level, risk_description=desc, confidence=conf,
            key_risk_factors=get_risk_factors(d),
            recommendations=get_recommendations(level),
            engineered_features=eng_feats,
            model_version=_metadata.get("model_name","1.0") if _metadata else "1.0",
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {e}")


@app.post("/predict/batch")
async def predict_batch(patients: List[PatientData]):
    if len(patients) > 100:
        raise HTTPException(400, "Max 100 patients per batch")
    if _model is None:
        raise HTTPException(503, "Model not loaded")

    out = []
    for i, p in enumerate(patients):
        try:
            d    = engineer(p.model_dump())
            X    = np.array([[d[f] for f in _feature_names]])
            prob = float(_model.predict_proba(_scaler.transform(X))[0][1])
            level = "HIGH" if prob >= 0.70 else "MODERATE" if prob >= 0.40 else "LOW"
            out.append({"index": i, "probability": round(prob,4),
                        "prediction": int(prob>=0.5), "risk_level": level})
        except Exception as e:
            out.append({"index": i, "error": str(e)})

    return {
        "total": len(patients),
        "summary": {
            "high":    sum(1 for r in out if r.get("risk_level")=="HIGH"),
            "moderate":sum(1 for r in out if r.get("risk_level")=="MODERATE"),
            "low":     sum(1 for r in out if r.get("risk_level")=="LOW"),
        },
        "results": out,
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
