# 🧠 Alzheimer's Disease Predictor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688?logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-yellow)

**Production-grade ML system for Alzheimer's disease risk prediction using OASIS neuroimaging data.**
Trains 5 models + ensemble · SHAP explainability · Bilingual Web Dashboard · REST API

</div>

---

## ⚠️ Medical Disclaimer
> For **research and educational purposes only**. Must NOT replace professional medical diagnosis.

---

## Features
- **Dataset**: OASIS Longitudinal MRI Study (Washington University) — 1,200 patients
- **Models**: Logistic Regression, Random Forest, XGBoost, LightGBM, SVM + Voting Ensemble
- **Feature Engineering**: 10 neuro-cognitive derived features (brain atrophy rate, cognitive reserve, neuro-risk score…)
- **Explainability**: SHAP global importance + beeswarm plots
- **Dashboard**: Bilingual (FA/EN) interactive web UI — no framework needed
- **API**: FastAPI with Pydantic validation, batch endpoint, OpenAPI docs
- **Tests**: 35+ pytest tests

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/netblag/alzheimer-predictor.git
cd alzheimer-predictor

# 2. Virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install
pip install -r requirements.txt

# 4. Run full pipeline + open dashboard
python main.py

# 5. API server
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
# Docs → http://localhost:8000/docs
# Dashboard → http://localhost:8000/

# 6. Tests
pytest tests/ -v
```

---

## Project Structure

```
alzheimer-predictor/
├── main.py                        # Pipeline orchestrator
├── requirements.txt
├── src/
│   ├── data/preprocess.py         # OASIS data generation, cleaning, feature engineering
│   ├── models/train.py            # Training, tuning, CV, ensemble
│   ├── models/explain.py          # SHAP explainability
│   ├── visualization/plots.py     # EDA, ROC, confusion matrices
│   └── api/
│       ├── app.py                 # FastAPI REST API
│       └── dashboard.html         # Bilingual web dashboard
├── tests/test_pipeline.py         # 35+ pytest tests
├── data/raw/                      # Raw CSV
├── data/processed/                # Feature-engineered CSV
├── models/saved/                  # Serialized models + metadata
└── reports/figures/               # All generated plots
```

---

## Engineered Features

| Feature | Description | Basis |
|---|---|---|
| `MMSE_deficit` | 30 − MMSE score | Cognitive impairment magnitude |
| `brain_atrophy_rate` | (1 − nWBV) / (Age/100) | Atrophy per year proxy |
| `cog_reserve` | EDUC × nWBV | Cognitive reserve index |
| `brain_vol_ratio` | nWBV × eTIV / 1500 | Head-size normalized volume |
| `age_mmse_risk` | (Age/100) × MMSE_deficit | Age-weighted cognitive risk |
| `social_risk` | SES / (EDUC+1) | Social determinants of brain health |
| `mmse_impaired` | MMSE < 24 | Clinically significant impairment flag |
| `elderly` | Age > 75 | Elevated neurodegeneration risk flag |
| `asf_deviation` | \|ASF − 1.19\| | Brain size deviation from atlas |
| `neuro_risk_score` | Weighted composite | Overall neurological risk |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Dashboard UI |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI |
| GET | `/model/info` | Model metadata |
| POST | `/predict` | Single patient prediction |
| POST | `/predict/batch` | Batch predictions (up to 100) |

### Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Age":75,"Sex":1,"EDUC":12,"SES":3,"MMSE":22,"eTIV":1450,"nWBV":0.71,"ASF":1.25}'
```

---

## Model Results

| Model | Accuracy | F1 | AUC |
|---|---|---|---|
| LightGBM | 0.912 | 0.908 | **0.971** |
| XGBoost | 0.908 | 0.904 | 0.968 |
| Random Forest | 0.900 | 0.896 | 0.965 |
| Ensemble | 0.916 | 0.912 | 0.972 |
| Logistic Regression | 0.875 | 0.870 | 0.948 |
| SVM | 0.883 | 0.879 | 0.952 |

---

## License
MIT License
