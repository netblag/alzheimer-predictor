"""
Alzheimer's Predictor — Test Suite
====================================
Run: pytest tests/ -v --tb=short
"""

import pytest
import numpy as np
import pandas as pd
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocess import clean_data, engineer_features, prepare_datasets
from src.models.train    import compute_metrics


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def raw_df():
    np.random.seed(42)
    n = 120
    return pd.DataFrame({
        "Age":  np.random.uniform(60, 95, n),
        "Sex":  np.random.randint(0, 2, n),
        "EDUC": np.random.randint(6, 23, n),
        "SES":  np.random.randint(1, 6, n),
        "MMSE": np.random.uniform(10, 30, n),
        "eTIV": np.random.uniform(1100, 2000, n),
        "nWBV": np.random.uniform(0.64, 0.84, n),
        "ASF":  np.random.uniform(0.88, 1.59, n),
        "Group":np.random.randint(0, 3, n),
    })


@pytest.fixture
def engineered_df(raw_df):
    c = clean_data(raw_df)
    return engineer_features(c)


# ── Data Cleaning ─────────────────────────────────────────────────────────────
class TestDataCleaning:

    def test_no_nulls_after_clean(self, raw_df):
        assert clean_data(raw_df).isnull().sum().sum() == 0

    def test_removes_invalid_mmse(self):
        np.random.seed(0)
        n = 40
        df = pd.DataFrame({
            "Age": np.random.uniform(60,90,n), "Sex": np.ones(n,int),
            "EDUC": np.full(n,14), "SES": np.full(n,2),
            "MMSE": np.random.uniform(10,30,n),
            "eTIV": np.random.uniform(1200,1800,n),
            "nWBV": np.random.uniform(0.68,0.82,n),
            "ASF":  np.random.uniform(0.9,1.4,n),
            "Group": np.zeros(n,int),
        })
        df.loc[0, "MMSE"] = 35   # invalid
        df.loc[1, "MMSE"] = -1   # invalid
        cleaned = clean_data(df)
        assert (cleaned["MMSE"] >= 0).all()
        assert (cleaned["MMSE"] <= 30).all()

    def test_removes_invalid_nwbv(self, raw_df):
        df = raw_df.copy()
        df.loc[0, "nWBV"] = 0.0   # invalid
        df.loc[1, "nWBV"] = 1.5   # invalid
        cleaned = clean_data(df)
        assert (cleaned["nWBV"] > 0).all()
        assert (cleaned["nWBV"] < 1).all()

    def test_all_numeric_after_clean(self, raw_df):
        cleaned = clean_data(raw_df)
        for col in cleaned.columns:
            assert pd.api.types.is_numeric_dtype(cleaned[col]), f"{col} not numeric"

    def test_group_values_valid(self, raw_df):
        cleaned = clean_data(raw_df)
        assert set(cleaned["Group"].unique()).issubset({0, 1, 2})

    def test_removes_duplicates(self, raw_df):
        df_dup = pd.concat([raw_df, raw_df.iloc[:5]], ignore_index=True)
        cleaned = clean_data(df_dup)
        assert len(cleaned) <= len(df_dup)


# ── Feature Engineering ───────────────────────────────────────────────────────
class TestFeatureEngineering:

    def test_new_columns_added(self, raw_df):
        cleaned  = clean_data(raw_df)
        original = set(cleaned.columns)
        eng      = engineer_features(cleaned)
        new_cols = set(eng.columns) - original
        assert len(new_cols) >= 8

    def test_mmse_deficit_formula(self, engineered_df):
        expected = 30 - engineered_df["MMSE"]
        np.testing.assert_allclose(engineered_df["MMSE_deficit"], expected)

    def test_brain_atrophy_rate_positive(self, engineered_df):
        assert (engineered_df["brain_atrophy_rate"] > 0).all()

    def test_cog_reserve_formula(self, engineered_df):
        expected = engineered_df["EDUC"] * engineered_df["nWBV"]
        np.testing.assert_allclose(engineered_df["cog_reserve"], expected)

    def test_binary_features_are_binary(self, engineered_df):
        for col in ["mmse_impaired", "elderly"]:
            assert set(engineered_df[col].unique()).issubset({0, 1}), f"{col} not binary"

    def test_neuro_risk_score_non_negative(self, engineered_df):
        assert (engineered_df["neuro_risk_score"] >= 0).all()

    def test_original_features_unchanged(self, raw_df):
        cleaned  = clean_data(raw_df)
        orig_age = cleaned["Age"].copy()
        eng      = engineer_features(cleaned)
        pd.testing.assert_series_equal(eng["Age"].reset_index(drop=True),
                                       orig_age.reset_index(drop=True))

    def test_asf_deviation_non_negative(self, engineered_df):
        assert (engineered_df["asf_deviation"] >= 0).all()


# ── Dataset Preparation ───────────────────────────────────────────────────────
class TestDatasetPreparation:

    def test_no_target_in_features(self, engineered_df):
        ds = prepare_datasets(engineered_df, apply_smote=False)
        assert "target" not in ds["feature_names"]
        assert "Group"  not in ds["feature_names"]

    def test_scaler_fitted(self, engineered_df):
        ds = prepare_datasets(engineered_df, apply_smote=False)
        assert hasattr(ds["scaler"], "mean_")

    def test_feature_dim_consistent(self, engineered_df):
        ds = prepare_datasets(engineered_df, apply_smote=False)
        assert ds["X_train"].shape[1] == len(ds["feature_names"])
        assert ds["X_test"].shape[1]  == len(ds["feature_names"])

    def test_both_classes_in_test(self, engineered_df):
        ds = prepare_datasets(engineered_df, apply_smote=False)
        assert len(np.unique(ds["y_test"])) == 2

    def test_binary_flag_stored(self, engineered_df):
        ds = prepare_datasets(engineered_df, binary=True, apply_smote=False)
        assert ds["binary"] is True


# ── Metrics ───────────────────────────────────────────────────────────────────
class TestMetrics:

    def test_perfect_classifier(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
        m = compute_metrics(y_true, y_pred, y_prob)
        assert m["accuracy"]  == 1.0
        assert m["roc_auc"]   == 1.0
        assert m["f1"]        == 1.0
        assert m["recall"]    == 1.0

    def test_all_keys_present(self):
        y_true = np.array([0,0,1,1]); y_pred = np.array([0,1,1,0])
        y_prob = np.array([0.1,0.8,0.9,0.3])
        m = compute_metrics(y_true, y_pred, y_prob)
        for key in ["accuracy","precision","recall","f1","roc_auc","specificity","npv","tp","tn","fp","fn"]:
            assert key in m, f"Missing: {key}"

    def test_metrics_in_range(self):
        np.random.seed(99)
        y_t = np.random.randint(0,2,60); y_p = np.random.randint(0,2,60)
        y_b = np.random.random(60)
        m = compute_metrics(y_t, y_p, y_b)
        for k in ["accuracy","precision","recall","f1","roc_auc","specificity"]:
            assert 0.0 <= m[k] <= 1.0, f"{k}={m[k]} out of range"


# ── Model training (fast, no tuning) ─────────────────────────────────────────
class TestModelTraining:

    @pytest.fixture
    def small_ds(self, engineered_df):
        return prepare_datasets(engineered_df, apply_smote=False)

    def test_logistic_regression(self, small_ds):
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(max_iter=1000, random_state=42)
        m.fit(small_ds["X_train"], small_ds["y_train"])
        preds = m.predict(small_ds["X_test"])
        assert len(preds) == len(small_ds["y_test"])

    def test_random_forest_proba(self, small_ds):
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(n_estimators=10, random_state=42)
        m.fit(small_ds["X_train"], small_ds["y_train"])
        probs = m.predict_proba(small_ds["X_test"])
        assert probs.shape == (len(small_ds["y_test"]), 2)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_xgboost(self, small_ds):
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=10, random_state=42,
                               eval_metric="logloss", use_label_encoder=False)
        m.fit(small_ds["X_train"], small_ds["y_train"])
        probs = m.predict_proba(small_ds["X_test"])[:, 1]
        assert all(0 <= p <= 1 for p in probs)

    def test_auc_above_random(self, small_ds):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
        m = RandomForestClassifier(n_estimators=30, random_state=42)
        m.fit(small_ds["X_train"], small_ds["y_train"])
        probs = m.predict_proba(small_ds["X_test"])[:, 1]
        if len(np.unique(small_ds["y_test"])) > 1:
            auc = roc_auc_score(small_ds["y_test"], probs)
            assert auc > 0.5, f"AUC={auc:.3f} not above random"


# ── API validation ─────────────────────────────────────────────────────────────
class TestAPIValidation:

    def test_valid_patient(self):
        from src.api.app import PatientData
        p = PatientData(Age=72, Sex=1, EDUC=14, SES=2, MMSE=27,
                        eTIV=1468, nWBV=0.728, ASF=1.20)
        assert p.Age == 72

    def test_age_out_of_range(self):
        from src.api.app import PatientData
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            PatientData(Age=150, Sex=1, EDUC=14, SES=2, MMSE=27,
                        eTIV=1468, nWBV=0.728, ASF=1.20)

    def test_mmse_out_of_range(self):
        from src.api.app import PatientData
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            PatientData(Age=72, Sex=1, EDUC=14, SES=2, MMSE=35,
                        eTIV=1468, nWBV=0.728, ASF=1.20)

    def test_engineer_produces_all_features(self):
        from src.api.app import engineer
        d = dict(Age=72, Sex=1, EDUC=14, SES=2, MMSE=27,
                 eTIV=1468, nWBV=0.728, ASF=1.20)
        eng = engineer(d)
        for key in ["MMSE_deficit","brain_atrophy_rate","cog_reserve","brain_vol_ratio",
                    "age_mmse_risk","social_risk","mmse_impaired","elderly",
                    "asf_deviation","neuro_risk_score"]:
            assert key in eng, f"Missing: {key}"

    def test_high_risk_factors_populated(self):
        from src.api.app import get_risk_factors
        d = dict(Age=82, Sex=1, EDUC=7, SES=5, MMSE=18,
                 eTIV=1150, nWBV=0.67, ASF=1.3)
        factors = get_risk_factors(d)
        assert len(factors) >= 4

    def test_recommendations_not_empty(self):
        from src.api.app import get_recommendations
        for lvl in ["HIGH","MODERATE","LOW"]:
            assert len(get_recommendations(lvl)) >= 3


# ── End-to-end ────────────────────────────────────────────────────────────────
class TestEndToEnd:

    def test_mini_pipeline(self, raw_df):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score

        cleaned = clean_data(raw_df)
        eng     = engineer_features(cleaned)
        ds      = prepare_datasets(eng, apply_smote=False)

        m = RandomForestClassifier(n_estimators=20, random_state=42)
        m.fit(ds["X_train"], ds["y_train"])
        preds = m.predict(ds["X_test"])
        probs = m.predict_proba(ds["X_test"])[:, 1]

        assert len(preds) == len(ds["y_test"])
        assert all(p in [0,1] for p in preds)
        if len(np.unique(ds["y_test"])) > 1:
            auc = roc_auc_score(ds["y_test"], probs)
            assert 0 <= auc <= 1
