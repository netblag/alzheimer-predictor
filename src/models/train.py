"""
Alzheimer's Predictor — Model Training & Evaluation
=====================================================
Trains 5 models + ensemble with hyperparameter tuning,
cross-validation, and comprehensive metric reporting.
"""

import numpy as np
import pandas as pd
import joblib, json, os
from datetime import datetime
from typing import Dict, Tuple, Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, average_precision_score
)
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")


MODEL_CONFIGS = {
    "Logistic Regression": {
        "model": LogisticRegression(random_state=42, max_iter=2000),
        "params": {
            "C": [0.01, 0.1, 1, 10, 100],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear", "saga"],
        },
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42, n_jobs=-1),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
            "max_features": ["sqrt", "log2"],
        },
    },
    "XGBoost": {
        "model": xgb.XGBClassifier(
            random_state=42, eval_metric="logloss",
            use_label_encoder=False, n_jobs=-1
        ),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        },
    },
    "LightGBM": {
        "model": lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7, -1],
            "learning_rate": [0.01, 0.1, 0.2],
            "num_leaves": [31, 63, 127],
            "subsample": [0.8, 1.0],
        },
    },
    "SVM": {
        "model": SVC(probability=True, random_state=42),
        "params": {
            "C": [0.1, 1, 10, 100],
            "kernel": ["rbf", "linear", "poly"],
            "gamma": ["scale", "auto"],
        },
    },
}


def compute_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy":      accuracy_score(y_true, y_pred),
        "precision":     precision_score(y_true, y_pred, zero_division=0),
        "recall":        recall_score(y_true, y_pred, zero_division=0),
        "f1":            f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":       roc_auc_score(y_true, y_prob),
        "avg_precision": average_precision_score(y_true, y_prob),
        "specificity":   tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "npv":           tn / (tn + fn) if (tn + fn) > 0 else 0.0,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def cross_validate_model(model, X_train, y_train, cv: int = 5) -> Dict:
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    cv_res = cross_validate(
        model, X_train, y_train, cv=skf,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
        return_train_score=True, n_jobs=-1,
    )
    return {
        m: {
            "mean": cv_res[f"test_{m}"].mean(),
            "std":  cv_res[f"test_{m}"].std(),
            "values": cv_res[f"test_{m}"].tolist(),
        }
        for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]
    }


def tune_model(name: str, config: dict, X_train, y_train, n_iter: int = 20):
    print(f"  🔧 Tuning {name}...", end="", flush=True)
    search = RandomizedSearchCV(
        config["model"], config["params"],
        n_iter=n_iter, cv=5, scoring="roc_auc",
        random_state=42, n_jobs=-1, verbose=0,
    )
    search.fit(X_train, y_train)
    print(f"  best AUC={search.best_score_:.4f}")
    return search.best_estimator_, search.best_params_


def train_all_models(datasets: dict, models_dir: str = "models/saved", tune: bool = True):
    print("\n" + "=" * 60)
    print("🧠 MODEL TRAINING & EVALUATION")
    print("=" * 60)

    X_train, X_val, X_test = datasets["X_train"], datasets["X_val"], datasets["X_test"]
    y_train, y_val, y_test = datasets["y_train"], datasets["y_val"], datasets["y_test"]

    os.makedirs(models_dir, exist_ok=True)
    results, trained = {}, {}

    for name, cfg in MODEL_CONFIGS.items():
        print(f"\n📌 {name}")
        if tune:
            model, best_params = tune_model(name, cfg, X_train, y_train)
        else:
            model = cfg["model"]
            model.fit(X_train, y_train)
            best_params = {}

        print(f"  📊 Cross-validating...", end="", flush=True)
        cv_scores = cross_validate_model(model, X_train, y_train)
        print(f"  CV AUC: {cv_scores['roc_auc']['mean']:.4f} ± {cv_scores['roc_auc']['std']:.4f}")

        y_val_pred = model.predict(X_val)
        y_val_prob = model.predict_proba(X_val)[:, 1]

        y_test_pred = model.predict(X_test)
        y_test_prob = model.predict_proba(X_test)[:, 1]

        val_metrics  = compute_metrics(y_val,  y_val_pred,  y_val_prob)
        test_metrics = compute_metrics(y_test, y_test_pred, y_test_prob)

        fpr, tpr, thr = roc_curve(y_test, y_test_prob)

        results[name] = {
            "val_metrics":  val_metrics,
            "test_metrics": test_metrics,
            "cv_scores":    cv_scores,
            "best_params":  best_params,
            "roc_curve":    {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thr.tolist()},
            "y_test_prob":  y_test_prob.tolist(),
            "y_test_pred":  y_test_pred.tolist(),
        }
        trained[name] = model
        joblib.dump(model, os.path.join(models_dir, f"{name.lower().replace(' ','_')}.pkl"))
        print(f"  ✅ Test  AUC={test_metrics['roc_auc']:.4f} | F1={test_metrics['f1']:.4f} | Recall={test_metrics['recall']:.4f}")

    # ── Ensemble (Top 3) ─────────────────────────────────────────────────────
    print(f"\n📌 Voting Ensemble (Top 3)...")
    top3 = sorted(results, key=lambda x: results[x]["test_metrics"]["roc_auc"], reverse=True)[:3]
    ensemble = VotingClassifier(
        estimators=[(n, trained[n]) for n in top3], voting="soft", n_jobs=-1
    )
    ensemble.fit(X_train, y_train)

    y_ens_pred = ensemble.predict(X_test)
    y_ens_prob = ensemble.predict_proba(X_test)[:, 1]
    ens_metrics = compute_metrics(y_test, y_ens_pred, y_ens_prob)
    fpr_e, tpr_e, thr_e = roc_curve(y_test, y_ens_prob)

    results["Ensemble (Top 3)"] = {
        "val_metrics":  ens_metrics,
        "test_metrics": ens_metrics,
        "cv_scores":    {},
        "best_params":  {"components": top3},
        "roc_curve":    {"fpr": fpr_e.tolist(), "tpr": tpr_e.tolist(), "thresholds": thr_e.tolist()},
        "y_test_prob":  y_ens_prob.tolist(),
        "y_test_pred":  y_ens_pred.tolist(),
    }
    trained["Ensemble (Top 3)"] = ensemble
    joblib.dump(ensemble, os.path.join(models_dir, "ensemble.pkl"))
    print(f"  ✅ Ensemble AUC={ens_metrics['roc_auc']:.4f} | F1={ens_metrics['f1']:.4f}")

    # ── Best model ───────────────────────────────────────────────────────────
    best_name = max(results, key=lambda x: results[x]["test_metrics"]["roc_auc"])
    print(f"\n🏆 Best: {best_name}  AUC={results[best_name]['test_metrics']['roc_auc']:.4f}")

    # ── Save artefacts ───────────────────────────────────────────────────────
    def _convert(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    with open(os.path.join(models_dir, "results.json"), "w") as f:
        json.dump(results, f, default=_convert, indent=2)

    joblib.dump(trained[best_name], os.path.join(models_dir, "best_model.pkl"))
    joblib.dump(datasets["scaler"],  os.path.join(models_dir, "scaler.pkl"))

    meta = {
        "best_model":    best_name,
        "model_name":    best_name,
        "feature_names": datasets["feature_names"],
        "training_date": datetime.now().isoformat(),
        "test_auc":      results[best_name]["test_metrics"]["roc_auc"],
        "test_f1":       results[best_name]["test_metrics"]["f1"],
        "test_recall":   results[best_name]["test_metrics"]["recall"],
    }
    with open(os.path.join(models_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return results, trained, best_name


def print_comparison(results: Dict):
    print("\n" + "=" * 80)
    print("📊 MODEL COMPARISON")
    print("=" * 80)
    hdr = f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}"
    print(hdr)
    print("-" * 80)
    for name, res in sorted(results.items(), key=lambda x: x[1]["test_metrics"]["roc_auc"], reverse=True):
        m = res["test_metrics"]
        print(f"{name:<25} {m['accuracy']:>10.4f} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['roc_auc']:>10.4f}")
    print("=" * 80)
