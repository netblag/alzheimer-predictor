"""
Alzheimer's Predictor — SHAP Explainability
============================================
Global feature importance + per-patient explanation plots.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings, os
warnings.filterwarnings("ignore")

PALETTE = {
    "primary":    "#7B2FBE",
    "secondary":  "#1A1A2E",
    "accent":     "#4CC9F0",
    "danger":     "#E63946",
    "success":    "#22d3a0",
    "warn":       "#F59E0B",
    "bg":         "#F8F6FF",
    "neutral":    "#2D3748",
}


def compute_shap_values(model, X_df: pd.DataFrame, feature_names: list, model_name: str = ""):
    print(f"  🔬 SHAP for {model_name}...", end="", flush=True)
    mtype = type(model).__name__.lower()
    try:
        if any(k in mtype for k in ["xgb", "lgbm", "lightgbm", "randomforest", "gradientboosting"]):
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_df)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            bg = shap.kmeans(X_df, min(50, len(X_df)))
            explainer   = shap.KernelExplainer(model.predict_proba, bg)
            shap_values = explainer.shap_values(X_df.iloc[:100], nsamples=100)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

        sv = np.array(shap_values)
        if sv.ndim == 3: sv = sv[1]
        print(" done ✓")
        return sv, explainer
    except Exception as e:
        print(f"\n  ⚠️  SHAP failed: {e}")
        return None, None


def plot_global_importance(shap_values, feature_names: list, output_path: str, top_n: int = 15):
    sv = np.array(shap_values)
    if sv.ndim == 3: sv = sv[1]

    n_feats    = min(sv.shape[1], len(feature_names))
    mean_shap  = np.abs(sv[:, :n_feats]).mean(axis=0)
    feat_names = list(feature_names[:n_feats])

    imp_df = (
        pd.DataFrame({"feature": feat_names, "importance": mean_shap})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    colors = [PALETTE["primary"] if i < 5 else PALETTE["accent"] if i < 10 else PALETTE["neutral"]
              for i in range(len(imp_df))]

    bars = ax.barh(range(len(imp_df)), imp_df["importance"].values,
                   color=colors, alpha=0.88, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(imp_df)))
    ax.set_yticklabels(imp_df["feature"].values, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12, color=PALETTE["neutral"])
    ax.set_title("🧠 Global Feature Importance (SHAP)", fontsize=16,
                 fontweight="bold", color=PALETTE["secondary"], pad=15)

    for i, (bar, val) in enumerate(zip(bars, imp_df["importance"].values)):
        ax.text(val + 0.001, i, f"{val:.4f}", va="center", fontsize=9, color=PALETTE["neutral"])

    patches = [
        mpatches.Patch(color=PALETTE["primary"], label="Top 5"),
        mpatches.Patch(color=PALETTE["accent"],  label="Top 6–10"),
        mpatches.Patch(color=PALETTE["neutral"], label="Top 11–15"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  💾 Saved: {output_path}")
    return imp_df


def plot_shap_summary(shap_values, X_df: pd.DataFrame, feature_names: list, output_path: str):
    sv = np.array(shap_values)
    if sv.ndim == 3: sv = sv[1]

    n = min(sv.shape[0], len(X_df))
    k = min(sv.shape[1] if sv.ndim > 1 else sv.shape[0], len(feature_names))
    sv      = sv[:n, :k]
    X_plot  = X_df.iloc[:n, :k].copy()
    X_plot.columns = list(feature_names[:k])

    fig, ax = plt.subplots(figsize=(12, 9))
    shap.summary_plot(sv, X_plot, plot_type="dot", max_display=15, show=False)
    plt.title("SHAP Summary — Feature Impact on Alzheimer's Prediction",
              fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 Saved: {output_path}")


def run_explainability_pipeline(trained_models: dict, datasets: dict,
                                best_model_name: str, output_dir: str):
    print("\n" + "=" * 60)
    print("🔬 SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    model        = trained_models[best_model_name]
    feature_names = datasets["feature_names"]
    X_test_df    = pd.DataFrame(datasets["X_test"], columns=feature_names)

    sv, _ = compute_shap_values(model, X_test_df, feature_names, best_model_name)
    if sv is None:
        print("  ⚠️  Skipping SHAP plots.")
        return None, None

    imp_df = plot_global_importance(sv, feature_names,
                                    os.path.join(output_dir, "shap_importance.png"))
    plot_shap_summary(sv, X_test_df, feature_names,
                      os.path.join(output_dir, "shap_summary.png"))

    print(f"\n  🏆 Top 5 Features:")
    for i, row in imp_df.head(5).iterrows():
        print(f"     {i+1}. {row['feature']:<35} SHAP: {row['importance']:.4f}")

    return sv, imp_df
