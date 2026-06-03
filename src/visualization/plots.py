"""
Alzheimer's Predictor — Visualization Suite
============================================
EDA dashboard, ROC curves, confusion matrices.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import warnings, os
warnings.filterwarnings("ignore")

PALETTE = {
    "purple":  "#7B2FBE", "dark":    "#1A1A2E",
    "cyan":    "#4CC9F0", "bg":      "#F8F6FF",
    "red":     "#E63946", "green":   "#22d3a0",
    "orange":  "#F59E0B", "neutral": "#2D3748",
    "lavender":"#A78BFA", "pink":    "#F472B6",
}

MODEL_COLORS = [
    PALETTE["purple"], PALETTE["cyan"], PALETTE["red"],
    PALETTE["orange"], PALETTE["green"], PALETTE["lavender"], PALETTE["pink"],
]


def plot_eda_overview(df: pd.DataFrame, output_path: str):
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor(PALETTE["bg"])
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35)
    fig.suptitle("🧠 Alzheimer's Dataset — Exploratory Analysis",
                 fontsize=20, fontweight="bold", color=PALETTE["dark"], y=0.98)

    group_labels = {0: "Nondemented", 1: "Demented", 2: "Converted"}
    group_colors = {0: PALETTE["green"], 1: PALETTE["red"], 2: PALETTE["orange"]}

    # 1. Group distribution
    ax1 = fig.add_subplot(gs[0, 0])
    counts = df["Group"].value_counts().sort_index()
    ax1.pie(counts.values,
            labels=[group_labels[i] for i in counts.index],
            colors=[group_colors[i] for i in counts.index],
            autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10})
    ax1.set_title("Diagnosis Distribution", fontsize=12, fontweight="bold", color=PALETTE["dark"])

    # 2. Age distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(PALETTE["bg"])
    for g, col, lbl in [(0, PALETTE["green"], "Nondemented"),
                         (1, PALETTE["red"],   "Demented"),
                         (2, PALETTE["orange"],"Converted")]:
        ax2.hist(df[df.Group==g]["Age"], bins=15, color=col, alpha=0.65, label=lbl, edgecolor="white")
    ax2.set_xlabel("Age", fontsize=10); ax2.set_ylabel("Count", fontsize=10)
    ax2.set_title("Age Distribution by Diagnosis", fontsize=12, fontweight="bold", color=PALETTE["dark"])
    ax2.legend(fontsize=9)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    # 3. MMSE distribution
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(PALETTE["bg"])
    for g, col, lbl in [(0, PALETTE["green"],"Nondemented"),
                         (1, PALETTE["red"],  "Demented"),
                         (2, PALETTE["orange"],"Converted")]:
        ax3.hist(df[df.Group==g]["MMSE"], bins=15, color=col, alpha=0.65, label=lbl, edgecolor="white")
    ax3.axvline(24, color=PALETTE["dark"], linestyle="--", linewidth=2, label="Impairment threshold (24)")
    ax3.set_xlabel("MMSE Score", fontsize=10); ax3.set_ylabel("Count", fontsize=10)
    ax3.set_title("MMSE Score Distribution", fontsize=12, fontweight="bold", color=PALETTE["dark"])
    ax3.legend(fontsize=8)
    ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)

    # 4. nWBV vs Age scatter
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor(PALETTE["bg"])
    for g, col, lbl in [(0, PALETTE["green"],"Nondemented"),
                         (1, PALETTE["red"],  "Demented"),
                         (2, PALETTE["orange"],"Converted")]:
        sub = df[df.Group==g]
        ax4.scatter(sub["Age"], sub["nWBV"], c=col, alpha=0.5, s=18, label=lbl, edgecolors="none")
    ax4.set_xlabel("Age", fontsize=10); ax4.set_ylabel("nWBV", fontsize=10)
    ax4.set_title("Brain Volume vs Age", fontsize=12, fontweight="bold", color=PALETTE["dark"])
    ax4.legend(fontsize=8)
    ax4.spines["top"].set_visible(False); ax4.spines["right"].set_visible(False)

    # 5. Correlation heatmap
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.set_facecolor(PALETTE["bg"])
    cols_corr = ["Age","MMSE","nWBV","eTIV","EDUC","SES","neuro_risk_score","Group"]
    avail = [c for c in cols_corr if c in df.columns]
    corr  = df[avail].corr()
    cmap  = LinearSegmentedColormap.from_list("neuro", [PALETTE["cyan"],"white",PALETTE["purple"]], N=256)
    im    = ax5.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax5.set_xticks(range(len(avail))); ax5.set_yticks(range(len(avail)))
    ax5.set_xticklabels(avail, rotation=45, ha="right", fontsize=9)
    ax5.set_yticklabels(avail, fontsize=9)
    for i in range(len(avail)):
        for j in range(len(avail)):
            ax5.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                     fontsize=8, color="white" if abs(corr.values[i,j]) > 0.5 else "black")
    ax5.set_title("Feature Correlation Matrix", fontsize=12, fontweight="bold", color=PALETTE["dark"])
    plt.colorbar(im, ax=ax5, shrink=0.8)

    # 6. Education by group
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.set_facecolor(PALETTE["bg"])
    for g, col, lbl in [(0, PALETTE["green"],"Nondemented"),
                         (1, PALETTE["red"],  "Demented"),
                         (2, PALETTE["orange"],"Converted")]:
        ax6.hist(df[df.Group==g]["EDUC"], bins=12, color=col, alpha=0.65, label=lbl, edgecolor="white")
    ax6.set_xlabel("Years of Education", fontsize=10)
    ax6.set_title("Education Distribution", fontsize=12, fontweight="bold", color=PALETTE["dark"])
    ax6.legend(fontsize=8)
    ax6.spines["top"].set_visible(False); ax6.spines["right"].set_visible(False)

    # 7. Neuro risk score
    ax7 = fig.add_subplot(gs[2, 1])
    ax7.set_facecolor(PALETTE["bg"])
    if "neuro_risk_score" in df.columns:
        for g, col, lbl in [(0, PALETTE["green"],"Nondemented"),
                             (1, PALETTE["red"],  "Demented"),
                             (2, PALETTE["orange"],"Converted")]:
            ax7.hist(df[df.Group==g]["neuro_risk_score"], bins=15, color=col, alpha=0.65,
                     label=lbl, edgecolor="white")
        ax7.set_xlabel("Neuro Risk Score", fontsize=10)
        ax7.set_title("Composite Risk Score", fontsize=12, fontweight="bold", color=PALETTE["dark"])
        ax7.legend(fontsize=8)
        ax7.spines["top"].set_visible(False); ax7.spines["right"].set_visible(False)

    # 8. Sex breakdown
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.set_facecolor(PALETTE["bg"])
    sex_grp = df.groupby(["Sex","Group"]).size().unstack(fill_value=0)
    x = np.arange(2)
    for i, (g, col) in enumerate(zip([0,1,2],[PALETTE["green"],PALETTE["red"],PALETTE["orange"]])):
        vals = sex_grp.get(g, pd.Series([0,0])).values[:2] if g in sex_grp.columns else [0,0]
        ax8.bar(x + (i-1)*0.27, vals, 0.26, color=col, alpha=0.8, label=group_labels[g])
    ax8.set_xticks(x); ax8.set_xticklabels(["Female","Male"], fontsize=10)
    ax8.set_ylabel("Count", fontsize=10)
    ax8.set_title("Diagnosis by Sex", fontsize=12, fontweight="bold", color=PALETTE["dark"])
    ax8.legend(fontsize=8)
    ax8.spines["top"].set_visible(False); ax8.spines["right"].set_visible(False)

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  💾 Saved: {output_path}")


def plot_roc_and_metrics(results: dict, output_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle("Model Performance Comparison", fontsize=18, fontweight="bold",
                 color=PALETTE["dark"], y=1.01)

    ax = axes[0]
    ax.set_facecolor(PALETTE["bg"])
    ax.plot([0,1],[0,1],"k--",alpha=0.4,linewidth=1,label="Random")
    for i,(name,res) in enumerate(results.items()):
        fpr = res["roc_curve"]["fpr"]; tpr = res["roc_curve"]["tpr"]
        auc = res["test_metrics"]["roc_auc"]
        ax.plot(fpr, tpr, linewidth=2.5, color=MODEL_COLORS[i % len(MODEL_COLORS)],
                label=f"{name} (AUC={auc:.3f})", alpha=0.85)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves", fontsize=14, fontweight="bold", color=PALETTE["dark"])
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(True, alpha=0.3); ax.set_xlim([-0.01,1.01]); ax.set_ylim([-0.01,1.01])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax2 = axes[1]
    ax2.set_facecolor(PALETTE["bg"])
    metrics_list = ["accuracy","precision","recall","f1","roc_auc"]
    names  = list(results.keys())
    x      = np.arange(len(metrics_list))
    width  = 0.8 / len(names)
    for i,(name,res) in enumerate(results.items()):
        m = res["test_metrics"]
        vals   = [m[k] for k in metrics_list]
        offset = (i - len(names)/2) * width + width/2
        ax2.bar(x + offset, vals, width*0.9,
                color=MODEL_COLORS[i % len(MODEL_COLORS)], alpha=0.85, label=name)
    ax2.set_xticks(x); ax2.set_xticklabels([m.upper() for m in metrics_list], fontsize=10)
    ax2.set_ylim([0,1.15]); ax2.set_ylabel("Score", fontsize=12)
    ax2.set_title("All Metrics Comparison", fontsize=14, fontweight="bold", color=PALETTE["dark"])
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(axis="y", alpha=0.3)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  💾 Saved: {output_path}")


def plot_confusion_matrices(results: dict, y_test, output_path: str):
    n      = len(results)
    ncols  = 3
    nrows  = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle("Confusion Matrices — All Models", fontsize=18,
                 fontweight="bold", color=PALETTE["dark"], y=1.01)

    axes = axes.flatten() if n > 1 else [axes]
    cmap = LinearSegmentedColormap.from_list("neuro_cm", ["white", PALETTE["dark"]], N=256)

    for i,(name,res) in enumerate(results.items()):
        ax = axes[i]; ax.set_facecolor(PALETTE["bg"])
        cm = np.array([[res["test_metrics"]["tn"], res["test_metrics"]["fp"]],
                        [res["test_metrics"]["fn"], res["test_metrics"]["tp"]]])
        im = ax.imshow(cm, cmap=cmap, interpolation="nearest")
        thresh = cm.max() / 2
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r,c]), ha="center", va="center",
                        fontsize=16, fontweight="bold",
                        color="white" if cm[r,c] > thresh else PALETTE["dark"])
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Healthy","Dementia"], fontsize=10)
        ax.set_yticklabels(["Healthy","Dementia"], fontsize=10, rotation=90, va="center")
        ax.set_xlabel("Predicted", fontsize=10); ax.set_ylabel("Actual", fontsize=10)
        auc = res["test_metrics"]["roc_auc"]
        ax.set_title(f"{name}\nAUC: {auc:.3f}", fontsize=11, fontweight="bold", color=PALETTE["dark"])

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  💾 Saved: {output_path}")


def run_visualization_pipeline(df: pd.DataFrame, results: dict, y_test, output_dir: str):
    print("\n" + "=" * 60)
    print("📊 GENERATING VISUALIZATIONS")
    print("=" * 60)
    os.makedirs(output_dir, exist_ok=True)

    print("  📈 EDA overview...")
    plot_eda_overview(df, os.path.join(output_dir, "eda_overview.png"))

    print("  📈 ROC & metrics...")
    plot_roc_and_metrics(results, os.path.join(output_dir, "roc_comparison.png"))

    print("  📈 Confusion matrices...")
    plot_confusion_matrices(results, y_test, os.path.join(output_dir, "confusion_matrices.png"))

    print(f"\n  ✅ All plots saved → {output_dir}/")
