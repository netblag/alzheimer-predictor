"""
Alzheimer's Predictor — Main Pipeline
======================================
Orchestrates: data → preprocess → train → visualize → SHAP → serve dashboard
 
Usage:
    python main.py              # Full pipeline + start API
    python main.py --no-tune    # Skip hyperparameter tuning (fast, ~30s)
    python main.py --no-serve   # Run pipeline only, don't start API
    python main.py --api-only   # Start API server only (needs trained model)
"""

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.preprocess      import run_pipeline
from src.models.train         import train_all_models, print_comparison
from src.models.explain       import run_explainability_pipeline
from src.visualization.plots  import run_visualization_pipeline
 
BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, "data/raw/oasis.csv")
PROC   = os.path.join(BASE, "data/processed")
MDL    = os.path.join(BASE, "models/saved")
FIGS   = os.path.join(BASE, "reports/figures")
DASH_SRC = os.path.join(BASE, "src/api/dashboard.html")
DASH_DST = os.path.join(BASE, "src/api/dashboard.html")   # same path, already there


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🧠  ALZHEIMER'S DISEASE RISK PREDICTOR                     ║
║       Machine Learning Pipeline v1.0                         ║
║                                                              ║
║   Dataset : OASIS Longitudinal MRI Study                     ║
║   Models  : LR | RF | XGBoost | LightGBM | SVM | Ensemble   ║
║   Analysis: SHAP Explainability + Neuro-Clinical Insights    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""


def run_full_pipeline(tune: bool = True):
    print(BANNER)
    t0 = time.time()

    # 1 ── Data
    datasets, df = run_pipeline(DATA, PROC)

    # 2 ── Train
    results, trained, best = train_all_models(datasets, models_dir=MDL, tune=tune)
    print_comparison(results)

    # 3 ── Visualize
    run_visualization_pipeline(df, results, datasets["y_test"], FIGS)

    # 4 ── SHAP
    run_explainability_pipeline(trained, datasets, best, FIGS)

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    m = results[best]["test_metrics"]
    print("\n" + "=" * 62)
    print("🎉  PIPELINE COMPLETE")
    print("=" * 62)
    print(f"  ⏱️  Time          : {elapsed:.1f}s")
    print(f"  🏆 Best model     : {best}")
    print(f"  📊 AUC            : {m['roc_auc']:.4f}")
    print(f"  📊 F1             : {m['f1']:.4f}")
    print(f"  📊 Recall         : {m['recall']:.4f}")
    print(f"  📊 Specificity    : {m['specificity']:.4f}")
    print(f"\n  📁 Models  → {MDL}/")
    print(f"  📁 Figures → {FIGS}/")
    print(f"\n  🚀 Start API:")
    print(f"     python main.py --api-only")
    print(f"     → http://localhost:8000          (dashboard)")
    print(f"     → http://localhost:8000/docs     (Swagger UI)")
    print("=" * 62)


def start_api():
    import uvicorn
    print("🚀 Starting API server on http://localhost:8000")
    print("   Dashboard → http://localhost:8000")
    print("   API docs  → http://localhost:8000/docs")
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=False)


def main():
    parser = argparse.ArgumentParser(description="Alzheimer's Predictor Pipeline")
    parser.add_argument("--no-tune",  action="store_true", help="Skip hyperparameter tuning")
    parser.add_argument("--no-serve", action="store_true", help="Run pipeline only, don't start API")
    parser.add_argument("--api-only", action="store_true", help="Start API server only")
    args = parser.parse_args()

    if args.api_only:
        start_api()
    elif args.no_serve:
        run_full_pipeline(tune=not args.no_tune)
    else:
        run_full_pipeline(tune=not args.no_tune)
        print("\n" + "─" * 62)
        start_api()


if __name__ == "__main__":
    main()
