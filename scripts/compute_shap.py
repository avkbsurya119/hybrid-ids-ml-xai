# scripts/compute_shap.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt

from preprocessor.helpers import load_config


# --------------------------------------------------------
# LOAD MODEL + ENCODER
# --------------------------------------------------------
def load_lightgbm_model():
    model_path = ROOT / "models/artifacts/lightgbm_model.txt"
    encoder_path = ROOT / "models/artifacts/label_encoder.joblib"

    booster = lgb.Booster(model_file=str(model_path))
    le = joblib.load(encoder_path)
    return booster, le


# --------------------------------------------------------
# LOAD TRANSFORMED DATA
# --------------------------------------------------------
def load_transformed():
    cfg = load_config(ROOT / "preprocessor" / "config.yaml")
    processed_dir = ROOT / cfg["processed_dir"]

    parts = sorted(processed_dir.glob("transformed_part*.parquet"))
    dfs = [pd.read_parquet(p) for p in parts]
    full = pd.concat(dfs, ignore_index=True)
    return full


# --------------------------------------------------------
# MAIN SHAP PIPELINE (FINAL PRODUCTION VERSION)
# --------------------------------------------------------
def compute_shap():
    model, label_encoder = load_lightgbm_model()

    # Feature names used by LightGBM
    model_features = model.feature_name()
    print("Model uses", len(model_features), "features.")

    df = load_transformed()
    print("Loaded transformed data:", df.shape)

    # ----------------------------------------------------
    # Force DataFrame columns to match LGBM feature names
    # ----------------------------------------------------
    if "Label" in df.columns:
        df = df.drop(columns=["Label"])

    # Normalize column names: replace spaces with underscores
    df.columns = [c.replace(" ", "_") for c in df.columns]

    # Select ONLY the features the model expects
    X = df[model_features]
    print("Final DF shape after alignment:", X.shape)

    # ----------------------------------------------------
    # Background sample
    # ----------------------------------------------------
    background = X.sample(n=5000, random_state=42)

    bg_path = ROOT / "models/artifacts/shap_background_sample.parquet"
    background.to_parquet(bg_path, index=False)
    print("Saved background sample →", bg_path)

    # ----------------------------------------------------
    # SHAP Explainer (TreeExplainer)
    # ----------------------------------------------------
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(background)

    # Save explainer
    expl_path = ROOT / "models/artifacts/shap_explainer.pkl"
    joblib.dump(explainer, expl_path)
    print("Saved explainer →", expl_path)

    docs_dir = ROOT / "docs"
    docs_dir.mkdir(exist_ok=True)

    # ----------------------------------------------------
    # SHAP Summary Plot (Corrected)
    # ----------------------------------------------------
    plt.figure()
    shap.summary_plot(
        shap_values.values,
        background,
        feature_names=model_features,
        show=False
    )
    summary_path = docs_dir / "shap_summary_plot.png"
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", summary_path)

    # ----------------------------------------------------
    # SHAP Bar Plot (Corrected)
    # ----------------------------------------------------
    plt.figure()
    shap.summary_plot(
        shap_values.values,
        background,
        feature_names=model_features,
        plot_type="bar",
        show=False
    )
    bar_path = docs_dir / "shap_bar_plot.png"
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", bar_path)

    # ----------------------------------------------------
    # GLOBAL FEATURE IMPORTANCE (Corrected)
    # ----------------------------------------------------
    vals = shap_values.values

    # Case 1: binary classifier → (samples, features)
    if vals.ndim == 2:
        mean_abs = np.abs(vals).mean(axis=0)

    # Case 2: multi-class → (samples, classes, features)
    elif vals.ndim == 3:
        mean_abs = np.abs(vals).mean(axis=(0, 1))

    else:
        raise ValueError(f"Unexpected SHAP shape: {vals.shape}")

    n_used = mean_abs.shape[0]

    print("\n>>> SHAP returned importance for", n_used, "features")
    print(">>> Model has", len(model_features), "features")

    # SHAP returns only features used in tree splits
    shap_feature_names = model_features[:n_used]

    importance_df = pd.DataFrame({
        "feature": shap_feature_names,
        "importance": mean_abs
    }).sort_values("importance", ascending=False)

    # Top 20 plot
    imp_path = docs_dir / "shap_feature_importance.png"
    plt.figure(figsize=(10, 14))
    plt.barh(importance_df["feature"][:20], importance_df["importance"][:20])
    plt.gca().invert_yaxis()
    plt.savefig(imp_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", imp_path)

    # ----------------------------------------------------
    # REPORT
    # ----------------------------------------------------
    report_path = docs_dir / "shap_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# SHAP Explainability Report\n\n")
        f.write("## Files Generated\n")
        f.write("- shap_summary_plot.png\n")
        f.write("- shap_bar_plot.png\n")
        f.write("- shap_feature_importance.png\n\n")
        f.write("## Top 20 Important Features\n")
        f.write(importance_df.head(20).to_markdown(index=False))

    print("Report saved →", report_path)


# --------------------------------------------------------
# RUN SCRIPT
# --------------------------------------------------------
if __name__ == "__main__":
    compute_shap()
