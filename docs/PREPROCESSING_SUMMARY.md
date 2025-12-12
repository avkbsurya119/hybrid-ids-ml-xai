# Preprocessing Summary (NIDS)

This document summarizes the preprocessing artifacts and how they are used.

## Artifacts produced
- `preprocessor/percentiles.json` — per-column percentile map (used for clipping)
- `preprocessor/transforms.json` — which numerical columns received `log1p`
- `models/artifacts/robust_scaler.joblib` — fitted RobustScaler to scale numeric features
- `data/processed/*.parquet` — transformed/partitioned dataset

## Scripts
- `scripts/compute_percentiles.py` — sample-based percentile calculation
- `scripts/transform_and_scale.py` — clipping, log1p, scaling, writes transformed parquet
- `scripts/generate_feature_report.py` — produces `docs/feature_report.csv` and `docs/top_correlations.csv`
- `scripts/validate_pipeline.py` — quick runtime validation of artifacts
- `app/ui/validation_streamlit.py` — simple UI for previewing transformed data

## Next suggestions
- Add unit tests that run on `data/demo/sample.csv` to validate transforms deterministically.
- Compute SHAP background sample if you are going to use XAI (ask before heavy runs).
- If you want end-to-end CI, add a small `data/demo/sample.csv` and a workflow that runs `scripts/validate_pipeline.py`.

