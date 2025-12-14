from fastapi import FastAPI, HTTPException
from api.inference_2stage import run_2stage_inference
from api.model_loader import model_bundle
from fastapi import UploadFile, File
import pandas as pd
from api.schemas import (
    FlowInput,
    PredictionResponse,
    CSVPredictionResponse   # ✅ ADD THIS
)
from pipeline.csv_inference_2stage import run_2stage_csv_inference


app = FastAPI(
    title="Hybrid Network Intrusion Detection System",
    version="1.0.0"
)


@app.get("/")
def root():
    return {"status": "IDS API running"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "nids-api",
        "models_loaded": {
            "binary": model_bundle.binary_model is not None,
            "attack": model_bundle.attack_model is not None,
            "autoencoder": model_bundle.autoencoder is not None
        }
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(flow: FlowInput):
    try:
        return run_2stage_inference(
            record=flow.features,
            bundle=model_bundle
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/csv", response_model=CSVPredictionResponse)
def predict_csv(file: UploadFile = File(...)):
    try:
        # -------- Read CSV --------
        df = pd.read_csv(file.file)
        df = df.replace([float("inf"), float("-inf")], 0).fillna(0)

        # -------- Run 2-stage inference --------
        out = run_2stage_csv_inference(
            df_raw=df,
            bin_model=model_bundle.binary_model,
            atk_model=model_bundle.attack_model,
            atk_encoder=model_bundle.attack_label_encoder,
            scaler=model_bundle.scaler,
            transforms=model_bundle.transforms,
            bin_features=model_bundle.binary_features,
            atk_features=model_bundle.attack_features,
            autoencoder=model_bundle.autoencoder,
            ae_threshold=model_bundle.autoencoder_threshold
        )

        # -------- Summary --------
        benign = (out["Final_Prediction"] == "BENIGN").sum()
        attack = len(out) - benign

        return {
            "total_rows": len(out),
            "attack_rows": attack,
            "benign_rows": benign,
            "results": out.rename(columns={
                "Final_Prediction": "final_prediction",
                "Binary_Route": "binary_route",
                "Confidence": "confidence",
                "Anomaly": "anomaly"
            }).to_dict(orient="records")
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
