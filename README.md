# Hybrid Network Intrusion Detection System (NIDS) - ML + XAI

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A production-ready, hybrid intrusion detection system combining **Machine Learning** and **Explainable AI (XAI)** for network security. Achieves **99.56% accuracy** with **99.99% attack detection rate** on the CICIDS2017 dataset.

## 🚀 Key Features

- **2-Stage Detection Pipeline**: Binary classifier (BENIGN vs ATTACK) + Multi-class attack type classifier (14 attack types)
- **Autoencoder Anomaly Detection**: PyTorch-based autoencoder for zero-day attack detection using reconstruction error
- **Explainable AI**: SHAP and LIME integration for model interpretability
- **Production-Ready API**: FastAPI backend with RESTful endpoints
- **Interactive Web UI**: Streamlit frontend for CSV upload and real-time predictions
- **Docker Deployment**: Fully containerized multi-service architecture
- **High Performance**: 99.56% overall accuracy, 99.99% attack detection, 99.97% DDoS detection

## 📊 Performance Metrics

| Metric | Score |
|--------|-------|
| Overall Accuracy | 99.56% |
| Attack Detection Rate (Binary) | 99.99% |
| DDoS Detection | 99.97% |
| Bot Detection | 99.93% |
| PortScan Detection | 99.87% |
| SSH-Patator Detection | 99.71% |

## 🏗️ Architecture

```
User Input (CSV)
    ↓
Preprocessing Pipeline (RobustScaler + Log1p Transforms)
    ↓
Binary Classifier (LightGBM) → BENIGN or ATTACK?
    ↓ (if ATTACK)
Attack Type Classifier (LightGBM) → 14 Attack Types
    ↓
Autoencoder Anomaly Detector (PyTorch) → Normal or Anomaly?
    ↓
Results + SHAP Explanations
```

### Supported Attack Types

Bot, DDoS, DoS GoldenEye, DoS Hulk, DoS Slowhttptest, DoS slowloris, FTP-Patator, Heartbleed, Infiltration, PortScan, SSH-Patator, Web Attack (Brute Force), Web Attack (SQL Injection), Web Attack (XSS)

## 🛠️ Installation

### Prerequisites

- Python 3.9+
- Docker & Docker Compose (for containerized deployment)
- Git

### Quick Start (Local)

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Hybrid-Network-Intrusion-Detection-System-ML-XAI.git
   cd Hybrid-Network-Intrusion-Detection-System-ML-XAI
   ```

2. **Create Python virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Place raw CSV data** (optional - for training)
   ```bash
   # Copy CICIDS2017 CSV files to data/raw/
   # Pre-trained models are already included in models/artifacts/
   ```

5. **Run the API server**
   ```bash
   python api/main.py
   ```

6. **Run the Streamlit UI** (in a separate terminal)
   ```bash
   streamlit run streamlit_app/app.py
   ```

## 🐳 Docker Deployment

The recommended deployment method using Docker Compose.

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Hybrid-Network-Intrusion-Detection-System-ML-XAI.git
   cd Hybrid-Network-Intrusion-Detection-System-ML-XAI
   ```

2. **Build and start containers**
   ```bash
   docker-compose up --build -d
   ```

3. **Access the services**
   - **Streamlit UI**: http://localhost:8501
   - **FastAPI Backend**: http://localhost:8000
   - **API Docs**: http://localhost:8000/docs

4. **View logs**
   ```bash
   docker-compose logs -f
   ```

5. **Stop containers**
   ```bash
   docker-compose down
   ```

For detailed Docker deployment instructions, see [DOCKER.md](DOCKER.md).

## 📖 Usage

### Web Interface (Streamlit)

1. Navigate to http://localhost:8501
2. Upload a CSV file containing network traffic data
3. View predictions with attack types, confidence scores, and anomaly detection
4. Export results as CSV

### API Endpoints

#### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.2, ..., 0.9]}'
```

#### Batch Prediction (CSV Upload)
```bash
curl -X POST "http://localhost:8000/predict/csv" \
  -F "file=@network_traffic.csv"
```

#### Health Check
```bash
curl "http://localhost:8000/health"
```

See full API documentation at http://localhost:8000/docs

## 📁 Project Structure

```
Hybrid-Network-Intrusion-Detection-System-ML-XAI/
├── api/                          # FastAPI backend
│   ├── main.py                  # API endpoints
│   ├── model_loader.py          # Model initialization
│   └── routes/                  # API route handlers
├── streamlit_app/               # Streamlit frontend
│   └── app.py                   # UI application
├── models/                      # Trained models
│   └── artifacts/               # Model files
│       ├── binary_lightgbm_model.txt
│       ├── attack_lightgbm_model.txt
│       ├── autoencoder_model.pth
│       ├── robust_scaler.joblib
│       ├── transforms.json
│       └── *_label_encoder.joblib
├── pipeline/                    # Data processing
│   ├── csv_preprocessor.py     # CSV preprocessing
│   └── inference.py            # 2-stage inference
├── scripts/                     # Training scripts
│   ├── retrain_classifiers_fixed.py
│   └── retrain_scaler_and_autoencoder.py
├── data/                        # Data directory
│   └── raw/                    # Place CICIDS2017 CSVs here
├── docker-compose.yml           # Multi-container orchestration
├── Dockerfile.api              # API container
├── Dockerfile.streamlit        # Streamlit container
├── DOCKER.md                   # Docker deployment guide
└── requirements.txt            # Python dependencies
```

## 🔄 Retraining Models

If you need to retrain models with custom data:

1. **Place raw CSV files** in `data/raw/`

2. **Retrain Binary and Attack Classifiers**
   ```bash
   python scripts/retrain_classifiers_fixed.py
   ```

3. **Retrain Autoencoder** (if needed)
   ```bash
   python scripts/retrain_scaler_and_autoencoder.py
   ```

4. **Restart services**
   ```bash
   docker-compose restart  # Docker deployment
   # OR restart Python processes manually
   ```

## 🧪 Testing

Run comprehensive pipeline tests:
```bash
python debug_predictions.py
```

This validates:
- Model artifacts integrity
- Preprocessing pipeline
- Binary and attack classification
- Autoencoder anomaly detection
- API endpoints
- Edge case handling

## 🔬 Technical Details

### Technologies Used

- **ML Framework**: LightGBM (gradient boosting), PyTorch (autoencoder)
- **Preprocessing**: scikit-learn RobustScaler, log1p transforms
- **Backend**: FastAPI, Uvicorn
- **Frontend**: Streamlit
- **Deployment**: Docker, Docker Compose
- **XAI**: SHAP, LIME (for model interpretability)

### Dataset

- **CICIDS2017**: Canadian Institute for Cybersecurity Intrusion Detection Dataset
- Contains network traffic with 14 attack types
- ~2.8M samples total (557K attacks, 2.27M benign)
- 78 network flow features

### Preprocessing Pipeline

1. Strip whitespace from column names
2. Apply log1p transforms to skewed features
3. Enforce feature order (78 features)
4. Replace infinities and NaN with 0
5. Clip extreme values (-1e9 to 1e9)
6. RobustScaler normalization
7. Convert to float32

**Critical**: Training and inference use identical preprocessing (no percentile clipping)

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- CICIDS2017 dataset by the Canadian Institute for Cybersecurity
- LightGBM team for the gradient boosting framework
- FastAPI and Streamlit communities

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

**Note**: Raw CICIDS2017 CSV files are not included in this repository due to size constraints. Pre-trained models are provided in `models/artifacts/` for immediate deployment.
