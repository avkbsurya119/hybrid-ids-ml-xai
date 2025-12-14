# Docker Deployment Guide

This guide explains how to deploy the Hybrid Network Intrusion Detection System using Docker.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- Trained models in `models/artifacts/` directory

## Architecture

The application consists of two services:

1. **API Service** (FastAPI) - Backend REST API for predictions
   - Port: 8000
   - Health check: http://localhost:8000/health

2. **Streamlit Service** - Frontend UI
   - Port: 8501
   - Health check: http://localhost:8501/_stcore/health

Both services share the same `models/` and `data/` volumes.

## Quick Start

### 1. Build and Run

```bash
# Build and start all services
docker-compose up --build -d

# View logs
docker-compose logs -f

# Check service status
docker-compose ps
```

### 2. Access Services

- **Streamlit UI**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs
- **API Health**: http://localhost:8000/health

### 3. Stop Services

```bash
# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Individual Service Commands

### Build Individual Services

```bash
# Build API only
docker build -f Dockerfile.api -t nids-api .

# Build Streamlit only
docker build -f Dockerfile.streamlit -t nids-streamlit .
```

### Run Individual Services

```bash
# Run API
docker run -p 8000:8000 -v ./models:/app/models -v ./data:/app/data nids-api

# Run Streamlit
docker run -p 8501:8501 -v ./models:/app/models -v ./data:/app/data nids-streamlit
```

## Environment Variables

### API Service
- `PYTHONUNBUFFERED=1` - Enable real-time logging

### Streamlit Service
- `PYTHONUNBUFFERED=1` - Enable real-time logging
- `API_URL=http://api:8000` - Backend API URL

## Volume Mounts

- `./models:/app/models` - Model artifacts (read-only recommended)
- `./data:/app/data` - Dataset storage

## Health Checks

Both services include health checks:

- **API**: Checks if FastAPI is responding and models are loaded
- **Streamlit**: Checks if Streamlit server is running

## Troubleshooting

### Container won't start

```bash
# Check logs
docker-compose logs api
docker-compose logs streamlit

# Rebuild from scratch
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Models not found

Ensure models are trained and present in `models/artifacts/`:
- `binary_lightgbm_model.txt`
- `attack_lightgbm_model.txt`
- `autoencoder_model.pth`
- `autoencoder_threshold.json`
- `robust_scaler.joblib`
- `transforms.json`
- `binary_feature_order.json`
- `attack_feature_order.json`
- `attack_label_encoder.joblib`

### Port already in use

Change ports in `docker-compose.yml`:

```yaml
ports:
  - "8080:8000"  # API on port 8080
  - "8502:8501"  # Streamlit on port 8502
```

## Production Deployment

For production, consider:

1. **Use environment-specific compose files**
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
   ```

2. **Set resource limits**
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '2'
         memory: 4G
   ```

3. **Enable HTTPS**
   - Use a reverse proxy (nginx, traefik)
   - Configure SSL certificates

4. **Set up monitoring**
   - Container metrics (Prometheus)
   - Log aggregation (ELK stack)

5. **Use secrets management**
   - Don't hardcode credentials
   - Use Docker secrets or external secret managers

## Development Mode

To develop with live code reload:

```yaml
# Add to docker-compose.yml under services
volumes:
  - ./api:/app/api
  - ./streamlit_app:/app/streamlit_app
```

Restart containers after code changes:
```bash
docker-compose restart api
docker-compose restart streamlit
```
