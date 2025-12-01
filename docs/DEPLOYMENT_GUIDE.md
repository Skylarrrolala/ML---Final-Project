# Deployment Guide - Sales Forecasting Ensemble Model

**Project**: Sales Forecasting Using Ensemble Machine Learning  
**Version**: 1.0  
**Status**: Production Ready  
**Last Updated**: December 2025

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Installation](#2-installation)
3. [Configuration](#3-configuration)
4. [Running the Model](#4-running-the-model)
5. [API Usage](#5-api-usage)
6. [Monitoring](#6-monitoring)
7. [Maintenance](#7-maintenance)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. System Requirements

### 1.1 Hardware

**Minimum**:
- CPU: 2 cores, 2.0 GHz
- RAM: 4 GB
- Storage: 1 GB free space

**Recommended**:
- CPU: 4+ cores, 2.5+ GHz
- RAM: 8+ GB
- Storage: 5+ GB free space
- GPU: Optional (speeds up LSTM training)

### 1.2 Software

**Operating System**:
- macOS 10.15+
- Ubuntu 18.04+
- Windows 10+

**Python**:
- Version: 3.8 - 3.11 (3.11 recommended)
- Package Manager: conda or pip

**Dependencies**:
- See `requirements.txt` for complete list
- Key packages: Prophet, TensorFlow, scikit-learn, pandas

---

## 2. Installation

### 2.1 Clone Repository

```bash
cd /your/desired/location
git clone https://github.com/your-username/sales-forecasting-ensemble.git
cd sales-forecasting-ensemble
```

### 2.2 Create Virtual Environment

**Using Conda** (Recommended):
```bash
conda create -n sales-forecast python=3.11
conda activate sales-forecast
```

**Using venv**:
```bash
python3.11 -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows
```

### 2.3 Install Dependencies

```bash
pip install -r requirements.txt
```

**Expected Installation Time**: 5-10 minutes

### 2.4 Verify Installation

```bash
python -c "import prophet; import tensorflow; import sklearn; print('✓ All dependencies installed')"
```

---

## 3. Configuration

### 3.1 Environment Variables

Create `.env` file in project root:

```bash
# Model Configuration
MODEL_VERSION=1.0
PROPHET_WEIGHT=0.6
LSTM_WEIGHT=0.4

# Data Configuration
DATA_PATH=data/cleaned.csv
TRAIN_TEST_SPLIT=0.75

# Forecasting Configuration
FORECAST_HORIZON=12  # months
CONFIDENCE_LEVEL=0.95

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/forecast.log

# Monitoring
ENABLE_MONITORING=true
ALERT_THRESHOLD_MAPE=25.0
```

### 3.2 Model Parameters

Edit `config/model_config.yaml`:

```yaml
prophet:
  growth: linear
  changepoint_prior_scale: 0.05
  seasonality_prior_scale: 10
  yearly_seasonality: true
  weekly_seasonality: false
  daily_seasonality: false
  seasonality_mode: multiplicative
  interval_width: 0.95

lstm:
  sequence_length: 12
  units: 50
  activation: relu
  optimizer: adam
  loss: mse
  epochs: 100
  batch_size: 32
  validation_split: 0.2

ensemble:
  weights:
    prophet: 0.6
    lstm: 0.4
  
retraining:
  schedule: monthly
  min_new_data_points: 1
  validation_months: 12
```

---

## 4. Running the Model

### 4.1 Training from Scratch

```bash
# Train ensemble model
python src/train.py --data data/cleaned.csv --output models/ensemble_v1.pkl

# Example output:
# Training Prophet model...
# Prophet MAPE: 21.6%
# Training LSTM model...
# LSTM MAPE: 32.6%
# Creating ensemble (60/40)...
# Ensemble MAPE: 19.3%
# Model saved to models/ensemble_v1.pkl
```

### 4.2 Making Predictions

```bash
# Generate 12-month forecast
python src/predict.py --model models/ensemble_v1.pkl --horizon 12 --output results/forecast_2025.csv

# Example output:
# Loading model...
# Generating 12-month forecast...
# Forecast saved to results/forecast_2025.csv
```

### 4.3 Using Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open notebooks/predictive.ipynb
# Run all cells to reproduce results
```

---

## 5. API Usage

### 5.1 Python API

```python
from src.models.ensemble import EnsembleForecaster

# Initialize model
forecaster = EnsembleForecaster(
    prophet_weight=0.6,
    lstm_weight=0.4
)

# Load data
import pandas as pd
data = pd.read_csv('data/cleaned.csv', parse_dates=['Order Date'])

# Train model
forecaster.fit(data, date_column='Order Date', target_column='Sales')

# Make predictions
forecast = forecaster.predict(periods=12, freq='M')

# Get confidence intervals
forecast_with_ci = forecaster.predict_with_intervals(
    periods=12,
    confidence_level=0.95
)

print(forecast_with_ci)
#    month   prediction  lower_bound  upper_bound
# 0  2025-01     65432        53210        77654
# 1  2025-02     67234        55012        79456
# ...
```

### 5.2 Command Line Interface

```bash
# Quick forecast
sales-forecast --data data/cleaned.csv --horizon 12

# With custom weights
sales-forecast --data data/cleaned.csv --prophet-weight 0.7 --lstm-weight 0.3

# With confidence intervals
sales-forecast --data data/cleaned.csv --horizon 12 --confidence 0.95

# Evaluate on test set
sales-forecast --data data/cleaned.csv --evaluate --test-size 12
```

### 5.3 REST API (Optional)

Start API server:
```bash
python src/api/server.py
# Server running on http://localhost:8000
```

Make predictions via HTTP:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data_path": "data/cleaned.csv",
    "horizon": 12,
    "confidence_level": 0.95
  }'
```

Response:
```json
{
  "model_version": "1.0",
  "forecast_date": "2025-01-15",
  "predictions": [
    {"month": "2025-02", "prediction": 65432, "lower": 53210, "upper": 77654},
    {"month": "2025-03", "prediction": 67234, "lower": 55012, "upper": 79456}
  ],
  "metrics": {
    "mape": 19.3,
    "r2": 0.840,
    "direction_accuracy": 83.3
  }
}
```

---

## 6. Monitoring

### 6.1 Performance Tracking

```bash
# Generate performance report
python src/evaluation/monitor.py --model models/ensemble_v1.pkl --data data/cleaned.csv

# Output: reports/performance_YYYY_MM_DD.md
```

**Key Metrics to Monitor**:
- MAPE (target: <20%)
- R² Score (target: >0.80)
- Direction Accuracy (target: >75%)
- Prediction Coverage (95% CI should contain ~95% of actuals)

### 6.2 Automated Alerts

Configure alerts in `config/monitoring.yaml`:

```yaml
alerts:
  email:
    enabled: true
    recipients:
      - data-team@company.com
    smtp_server: smtp.gmail.com
    smtp_port: 587
    
  conditions:
    - metric: mape
      threshold: 25.0
      operator: greater_than
      message: "MAPE exceeded 25% threshold"
      
    - metric: r2
      threshold: 0.75
      operator: less_than
      message: "R² dropped below 0.75"
```

Run monitoring daemon:
```bash
python src/monitoring/daemon.py
# Monitoring active. Checking performance every 24 hours...
```

### 6.3 Dashboard

Start monitoring dashboard:
```bash
streamlit run app/streamlit_app.py
# Dashboard available at http://localhost:8501
```

**Dashboard Features**:
- Real-time performance metrics
- Actual vs Predicted charts
- Error distribution plots
- Model component breakdown
- Historical performance trends

---

## 7. Maintenance

### 7.1 Retraining Schedule

**Monthly Retraining** (Recommended):
```bash
# Automated retraining script
python src/maintenance/retrain.py --schedule monthly --auto-deploy

# Manual retraining
python src/train.py --data data/cleaned.csv --incremental --output models/ensemble_v2.pkl
```

**Retraining Triggers**:
- MAPE exceeds 25% for 2+ consecutive months
- R² drops below 0.75
- New month of data available
- Quarterly review cycle

### 7.2 Model Versioning

```bash
# Save model with version
python src/train.py --data data/cleaned.csv --version 1.1 --output models/ensemble_v1.1.pkl

# List available versions
python src/models/list_versions.py
# v1.0 - 2024-12-01 - MAPE: 19.3%
# v1.1 - 2025-01-01 - MAPE: 18.9%

# Rollback to previous version
python src/models/rollback.py --version 1.0
```

### 7.3 Data Updates

```bash
# Add new month of data
python src/data/update.py --new-data data/new_month.csv --append

# Revalidate data quality
python src/data/validate.py --data data/cleaned.csv
# ✓ No missing values
# ✓ No duplicates
# ✓ Date range: 2014-12 to 2025-01
# ✓ Data quality checks passed
```

### 7.4 Backup and Recovery

```bash
# Backup models and data
python src/maintenance/backup.py --destination backups/

# Restore from backup
python src/maintenance/restore.py --source backups/2025-01-01/
```

---

## 8. Troubleshooting

### 8.1 Common Issues

#### Issue: Prophet Installation Fails

**Symptoms**: 
```
ERROR: Could not build wheels for prophet
```

**Solution**:
```bash
# Install system dependencies (macOS)
brew install cmake

# Install system dependencies (Ubuntu)
sudo apt-get install python3-dev build-essential

# Retry installation
pip install prophet
```

#### Issue: TensorFlow Not Using GPU

**Symptoms**:
```
LSTM training is slow
```

**Solution**:
```bash
# Verify GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Install GPU-enabled TensorFlow
pip install tensorflow[and-cuda]
```

#### Issue: High MAPE (>25%)

**Symptoms**:
```
Model accuracy degraded
```

**Solution**:
1. Check for data quality issues
2. Verify no missing values
3. Retrain model with latest data
4. Consider adjusting ensemble weights
5. Review for external events (holidays, promotions)

#### Issue: Predictions Look Unrealistic

**Symptoms**:
```
Negative values or extreme spikes
```

**Solution**:
1. Check input data for outliers
2. Verify date column is properly parsed
3. Ensure target column is numeric
4. Review training data for anomalies
5. Check model file integrity

### 8.2 Debug Mode

```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG
python src/train.py --data data/cleaned.csv --verbose

# Check logs
tail -f logs/forecast.log
```

### 8.3 Performance Optimization

**Slow Training**:
```python
# Reduce LSTM epochs
model_config['lstm']['epochs'] = 50  # down from 100

# Reduce cross-validation folds
cv_config['n_folds'] = 12  # down from 24

# Use smaller batch size
model_config['lstm']['batch_size'] = 64  # up from 32
```

**Memory Issues**:
```python
# Process data in chunks
for chunk in pd.read_csv('data/large.csv', chunksize=10000):
    process_chunk(chunk)

# Clear TensorFlow session
from tensorflow.keras import backend as K
K.clear_session()
```

### 8.4 Getting Help

**Documentation**:
- Full docs: `docs/`
- API reference: `docs/api.md`
- Examples: `notebooks/`

**Support**:
- GitHub Issues: [repository-url]/issues
- Email: [your-email@aupp.edu.kh]
- Slack: #sales-forecasting (if applicable)

---

## 9. Production Checklist

Before deploying to production, verify:

- [ ] All dependencies installed (`requirements.txt`)
- [ ] Model trained and validated (MAPE < 20%)
- [ ] Configuration files set up (`.env`, `model_config.yaml`)
- [ ] Monitoring dashboard deployed
- [ ] Alerts configured
- [ ] Backup system in place
- [ ] Retraining schedule established
- [ ] Documentation reviewed
- [ ] Team trained on usage
- [ ] Rollback plan documented
- [ ] Security review completed
- [ ] Performance benchmarks met

---

## 10. Security Considerations

### 10.1 Data Privacy

- Ensure data files are not committed to version control
- Add `data/*.csv` to `.gitignore`
- Use environment variables for sensitive config
- Encrypt data at rest (if required)

### 10.2 API Security

```python
# Add authentication to REST API
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/predict")
async def predict(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Verify token
    verify_token(credentials.credentials)
    # ... rest of code
```

### 10.3 Access Control

```bash
# Restrict file permissions
chmod 600 .env
chmod 600 config/secrets.yaml
chmod 700 models/
```

---

## 11. Performance Benchmarks

**Expected Performance** (on reference hardware):

| Operation | Time | Notes |
|-----------|------|-------|
| Model Loading | <1 sec | From .pkl file |
| Prophet Training | ~5 sec | 36 months data |
| LSTM Training | ~2 min | 100 epochs, CPU |
| LSTM Training | ~20 sec | 100 epochs, GPU |
| Prediction (12 months) | <1 sec | Both models |
| Full Pipeline | ~3 min | Train + predict + evaluate |

**Reference Hardware**: MacBook Pro M1, 16GB RAM

---

## 12. Appendix

### 12.1 File Structure

```
sales-forecasting-ensemble/
├── app/                    # Streamlit dashboard
├── config/                 # Configuration files
├── data/                   # Data files (not in git)
├── docs/                   # Documentation
├── models/                 # Saved models
├── notebooks/              # Jupyter notebooks
├── paper/                  # Research paper
├── presentation/           # Presentation slides
├── reports/                # Generated reports
├── results/                # Forecast outputs
├── src/                    # Source code
│   ├── models/            # Model implementations
│   ├── evaluation/        # Metrics and validation
│   ├── utils/             # Helper functions
│   ├── api/               # REST API (optional)
│   └── monitoring/        # Monitoring tools
├── tests/                  # Unit tests
├── .env                    # Environment variables
├── .gitignore             # Git ignore patterns
├── README.md              # Main documentation
├── requirements.txt       # Python dependencies
└── setup.py              # Package setup
```

### 12.2 Version History

- **v1.0** (2024-12-01): Initial release
  - Prophet + LSTM ensemble
  - MAPE: 19.3%, R²: 0.84
  - Production ready

---

**Document Version**: 1.0  
**Last Updated**: December 2025  
**Maintainer**: [Your Name]  
**Status**: Production Ready
