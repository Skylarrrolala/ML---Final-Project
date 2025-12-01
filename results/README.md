# Results Directory

This directory contains all model outputs, trained models, and performance metrics.

## 📁 Directory Structure

### `saved_models/` - All Trained Models
Serialized models ready for loading and predictions:
- `lstm_model.h5` - LSTM neural network (Keras HDF5 format)
- `lstm_scaler.pkl` - MinMax scaler for LSTM preprocessing
- `prophet_model.pkl` - Facebook Prophet statistical model
- `ensemble_config.pkl` - Ensemble weights and metrics
- `feature_scaler_X.pkl` - Feature scaler (StandardScaler)
- `feature_scaler_y.pkl` - Target scaler (StandardScaler)

### `production_model/` - Best Model Package ⭐
XGBoost deployment package (11.6% MAPE):
- `xgboost_model.pkl` - Trained XGBoost model
- `feature_names.json` - List of 43 features
- `model_metadata.json` - Model configuration
- `training_statistics.json` - Performance metrics
- `prediction_example.py` - Usage example
- `README.md` - Deployment documentation

### `xgboost_optimized/` - XGBoost Results
Performance data and visualizations:
- `results.json` - Complete metrics (MAPE, R², MAE, RMSE)
- `feature_importance.csv` - Feature rankings
- `predictions.csv` - Test set predictions
- `predictions.png` - Visualization

### `tree_ensemble/` - Ensemble Results
Tree-based ensemble outputs:
- `results.json` - Performance metrics
- `feature_importance.csv` - Feature importance

### `visual_comparison/` - Model Comparison
Comparison charts across all models:
- `summary.json` - Comparison summary
- `performance_comparison.png` - Bar charts
- `prediction_comparison.png` - Prediction plots
- `feature_importance.png` - Feature rankings
- `improvement_summary.png` - Improvement visualization

### `metrics/`, `model_outputs/`, `visualizations/`
Reserved for additional outputs (currently empty)

---

## 🎯 Performance Summary

| Model | MAPE | R² | MAE ($) | Location |
|-------|------|-------|---------|----------|
| **XGBoost** | **11.6%** | **0.856** | **6,016** | `production_model/` |
| Ensemble | 15.2% | 0.826 | 6,881 | `saved_models/` |
| Prophet | 19.6% | 0.865 | 7,285 | `saved_models/` |
| LSTM | 30.3% | 0.405 | 12,242 | `saved_models/` |

---

## 🚀 Quick Usage

### Load Best Model (XGBoost)
```python
import pickle

# Load model
with open('results/production_model/xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load feature names
import json
with open('results/production_model/feature_names.json', 'r') as f:
    features = json.load(f)

# Make prediction
# prediction = model.predict(X_new)
```

### Load LSTM Model
```python
from keras.models import load_model
import pickle

# Load model
lstm = load_model('results/saved_models/lstm_model.h5')

# Load scaler
with open('results/saved_models/lstm_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Prepare data and predict
# scaled_data = scaler.transform(data)
# prediction = lstm.predict(scaled_data)
```

### Load Prophet Model
```python
import pickle

with open('results/saved_models/prophet_model.pkl', 'rb') as f:
    prophet = pickle.load(f)

# Make future predictions
# future = prophet.make_future_dataframe(periods=12, freq='MS')
# forecast = prophet.predict(future)
```

---

## 📊 Feature Importance (Top 5)

From XGBoost model:
1. **num_orders** (48.5%) - Number of orders per month
2. **volatility_momentum** (12.2%) - Market volatility changes
3. **sales_percentile** (9.8%) - Sales rank position
4. **sales_zscore** (7.7%) - Standardized sales value
5. **sales_lag_12** (3.6%) - Sales from 12 months ago

See `xgboost_optimized/feature_importance.csv` for complete rankings.

---

**Note**: All models trained on 36 months (Jan 2015 - Dec 2017), tested on 12 months (Jan - Dec 2018).
