# Model Performance Improvement Guide

This directory contains a complete pipeline to boost your sales forecasting model's performance and reliability.

## 🎯 Current Performance
- **Baseline MAPE**: 19.3%
- **R² Score**: 0.840
- **Model**: Prophet (60%) + LSTM (40%) Weighted Ensemble

## 🚀 Improvement Pipeline

### Quick Start (Automated)

Run the complete pipeline:

```bash
cd src/evaluation
python run_improvement_pipeline.py
```

This orchestrates all improvement steps automatically.

### Manual Step-by-Step

#### 1. Baseline Evaluation
Reproduce current model performance and establish metrics baseline.

```bash
python baseline_evaluation.py
```

**Outputs**:
- `results/metrics/baseline_results.json`
- `results/metrics/baseline_summary.txt`
- `results/metrics/baseline_evaluation.png`

#### 2. Feature Engineering
Add lag features, rolling statistics, date features, and interactions.

```bash
python feature_engineering.py
```

**Outputs**:
- `data/featured.csv` (50+ features)

**New Features**:
- Lag features (1, 2, 3, 6, 12 months)
- Rolling stats (mean, std, min, max over 3/6/12 months)
- Date features (month, quarter, cyclical encodings)
- Growth features (MoM, YoY, MACD, momentum)
- Statistical features (z-scores, percentiles)
- Interaction terms

#### 3. Hyperparameter Tuning (Optional - ~30-60 min)
Optimize Prophet and LSTM hyperparameters using Optuna.

```bash
python hyperparameter_tuning.py
```

**Outputs**:
- `results/tuning/tuned_hyperparameters.json`
- MLflow experiment tracking (view with `mlflow ui`)

**Tuned Parameters**:
- Prophet: changepoint scale, seasonality scale/mode
- LSTM: sequence length, units, dropout, learning rate, batch size

#### 4. Advanced Ensemble
Train stacking ensemble with XGBoost, LightGBM, Prophet, and LSTM.

```bash
python advanced_ensemble.py
```

**Outputs**:
- `results/ensemble/stacking_results.json`

**Models**:
- Base: Prophet, LSTM, XGBoost, LightGBM
- Meta-learner: Ridge regression (stacking)

#### 5. Uncertainty Quantification
Add reliable prediction intervals using conformal prediction and quantile regression.

```bash
python uncertainty_quantification.py
```

**Outputs**:
- `results/uncertainty/uncertainty_results.json`
- `results/uncertainty/uncertainty_visualization.png`

**Methods**:
- Conformal prediction (distribution-free guarantees)
- Quantile regression (flexible intervals)

## 📊 Expected Improvements

| Improvement Area | Expected Gain |
|-----------------|---------------|
| Feature Engineering | 2-5% MAPE reduction |
| Hyperparameter Tuning | 1-3% MAPE reduction |
| Advanced Ensemble | 3-7% MAPE reduction |
| **Total Expected** | **5-15% MAPE reduction** |

**Target**: MAPE from 19.3% → **12-16%** (production-grade)

## 📈 Results Structure

```
results/
├── metrics/
│   ├── baseline_results.json
│   ├── baseline_summary.txt
│   └── baseline_evaluation.png
├── tuning/
│   └── tuned_hyperparameters.json
├── ensemble/
│   └── stacking_results.json
├── uncertainty/
│   ├── uncertainty_results.json
│   └── uncertainty_visualization.png
└── improvement_summary.json
```

## 🔍 Key Metrics to Track

1. **Accuracy**: MAPE, MAE, RMSE, R²
2. **Reliability**: Prediction interval coverage
3. **Robustness**: Cross-validation stability
4. **Direction**: Trend prediction accuracy

## 💡 Tips for Maximum Improvement

### 1. Data Quality (Biggest Impact)
- Remove outliers carefully
- Check for data leakage
- Validate date consistency
- Handle missing values properly

### 2. Feature Selection
- Use feature importance from tree models
- Remove highly correlated features (VIF > 10)
- Keep domain-relevant features

### 3. Model Selection
- XGBoost/LightGBM often work best for tabular time series
- Prophet excels at capturing seasonality
- LSTM needs more data but captures complex patterns

### 4. Ensemble Strategy
- Stacking > Weighted Average for diverse models
- Use cross-validated predictions for meta-learner
- Don't overfit meta-learner (simple Ridge/Lasso works best)

### 5. Hyperparameter Tuning
- Focus on: learning rate, tree depth, regularization
- Use Bayesian optimization (Optuna) over grid search
- Budget 50-100 trials for good results

## 🛠️ Troubleshooting

### Issue: "MAPE increased after improvement"
**Solution**: 
- Check for overfitting (compare train vs test)
- Validate feature engineering didn't introduce leakage
- Try simpler models first

### Issue: "Prediction intervals too wide"
**Solution**:
- Use quantile regression instead of conformal prediction
- Calibrate on larger dataset
- Consider asymmetric intervals

### Issue: "Ensemble worse than best individual model"
**Solution**:
- Ensure model diversity (different algorithms)
- Use proper cross-validation for stacking
- Try different meta-learner regularization

## 📚 Additional Resources

- **Feature Engineering**: `feature_engineering.py` (50+ features)
- **Model Tuning**: `hyperparameter_tuning.py` (Optuna + MLflow)
- **Ensemble Methods**: `advanced_ensemble.py` (Stacking)
- **Uncertainty**: `uncertainty_quantification.py` (Conformal + Quantile)

## ✅ Checklist for Production

- [ ] MAPE < 15% on test set
- [ ] R² > 0.85
- [ ] Prediction interval coverage ≥ 90%
- [ ] Direction accuracy > 80%
- [ ] Cross-validation MAPE within ±2% of test MAPE
- [ ] All diagnostic tests passed (normality, bias, autocorrelation)
- [ ] Model versioning and experiment tracking setup
- [ ] Monitoring and retraining pipeline defined

## 🎓 Learning Path

1. **Beginner**: Run automated pipeline → understand outputs
2. **Intermediate**: Run each step manually → tweak parameters
3. **Advanced**: Modify scripts → add custom features/models

## 📞 Need Help?

Check these files for detailed implementation:
- Code comments in each `.py` file
- Docstrings for every function
- Inline explanations for complex logic

## 🔄 Iteration Workflow

1. **Baseline** → establish current performance
2. **Improve** → apply one technique at a time
3. **Evaluate** → measure improvement
4. **Iterate** → repeat with next technique
5. **Deploy** → production-ready model

---

**Remember**: Small, incremental improvements compound. Don't expect 50% gains from one change — aim for consistent 2-5% improvements across multiple areas.
