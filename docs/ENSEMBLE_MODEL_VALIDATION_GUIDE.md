# 🧪 Ensemble Model Validation & Testing Guide

## Overview
This document provides a comprehensive guide for validating and testing the ensemble model (Prophet + LSTM) accuracy in the `predictive.ipynb` notebook.

## Ensemble Model Configuration
- **Composition**: 60% Prophet + 40% LSTM
- **Prophet**: Captures seasonality and trend patterns
- **LSTM**: Learns long-term dependencies and complex patterns
- **Data**: Monthly sales aggregated from the cleaned dataset

---

## Validation Methodology

### 1. Train/Test Split Validation ✅

**Purpose**: Evaluate model performance on unseen data

**Implementation**:
```python
# Split data: Last 12 months for testing
train_size = len(monthly_sales) - 12
train_data = monthly_sales[:train_size]
test_data = monthly_sales[train_size:]
```

**Key Metrics to Calculate**:
- **MAE (Mean Absolute Error)**: Average absolute prediction error in dollars
- **RMSE (Root Mean Squared Error)**: Penalizes larger errors more heavily
- **MAPE (Mean Absolute Percentage Error)**: Error as a percentage (industry standard)
- **R² Score**: Proportion of variance explained (0-1, higher is better)
- **Direction Accuracy**: % of times the model correctly predicts trend direction

**Interpretation Guidelines**:
| MAPE Range | Assessment |
|------------|------------|
| < 10% | Excellent |
| 10-20% | Good |
| 20-30% | Acceptable |
| > 30% | Needs Improvement |

---

### 2. Month-by-Month Performance Analysis ✅

**Purpose**: Identify which months the model performs best/worst

**What to Track**:
```
For each test month:
- Actual Sales
- Prophet Prediction
- LSTM Prediction  
- Ensemble Prediction
- Absolute Error
- Percentage Error
```

**Key Insights to Look For**:
- Does the model perform better during certain seasons?
- Are there outlier months with high errors?
- Is the ensemble consistently better than individual models?

---

### 3. Residual Analysis ✅

**Purpose**: Check if prediction errors are random or systematic

**Diagnostic Tests**:

**a) Normality Test (Shapiro-Wilk)**
- H₀: Residuals are normally distributed
- **Pass if**: p-value > 0.05
- **Why it matters**: Normal residuals indicate the model captured most patterns

**b) Bias Test (One-sample t-test)**
- H₀: Mean residual = 0
- **Pass if**: p-value > 0.05
- **Why it matters**: No bias means predictions aren't systematically high or low

**c) Autocorrelation Test**
- H₀: Consecutive residuals are independent  
- **Pass if**: p-value > 0.05
- **Why it matters**: No autocorrelation means model captured temporal dependencies

**Visual Checks**:
1. **Residual Plot**: Should show random scatter around zero
2. **Histogram**: Should resemble a bell curve
3. **Q-Q Plot**: Points should follow the diagonal line
4. **Time Series Plot**: No obvious patterns over time

---

### 4. Cross-Validation (Walk-Forward) ✅

**Purpose**: Test model stability across different time periods

**Method**: Expanding Window Validation
```
Start: 24 months training → predict month 25
Step 1: 25 months training → predict month 26
Step 2: 26 months training → predict month 27
... continue until end of data
```

**Advantages**:
- Simulates real-world forecasting
- Tests model on multiple time periods
- More robust than single train/test split

**Metrics to Report**:
- CV Mean Absolute Error
- CV RMSE
- CV MAPE
- CV R² Score
- Number of validation iterations

**What to Look For**:
- CV metrics should be similar to test set metrics
- Large differences indicate overfitting or instability

---

### 5. Statistical Significance Testing ✅

**Purpose**: Determine if ensemble is statistically better than individual models

**Tests to Perform**:

**a) Paired t-test: Ensemble vs Prophet**
```python
t_stat, p_val = stats.ttest_rel(ensemble_errors, prophet_errors)
```
- **Significant if**: p < 0.05
- **Interpretation**: Ensemble performs statistically different from Prophet

**b) Paired t-test: Ensemble vs LSTM**
```python
t_stat, p_val = stats.ttest_rel(ensemble_errors, lstm_errors)
```
- **Significant if**: p < 0.05
- **Interpretation**: Ensemble performs statistically different from LSTM

**c) Friedman Test (Overall Comparison)**
```python
stat, p = stats.friedmanchisquare(prophet_errors, lstm_errors, ensemble_errors)
```
- **Significant if**: p < 0.05
- **Interpretation**: At least one model performs differently

---

## Key Performance Indicators (KPIs)

### Primary KPIs
1. **MAPE** - Most interpretable for business stakeholders
2. **R²** - Shows how much variance is explained
3. **Direction Accuracy** - Critical for trend prediction

### Secondary KPIs
4. **MAE** - Absolute error in dollars
5. **RMSE** - Penalizes large errors
6. **Residual Statistics** - Checks for bias and patterns

---

## Expected Results

Based on the ensemble methodology (60% Prophet + 40% LSTM):

### Ensemble Should Outperform When:
✓ Data has both strong seasonality AND complex patterns
✓ Test period includes various market conditions
✓ Individual models complement each other's weaknesses

### Individual Models Might Win When:
- **Prophet wins**: Strong seasonal patterns, regular business
- **LSTM wins**: Complex non-linear trends, irregular patterns

### Ensemble Value Proposition:
Even if not always best, ensemble provides:
- **Robustness**: Less sensitive to any single model's failures
- **Balanced predictions**: Combines different perspectives
- **Risk mitigation**: Smoother, more stable forecasts

---

## Validation Checklist

Before claiming the ensemble model is validated:

- [ ] Train/test split shows reasonable MAPE (< 30%)
- [ ] R² score is positive and meaningful (> 0.5)
- [ ] Residuals pass normality test OR are close to normal
- [ ] No significant bias in predictions (mean residual ≈ 0)
- [ ] Cross-validation metrics align with test metrics
- [ ] Direction accuracy > 60% (better than random)
- [ ] Ensemble provides value over individual models
- [ ] Statistical tests performed and interpreted
- [ ] Visualizations created for stakeholder communication

---

## Recommended Visualizations

1. **Actual vs Predicted** - Line plot showing all three models
2. **Ensemble with Confidence Intervals** - Shows uncertainty
3. **Residual Plot** - Scatter plot of residuals vs predictions
4. **Histogram of Residuals** - Distribution check
5. **Q-Q Plot** - Normality assessment
6. **Time Series of Errors** - Pattern identification
7. **Model Comparison Bar Chart** - Metric comparison
8. **Cross-Validation Results** - Walk-forward performance

---

## How to Run the Validation

### Option 1: Use the Validation Cells in predictive.ipynb

The notebook now includes cells for comprehensive validation:

1. **Cell: Data Preparation** - Loads and splits data
2. **Cell: Model Training** - Trains Prophet and LSTM
3. **Cell: Test Predictions** - Generates predictions on test set
4. **Cell: Performance Metrics** - Calculates all metrics
5. **Cell: Visualizations** - Creates comparison plots
6. **Cell: Month-by-Month Analysis** - Detailed breakdown
7. **Cell: Residual Analysis** - Diagnostic tests
8. **Cell: Cross-Validation** - Walk-forward validation
9. **Cell: Statistical Tests** - Significance testing
10. **Cell: Final Summary** - Comprehensive report

### Option 2: Manual Validation

If you prefer to validate manually, use this structure:

```python
# 1. Calculate metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
mape = np.mean(np.abs((actual - predicted) / actual)) * 100
r2 = r2_score(actual, predicted)

# 2. Test residuals
from scipy import stats

residuals = actual - predicted
shapiro_stat, shapiro_p = stats.shapiro(residuals)
t_stat, t_p = stats.ttest_1samp(residuals, 0)

# 3. Compare models
t_stat, p_val = stats.ttest_rel(ensemble_errors, prophet_errors)

print(f"MAPE: {mape:.2f}%")
print(f"R²: {r2:.4f}")
print(f"Normality: {'PASS' if shapiro_p > 0.05 else 'FAIL'}")
print(f"Bias: {'PASS' if t_p > 0.05 else 'FAIL'}")
```

---

## Reporting Results

### For Technical Audiences:
- Report all metrics with confidence intervals
- Include statistical test results
- Show diagnostic plots
- Discuss residual patterns

### For Business Stakeholders:
- Focus on MAPE and dollar errors
- Use visualizations of actual vs predicted
- Explain direction accuracy in business terms
- Provide forecast confidence levels

### Sample Summary Statement:

> "The ensemble model (60% Prophet + 40% LSTM) achieves a MAPE of 21.5% on the 12-month test set, with an R² of 0.78, meaning it explains 78% of sales variance. The model correctly predicts the direction of sales changes 75% of the time. Cross-validation across 20 time periods confirms model stability with consistent performance (CV MAPE: 22.1%). Statistical tests show the ensemble significantly outperforms individual models (p < 0.05). The model is production-ready for monthly sales forecasting with recommended safety margins of ±20% for inventory planning."

---

## Troubleshooting

### If MAPE > 30%:
- Check for outliers in data
- Consider additional features
- Try different model weights
- Increase training data
- Investigate underperforming months

### If Residuals Show Patterns:
- Add seasonal adjustments
- Include external variables
- Try different model architectures
- Check for data quality issues

### If Cross-Validation Fails:
- Reduce validation window
- Check for data leakage
- Verify model is truly making forward predictions
- Consider non-stationary data transformations

---

## Next Steps After Validation

1. **Document Results** - Create validation report
2. **Share with Stakeholders** - Present findings
3. **Deploy Model** - Move to production if validated
4. **Monitor Performance** - Track actual vs predicted
5. **Retrain Schedule** - Plan regular updates
6. **Set Alerts** - Define thresholds for model degradation

---

## Conclusion

A properly validated ensemble model should:
- ✅ Achieve acceptable accuracy (MAPE < 30%)
- ✅ Pass diagnostic tests (or understand why not)
- ✅ Perform consistently across time periods
- ✅ Provide value over individual models
- ✅ Be interpretable and explainable

Use this guide to systematically validate your ensemble model and build confidence in its predictions!
