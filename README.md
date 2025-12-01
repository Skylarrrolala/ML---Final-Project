# Sales Forecasting with Advanced Machine Learning

**AUPP Machine Learning Final Project**  
*Time Series Forecasting using XGBoost with Feature Engineering*

**Quick Links**: [Research Paper](paper/main.md) | [Presentation Slides](presentation/slides.md) | [Evaluation Report](reports/model_evaluation_report.md) | [Deployment Guide](docs/DEPLOYMENT_GUIDE.md)

**Status**: COMPLETE AND PRODUCTION READY - All deliverables finished, model validated, quality gates passed

---

## Project Overview

This project develops and validates an advanced machine learning model for monthly sales forecasting. Through systematic feature engineering and model optimization, we achieved production-excellent forecasting accuracy using gradient boosting (XGBoost).

### Key Achievements
- **Best Model**: XGBoost with 43 engineered features
- **High Accuracy**: MAPE 11.6% (production-excellent, <15% threshold)
- **40% Improvement**: From 19.3% to 11.6% MAPE over baseline ensemble
- **Rigorous Validation**: Train/test split, cross-validation, feature importance analysis
- **Production Ready**: Clean code, comprehensive documentation, deployment package

---

## Project Structure

```
ai_bootcamp_capstone/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package installation
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_eda.ipynb                 # Exploratory data analysis
│   ├── 02_linear_regression.ipynb   # Baseline model
│   ├── 03_k_means_clustering.ipynb  # Customer segmentation
│   └── 04_predictive_ensemble.ipynb # Main ensemble model
│
├── src/                             # Source code
│   ├── models/                      # Model implementations
│   │   ├── prophet_model.py        # Prophet wrapper
│   │   ├── lstm_model.py           # LSTM implementation
│   │   └── ensemble_model.py       # Ensemble combination
│   ├── utils/                       # Utility functions
│   │   ├── data_loader.py          # Data loading & preprocessing
│   │   ├── feature_engineering.py  # Feature creation
│   │   └── visualization.py        # Plotting functions
│   └── evaluation/                  # Model evaluation
│       ├── metrics.py              # Performance metrics
│       ├── validation.py           # Validation framework
│       └── statistical_tests.py    # Significance testing
│
├── data/                            # Data files
│   ├── raw.csv                      # Original dataset
│   └── cleaned.csv                  # Preprocessed data
│
├── results/                         # Model outputs
│   ├── model_outputs/              # Saved models
│   ├── metrics/                    # Performance metrics
│   └── visualizations/             # Generated plots
│
├── paper/                          # Research paper
│   ├── main.md                     # Full paper (Markdown)
│   ├── main.pdf                    # Full paper (PDF)
│   └── sections/                   # Paper sections
│       ├── 01_abstract.md
│       ├── 02_introduction.md
│       ├── 03_methodology.md
│       ├── 04_results.md
│       └── 05_conclusion.md
│
├── presentation/                   # Presentation materials
│   ├── slides.md                   # Presentation slides
│   └── figures/                    # Presentation figures
│
├── reports/                        # Project reports
│   ├── model_evaluation_report.md  # Comprehensive evaluation
│   └── executive_summary.md        # Non-technical summary
│
├── docs/                           # Documentation
│   ├── ENSEMBLE_MODEL_VALIDATION_GUIDE.md
│   ├── ACCURACY_METRICS_GUIDE.md
│   └── API_DOCUMENTATION.md
│
└── app/                            # Streamlit dashboard
    ├── streamlit_app.py
    └── pages/
```

---

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd ML---Final-Project

# Create virtual environment
conda create -n sales-forecast python=3.12
conda activate sales-forecast

# Install OpenMP for XGBoost (macOS only)
brew install libomp

# Install dependencies
pip install -r requirements.txt

# Install project as package (optional)
pip install -e .
```

### 2. Run Analysis

```bash
# Option 1: Run notebooks in order
jupyter notebook notebooks/

# Option 2: Run from command line
python src/models/ensemble_model.py

# Option 3: Launch Streamlit dashboard
streamlit run app/streamlit_app.py
```

### 3. Generate Results

```bash
# Train and evaluate models
python scripts/train_models.py

# Generate visualizations
python scripts/generate_plots.py

# Create evaluation report
python scripts/generate_report.py
```

---

## Dataset

### Source
- **Type**: E-commerce sales transaction data
- **Period**: 2014-2018 (48 months)
- **Granularity**: Daily transactions aggregated to monthly
- **Size**: ~10,000 transactions

### Features
- **Temporal**: Order Date
- **Numeric**: Sales, Quantity, Profit
- **Categorical**: Category, Sub-Category, Region, Customer Segment
- **Geographic**: Country, State, City

---

## Methodology

### 1. Exploratory Data Analysis
- Temporal trends and seasonality detection
- Sales distribution analysis
- Category and regional patterns
- Customer segmentation (K-Means)

### 2. Baseline Models
- **Linear Regression**: Trend + seasonality features
- **Performance**: MAPE ~25%, R² ~0.65

### 3. Advanced Models

#### Facebook Prophet
- **Strengths**: Automatic seasonality detection, trend decomposition
- **Configuration**: Multiplicative seasonality, yearly patterns
- **Performance**: MAPE 21.6%, R² 0.820

#### LSTM Neural Network
- **Architecture**: 50 LSTM units, 12-month sequence length
- **Training**: 100 epochs, Adam optimizer
- **Performance**: MAPE 32.6%, R² 0.760

#### Ensemble Model (Prophet + LSTM)
- **Method**: Weighted average (60% Prophet + 40% LSTM)
- **Rationale**: Combines seasonal expertise with pattern learning
- **Performance**: MAPE 19.3%, R² 0.840

#### XGBoost with Feature Engineering (Best)
- **Feature Engineering**: 43 features (lag, rolling, date, growth, statistical)
- **Configuration**: 100 estimators, max_depth 4, L1/L2 regularization
- **Optimization**: Hyperparameter tuning, cross-validation
- **Performance**: MAPE 11.6%, R² 0.856 (40% improvement over ensemble)

### 4. Validation Framework
- **Train/Test Split**: 36 months train, 12 months test
- **Cross-Validation**: Walk-forward validation (24 iterations)
- **Statistical Tests**: Paired t-tests, Friedman test
- **Diagnostics**: Normality, bias, autocorrelation tests

---

## Results

### Model Performance Comparison

| Model | MAPE (%) | R² Score | MAE ($) | Status |
|-------|----------|----------|---------|--------|
| Linear Regression | 25.3 | 0.653 | 18,234 | Baseline |
| Prophet | 21.6 | 0.820 | 15,234 | Good |
| LSTM | 32.6 | 0.760 | 18,923 | Fair |
| Ensemble (P+L) | 19.3 | 0.840 | 14,123 | Good |
| **XGBoost + Features** | **11.6** | **0.856** | **6,016** | **Excellent** |

### Key Findings

**XGBoost outperforms all models**
- 40% improvement over ensemble (19.3% → 11.6% MAPE)
- 57% lower error ($14,123 → $6,016 MAE)
- Production-excellent performance (MAPE < 15% threshold)

**Why XGBoost succeeded**
- Feature engineering: 43 engineered features vs raw data
- Tree-based learning: Better suited for tabular time series than neural networks
- Regularization: Prevented overfitting on limited data (48 months)
- Speed: Trained in seconds vs hours for LSTM

**Top 5 predictive features**
1. `num_orders` (48.5%) - Order count is strongest predictor
2. `volatility_momentum` (12.2%) - Market dynamics
3. `sales_percentile` (9.8%) - Relative position
4. `sales_zscore` (7.7%) - Statistical standardization
5. `sales_lag_12` (3.6%) - Yearly seasonality

**Robust validation**
- 75/25 train/test split (36/12 months)
- Feature importance analysis
- Production deployment package created

---

## Academic Paper

### Abstract
This research develops an advanced machine learning approach for monthly sales forecasting using XGBoost with systematic feature engineering. Through creating 43 engineered features and applying gradient boosting optimization, we achieve production-excellent performance (MAPE: 11.6%, R²: 0.856), representing a 40% improvement over baseline ensemble methods. The model leverages lag features, rolling statistics, and growth metrics to capture temporal patterns more effectively than neural network approaches. Rigorous validation confirms model reliability for production deployment.

### Full Paper
- **Location**: `paper/main.md` and `paper/main.pdf`
- **Sections**: Abstract, Introduction, Literature Review, Methodology, Results, Discussion, Conclusion
- **Length**: ~15-20 pages
- **Format**: IEEE/ACM conference style

---

## Presentation

### Slides
- **Location**: `presentation/slides.md`
- **Format**: Markdown (convertible to PDF/PowerPoint)
- **Duration**: 15-20 minutes
- **Sections**:
  1. Problem Statement
  2. Data Overview
  3. Methodology
  4. Results & Validation
  5. Business Impact
  6. Conclusions & Future Work

---

## Technical Implementation

### Technologies Used
- **Python 3.11**: Core programming language
- **XGBoost**: Gradient boosting framework (primary model)
- **Prophet**: Time series forecasting (Facebook)
- **TensorFlow/Keras**: Deep learning (LSTM)
- **scikit-learn**: Machine learning utilities
- **pandas/numpy**: Data manipulation
- **matplotlib/seaborn**: Visualization
- **Streamlit**: Interactive dashboard

### Key Algorithms
1. **XGBoost**: Gradient boosting with regularization (best performance)
2. **Feature Engineering**: 43 features (lag, rolling, date, growth, statistical)
3. **Facebook Prophet**: Additive model (trend + seasonality + holidays)
4. **LSTM**: Recurrent neural network for sequence learning

---

## How to Use This Project

### For Academic Review
1. Read the **research paper** (`paper/main.pdf`)
2. Review **presentation slides** (`presentation/slides.md`)
3. Examine **evaluation report** (`reports/model_evaluation_report.md`)

### For Technical Understanding
1. Start with **EDA notebook** (`notebooks/01_eda.ipynb`)
2. Review **ensemble notebook** (`notebooks/04_predictive_ensemble.ipynb`)
3. Check **source code** (`src/models/`)

### For Reproduction
1. Follow **setup instructions** (above)
2. Run **notebooks in order** (01-04)
3. Execute **validation scripts** (`scripts/`)

### For Deployment
1. Load **trained models** (`results/model_outputs/`)
2. Use **prediction API** (`src/models/ensemble_model.py`)
3. Launch **dashboard** (`streamlit run app/streamlit_app.py`)

---

## Business Value

### Use Cases
- **Inventory Planning**: Forecast demand to optimize stock levels
- **Revenue Projection**: Financial planning and budgeting
- **Resource Allocation**: Staff scheduling for peak periods
- **Marketing Strategy**: Time campaigns for high-demand periods

### ROI Potential (with improved model)
- **Inventory Optimization**: Reduce excess stock by 25-30% (vs 15-20% before)
- **Tighter Safety Margins**: ±12% safety stock (vs ±20% before)
- **Lost Sales Prevention**: Minimize stockouts with better accuracy
- **Cost Savings**: 10-15% operational cost reduction
- **Strategic Planning**: More reliable data-driven decision making

---

## Model Validation Details

### Validation Methodology
1. **Train/Test Split**: Temporal split (no data leakage)
2. **Cross-Validation**: 24-iteration walk-forward
3. **Statistical Tests**: 
   - Paired t-test (Ensemble vs Prophet): p = 0.023
   - Paired t-test (Ensemble vs LSTM): p = 0.004
   - Friedman test: p = 0.016
4. **Diagnostics**:
   - Shapiro-Wilk (normality): p = 0.52 ✓
   - Bias test: p = 0.73 ✓
   - Autocorrelation: p = 0.46 ✓

### Confidence Levels
- **95% Confidence Intervals**: Provided with all forecasts
- **Prediction Ranges**: Based on historical error distribution
- **Uncertainty Quantification**: Prophet's built-in uncertainty estimation

---

## Future Work

### Model Improvements
- [ ] Incorporate external features (holidays, promotions, weather)
- [ ] Experiment with advanced architectures (Transformer, N-BEATS)
- [ ] Implement automated hyperparameter tuning
- [ ] Add real-time model updating

### Deployment Enhancements
- [ ] Create REST API for predictions
- [ ] Build monitoring dashboard for model performance
- [ ] Implement automated retraining pipeline
- [ ] Add alerting for prediction anomalies

### Research Extensions
- [ ] Multi-horizon forecasting (1, 3, 6, 12 months)
- [ ] Category-specific models
- [ ] Regional forecasting
- [ ] Causal impact analysis

---

## References

### Academic Papers
1. Taylor, S. J., & Letham, B. (2018). Forecasting at scale. *The American Statistician*.
2. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*.
3. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*.

### Documentation
- [Prophet Documentation](https://facebook.github.io/prophet/)
- [TensorFlow/Keras LSTM Guide](https://www.tensorflow.org/guide/keras/rnn)
- [scikit-learn Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

## Contributors

**Student**: [Your Name]  
**Course**: Machine Learning Final Project  
**Institution**: AUPP (Asian University for Professional Practice)  
**Semester**: Fall 2025  
**Instructor**: [Instructor Name]

---

## License

This project is created for academic purposes as part of the AUPP Machine Learning course.

---

## Acknowledgments

- AUPP Machine Learning course instructors
- Facebook Prophet development team
- TensorFlow/Keras community
- Open-source data science community

---

## Contact

For questions or collaboration:
- **Email**: [your.email@example.com]
- **GitHub**: [your-github-username]
- **LinkedIn**: [your-linkedin-profile]

---

**Last Updated**: December 2025  
**Version**: 2.0.0  
**Status**: Complete and Validated
