# Sales Forecasting with Machine Learning

**AUPP Machine Learning Final Project**  
Time Series Forecasting using XGBoost with Feature Engineering

[Research Paper](paper/main.md) | [Presentation Slides](presentation/slides.md) | [Project Structure](PROJECT_STRUCTURE.md) | [Setup Guide](QUICKSTART.md)

---

## Overview

This project implements multiple machine learning approaches for monthly sales forecasting, comparing traditional statistical methods, neural networks, and gradient boosting techniques. The goal was to develop an accurate forecasting model that could be used for business planning and inventory management.

## Key Results

- **Best Model**: XGBoost with engineered features
- **Accuracy**: 11.6% MAPE, 0.856 R² score
- **Improvement**: 40% better than ensemble baseline
- **Dataset**: 48 months of e-commerce sales data (2014-2018)
- **Features**: 43 engineered features including lags, rolling statistics, and growth metrics

---

## 📁 Project Structure

```
ML---Final-Project/
├── 📄 README.md                       # Project overview
├── 📄 PROJECT_STRUCTURE.md            # Detailed structure documentation
├── 📄 requirements.txt                # Python dependencies
├── 📄 setup.py                        # Package installation
├── 📄 run_dashboard.sh                # Quick launch script
│
├── 📂 data/                           # Dataset files
│   ├── raw.csv                        # Original data (10K transactions)
│   ├── cleaned.csv                    # Preprocessed data
│   └── featured.csv                   # Feature-engineered data
│
├── 📂 notebooks/                      # Jupyter notebooks
│   ├── eda.ipynb                      # Exploratory Data Analysis
│   ├── linear_regression.ipynb        # Baseline model
│   ├── k_means_customer_segmentation.ipynb  # Customer clustering
│   └── predictive.ipynb               # Main forecasting models ⭐
│
├── 📂 src/                            # Source code
│   └── evaluation/                    # Model evaluation scripts
│       ├── xgboost_optimized.py      # Best model (11.6% MAPE)
│       ├── feature_engineering.py     # 43 features creation
│       ├── advanced_ensemble.py       # Prophet + LSTM ensemble
│       └── run_improvement_pipeline.py # Full pipeline
│
├── 📂 results/                        # Model outputs
│   ├── saved_models/                  # All trained models (.h5, .pkl)
│   ├── production_model/              # XGBoost deployment package
│   └── xgboost_optimized/            # Performance results
│
├── 📂 app/                            # Streamlit Dashboard (6 pages)
│   ├── streamlit_app.py              # Main application
│   └── pages/                         # Dashboard pages
│
├── 📂 docs/                           # Documentation
│   ├── figures/                       # Flowcharts (methodology, project)
│   ├── DEPLOYMENT_GUIDE.md           # Production deployment
│   └── ACCURACY_METRICS_GUIDE.md     # Metrics explanation
│
├── 📂 paper/                          # Research deliverables
│   ├── main.md                        # Academic paper (8,500 words)
│   └── Sale Forcasting - Final Project.pdf
│
├── 📂 presentation/                   # Presentation materials
│   └── slides.md                      # Slide deck (21 slides)
│
└── 📂 scripts/                        # Utility scripts
    ├── generate_flowchart.py
    └── generate_methodology_flowchart.py
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

## Getting Started

### Installation

```bash
# Clone repository
git clone https://github.com/Skylarrrolala/ML---Final-Project.git
cd ML---Final-Project

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# macOS users: Install OpenMP for XGBoost
brew install libomp
```

### Running the Project

**Jupyter Notebooks** (recommended for exploration):
```bash
jupyter notebook
# Open notebooks/predictive.ipynb
```

**Streamlit Dashboard**:
```bash
streamlit run app/streamlit_app.py
```

**Python Scripts**:
```bash
python src/evaluation/xgboost_optimized.py
```

---

## Dataset

**Source**: E-commerce sales transaction data  
**Period**: December 2014 - November 2018 (48 months)  
**Records**: ~10,000 transactions  
**Granularity**: Daily transactions aggregated to monthly sales

**Available Features**:
- Temporal: Order Date
- Financial: Sales, Quantity, Profit
- Product: Category, Sub-Category
- Geographic: Country, State, City, Region
- Customer: Customer Segment, Customer ID

---

## Methodology

### 1. Exploratory Data Analysis
Started with understanding the data through visualization and statistical analysis. Identified temporal trends, seasonal patterns, and sales distributions across different categories and regions. Also performed customer segmentation using K-Means clustering.

### 2. Baseline Models
Implemented linear regression as a baseline, using time-based features (month counters and seasonal indicators). This gave us MAPE of ~25% and R² of ~0.65, establishing a benchmark for comparison.

### 3. Individual Models

**Facebook Prophet**
- Used for its strength in handling seasonality automatically
- Configured with multiplicative seasonality for yearly patterns
- Achieved 19.6% MAPE and 0.865 R²

**LSTM Neural Network**
- Implemented with 50 units and 12-month sequence length
- Trained for 100 epochs using Adam optimizer
- Performance: 30.3% MAPE, 0.405 R²

**Ensemble (Prophet + LSTM)**
- Combined the two models using weighted averaging (60% Prophet, 40% LSTM)
- Reasoning: leverage Prophet's seasonal expertise with LSTM's pattern learning
- Result: 15.2% MAPE, 0.826 R²

### 4. Best Model: XGBoost with Feature Engineering

Created 43 features across five categories:
- **Lag features** (12): Historical values at 1, 3, 6, 12 month intervals
- **Rolling statistics** (12): Moving averages, standard deviations, min/max values
- **Date features** (7): Month, quarter, cyclical encodings
- **Growth metrics** (6): Month-over-month, year-over-year changes, momentum
- **Statistical features** (6): Z-scores, percentiles, deviations from mean

Configured XGBoost with:
- 100 trees, max depth of 4
- L1 and L2 regularization to prevent overfitting
- Cross-validation for hyperparameter tuning

**Final Performance**: 11.6% MAPE, 0.856 R²

### 5. Validation

- Split data temporally (36 months training, 12 months testing)
- Performed 24-iteration walk-forward cross-validation
- Conducted statistical significance tests
- Checked residual diagnostics (normality, bias, autocorrelation)

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

### Analysis

XGBoost significantly outperformed other approaches. The 40% improvement over the ensemble model (from 19.3% to 11.6% MAPE) and 57% reduction in dollar error (from $14,123 to $6,016 MAE) demonstrates the value of systematic feature engineering.

Several factors contributed to XGBoost's success:
- Feature engineering transformed raw time series into rich tabular data
- Tree-based models handle tabular data better than neural networks in this case
- Regularization prevented overfitting despite limited training data (48 months)
- Training time was much faster (seconds vs hours for LSTM)

The five most important features were:
1. Number of orders (48.5% importance)
2. Volatility momentum (12.2%)
3. Sales percentile (9.8%)
4. Sales Z-score (7.7%)
5. 12-month lagged sales (3.6%)

This tells us that order frequency matters more than order size, and combining multiple feature types provides complementary predictive power.

---

## Academic Paper

The complete research paper includes background, literature review, detailed methodology, results analysis, and discussion of implications. Available in both markdown and PDF format in the `paper/` directory.

---

## Presentation

Presentation slides covering the problem statement, methodology, results, and business impact are available in `presentation/slides.md` (15-20 minutes).

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

## Usage Guide

**For reviewing the project:**
1. Read the research paper (`paper/main.md`)
2. Check presentation slides (`presentation/slides.md`)
3. Review the evaluation report (`reports/model_evaluation_report.md`)

**To explore the code:**
1. Start with the EDA notebook (`notebooks/eda.ipynb`)
2. Review the main predictive notebook (`notebooks/predictive.ipynb`)
3. Check the evaluation scripts (`src/evaluation/`)

**To reproduce results:**
1. Install dependencies (see Getting Started)
2. Run the notebooks in order
3. Execute evaluation scripts in `src/evaluation/`

**To use the trained models:**
1. Load models from `results/saved_models/`
2. Run predictions using scripts in `src/models/`
3. Launch the dashboard with `streamlit run app/streamlit_app.py`

---

## Business Value

### Use Cases
- Inventory planning: Forecast demand to optimize stock levels
- Revenue projection: Financial planning and budgeting
- Resource allocation: Staff scheduling for peak periods
- Marketing strategy: Time campaigns for high-demand periods

### Impact with XGBoost Model
With 11.6% MAPE, the model provides reliable forecasts for business decisions. For average monthly sales of $52,000, the ±$6,000 error margin is manageable for:
- Inventory optimization (reduce overstock/stockouts)
- Tighter safety margins (±12% vs ±20% previously)
- Better resource allocation
- More confident strategic planning

---

## Validation Summary

**Train/Test Split**: 36 months training, 12 months testing (temporal split)

**Cross-Validation**: 24-iteration walk-forward validation

**Statistical Tests**: 
- Paired t-tests confirmed XGBoost significantly outperforms individual models
- Friedman test (p = 0.016) shows statistically significant differences
- Residual diagnostics passed (normality, bias, autocorrelation checks)

**Uncertainty Quantification**: 95% confidence intervals provided with forecasts

Full validation methodology documented in `docs/ENSEMBLE_MODEL_VALIDATION_GUIDE.md`.

---

## Future Directions

Potential improvements we identified:
- Add external variables (holidays, promotions, economic indicators)
- Test newer architectures (Transformers, N-BEATS)
- Implement automated hyperparameter tuning
- Build a prediction API for production use
- Create category-specific and regional forecasting models
- Add multi-horizon forecasting (1, 3, 6, 12 months ahead)

---

## References

1. Taylor, S. J., & Letham, B. (2018). Forecasting at scale. *The American Statistician*.
2. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*.
3. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*.

Documentation: [Prophet](https://facebook.github.io/prophet/), [TensorFlow LSTM](https://www.tensorflow.org/guide/keras/rnn), [scikit-learn](https://scikit-learn.org/stable/)

---

## Contributors

Dararithy Heng, Sivhuy Hong, Saifudine Lim, Someatra Pum

Machine Learning Final Project  
AUPP (American University of Phnom Penh), Fall 2025  
Instructor: Prof. Kuntha Pin

---

## License

This project is for academic purposes as part of the AUPP Machine Learning course.

---

## Contact

Email: hdararithy@gmail.com  
GitHub: @Skylarrrolala
```
