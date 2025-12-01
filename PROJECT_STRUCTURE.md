# ML Sales Forecasting - Project Structure

## Directory Organization

```
ML---Final-Project/
├── README.md                          # Main project documentation
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package installation config
├── run_dashboard.sh                  # Quick launch script for Streamlit
├── .gitignore                        # Git ignore rules
│
├── data/                             # Dataset files
│   ├── raw.csv                          # Original data (10K transactions)
│   ├── cleaned.csv                      # Preprocessed data
│   └── featured.csv                     # Feature-engineered data
│
├── notebooks/                        # Jupyter notebooks (analysis & modeling)
│   ├── eda.ipynb                        # Exploratory Data Analysis
│   ├── linear_regression.ipynb          # Baseline linear model
│   ├── k_means_customer_segmentation.ipynb  # Customer clustering
│   └── predictive.ipynb                 # Main forecasting models (Prophet, LSTM, Ensemble, XGBoost)
│
├── src/                              # Source code modules
│   ├── evaluation/                      # Model evaluation scripts
│   │   ├── baseline_evaluation.py       # Linear regression baseline
│   │   ├── feature_engineering.py       # 43 features creation
│   │   ├── xgboost_optimized.py        # Best model (11.6% MAPE)
│   │   ├── tree_ensemble_simple.py      # Tree-based ensemble
│   │   ├── advanced_ensemble.py         # Prophet + LSTM ensemble
│   │   ├── hyperparameter_tuning.py     # Model optimization
│   │   ├── visual_comparison.py         # Performance visualizations
│   │   ├── uncertainty_quantification.py # Confidence intervals
│   │   ├── save_production_model.py     # Model serialization
│   │   ├── run_improvement_pipeline.py  # Full pipeline execution
│   │   ├── quickstart.py               # Quick model testing
│   │   └── README.md                    # Evaluation documentation
│   ├── models/                          # Model architectures (empty - using notebooks)
│   └── utils/                           # Helper utilities (empty)
│
├── results/                          # Model outputs & artifacts
│   ├── saved_models/                    # Trained models for deployment
│   │   ├── lstm_model.h5               # LSTM neural network
│   │   ├── lstm_scaler.pkl             # MinMax scaler for LSTM
│   │   ├── prophet_model.pkl           # Prophet model
│   │   ├── ensemble_config.pkl         # Ensemble weights
│   │   ├── feature_scaler_X.pkl        # Feature scaler
│   │   └── feature_scaler_y.pkl        # Target scaler
│   ├── production_model/               # XGBoost production package
│   │   ├── xgboost_model.pkl          # Best model (11.6% MAPE)
│   │   ├── feature_names.json         # Feature list
│   │   ├── model_metadata.json        # Model info
│   │   ├── training_statistics.json   # Performance stats
│   │   ├── prediction_example.py      # Usage example
│   │   └── README.md                   # Deployment guide
│   ├── xgboost_optimized/             # XGBoost results
│   │   ├── results.json               # Performance metrics
│   │   ├── feature_importance.csv     # Feature rankings
│   │   ├── predictions.csv            # Test predictions
│   │   └── predictions.png            # Visualization
│   ├── tree_ensemble/                 # Ensemble results
│   │   ├── results.json
│   │   └── feature_importance.csv
│   ├── visual_comparison/             # Comparison charts
│   │   ├── summary.json
│   │   ├── performance_comparison.png
│   │   ├── prediction_comparison.png
│   │   ├── feature_importance.png
│   │   └── improvement_summary.png
│   ├── metrics/                       # Evaluation metrics (empty)
│   ├── model_outputs/                 # Model outputs (empty)
│   └── visualizations/                # Generated plots (empty)
│
├── app/                              # Streamlit Dashboard Application
│   ├── streamlit_app.py               # Main app entry point
│   ├── config.py                      # App configuration
│   ├── data_loader.py                 # Data loading utilities
│   ├── utils.py                       # Helper functions
│   ├── README.md                      # App documentation
│   └── pages/                         # Multi-page app components
│       ├── __init__.py
│       ├── overview.py                # Dashboard overview
│       ├── time_analysis.py           # Time series analysis
│       ├── geographic_analysis.py     # Regional insights
│       ├── product_analysis.py        # Product performance
│       ├── customer_analysis.py       # Customer segmentation
│       └── ai_insights.py             # ML predictions
│
├── docs/                             # Documentation & guides
│   ├── figures/                       # Generated flowcharts
│   │   ├── methodology_flowchart.png  # Research methodology diagram
│   │   ├── methodology_flowchart.pdf
│   │   ├── project_flowchart.png      # Overall project flow
│   │   └── project_flowchart.pdf
│   ├── ACCURACY_METRICS_GUIDE.md      # Metrics explanation
│   ├── ENSEMBLE_MODEL_VALIDATION_GUIDE.md  # Validation framework
│   ├── DEPLOYMENT_GUIDE.md            # Production deployment
│   └── README_VALIDATION.md           # Validation documentation
│
├── paper/                            # Research Paper
│   ├── main.md                        # Full academic paper (8,500 words)
│   └── Sale Forcasting - Final Project.pdf  # PDF version
│
├── presentation/                     # Presentation Materials
│   └── slides.md                      # Complete slide deck (21 slides)
│
├── reports/                          # Analysis Reports
│   └── model_evaluation_report.md     # Model evaluation summary
│
└── scripts/                          # Utility Scripts
    ├── generate_flowchart.py          # Project flowchart generator
    └── generate_methodology_flowchart.py  # Methodology diagram generator
```

## Key Project Components

### Data Pipeline
1. **Raw Data** → **Cleaned Data** → **Featured Data**
2. **EDA** (eda.ipynb) → **Modeling** (predictive.ipynb)

### Models Implemented
1. **Baseline**: Linear Regression (25.3% MAPE)
2. **Statistical**: Facebook Prophet (19.6% MAPE)
3. **Deep Learning**: LSTM Neural Network (30.3% MAPE)
4. **Ensemble**: Prophet + LSTM (15.2% MAPE)
5. **Best**: XGBoost + 43 Features (11.6% MAPE)

### Production Assets
- **Saved Models**: `results/saved_models/` (all trained models)
- **Production Model**: `results/production_model/` (XGBoost deployment package)
- **Dashboard**: `app/` (Streamlit 6-page application)

### Documentation
- **README.md**: Project overview & setup
- **paper/main.md**: Academic research paper
- **presentation/slides.md**: Presentation deck
- **docs/**: Comprehensive guides & flowcharts

## Quick Start

### Run Notebooks
```bash
jupyter notebook notebooks/predictive.ipynb
```

### Launch Dashboard
```bash
streamlit run app/streamlit_app.py
# or
./run_dashboard.sh
```

### Train Models
```bash
python src/evaluation/run_improvement_pipeline.py
```

### Generate Flowcharts
```bash
python scripts/generate_methodology_flowchart.py
```

## Performance Summary

| Model | MAPE | R² | MAE ($) | Status |
|-------|------|-------|---------|--------|
| Linear Regression | 25.3% | 0.653 | 18,234 | Baseline |
| Prophet | 19.6% | 0.865 | 7,285 | Good |
| LSTM | 30.3% | 0.405 | 12,242 | Poor |
| Ensemble | 15.2% | 0.826 | 6,881 | Good |
| **XGBoost** | **11.6%** | **0.856** | **6,016** | **Best** |

