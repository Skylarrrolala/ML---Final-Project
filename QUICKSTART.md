# Quick Start Guide

## Installation & Setup

### 1. Clone & Navigate
```bash
cd "ML---Final-Project"
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt

# macOS users: Install OpenMP for XGBoost
brew install libomp
```

### 4. Install Package (Optional)
```bash
pip install -e .
```

---

## Common Tasks

### Run Jupyter Notebooks

**1. Launch Jupyter**
```bash
jupyter notebook
```

**2. Open Key Notebooks**
- `notebooks/eda.ipynb` - Data exploration
- `notebooks/predictive.ipynb` - All forecasting models (main notebook)

### Run Streamlit Dashboard

**Method 1: Using script**
```bash
./run_dashboard.sh
```

**Method 2: Direct command**
```bash
streamlit run app/streamlit_app.py
```

**Access**: Open browser to `http://localhost:8501`

### Train Models from Scratch

**Run complete pipeline**
```bash
python src/evaluation/run_improvement_pipeline.py
```

**Train specific models**
```bash
# XGBoost (best model)
python src/evaluation/xgboost_optimized.py

# Ensemble
python src/evaluation/advanced_ensemble.py

# Feature engineering
python src/evaluation/feature_engineering.py
```

### Generate Documentation

**Flowcharts**
```bash
python scripts/generate_methodology_flowchart.py
python scripts/generate_flowchart.py
```

**Outputs**: `docs/figures/methodology_flowchart.png` & `project_flowchart.png`

---

## View Results

### Model Performance
- **Metrics**: `results/xgboost_optimized/results.json`
- **Feature Importance**: `results/xgboost_optimized/feature_importance.csv`
- **Predictions**: `results/xgboost_optimized/predictions.csv`

### Saved Models
- **LSTM**: `results/saved_models/lstm_model.h5`
- **Prophet**: `results/saved_models/prophet_model.pkl`
- **XGBoost**: `results/production_model/xgboost_model.pkl` (best model)

### Documentation
- **Research Paper**: `paper/main.md`
- **Presentation**: `presentation/slides.md`
- **Flowcharts**: `docs/figures/`

---

## Troubleshooting

### Issue: OpenMP error on macOS
```bash
brew install libomp
```

### Issue: Streamlit import error
```bash
pip install streamlit plotly
```

### Issue: LSTM model won't save
```bash
# Models saved as .h5 (HDF5 format)
# Warning about .keras format is normal, .h5 works fine
```

### Issue: Jupyter kernel not found
```bash
python -m ipykernel install --user --name=.venv
```

---

## Project Workflow

```
1. Data Preparation
   └─ notebooks/eda.ipynb

2. Model Development
   └─ notebooks/predictive.ipynb

3. Model Evaluation
   └─ src/evaluation/xgboost_optimized.py

4. Production Deployment
   └─ results/production_model/

5. Dashboard Application
   └─ app/streamlit_app.py
```

---

## Learning Path

**For Beginners**:
1. `README.md` - Project overview
2. `PROJECT_STRUCTURE.md` - File organization
3. `notebooks/eda.ipynb` - Data exploration
4. `presentation/slides.md` - Summary presentation

**For Developers**:
1. `notebooks/predictive.ipynb` - Model implementations
2. `src/evaluation/` - Source code
3. `paper/main.md` - Research methodology
4. `docs/DEPLOYMENT_GUIDE.md` - Production guide

**For Reviewers**:
1. `presentation/slides.md` - 21-slide deck
2. `paper/main.md` - 8,500-word paper
3. `docs/figures/methodology_flowchart.png` - Visual summary
4. Dashboard: `streamlit run app/streamlit_app.py`

---

## Verification Checklist

After setup, verify everything works:

```bash
# Test Python imports
python -c "import pandas, numpy, sklearn, prophet, keras, xgboost; print('All imports successful')"

# Test Streamlit
python -c "import sys; sys.path.insert(0, 'app'); from streamlit_app import main; print('Streamlit ready')"

# Test model loading
python -c "from keras.models import load_model; import pickle; print('Model loading ready')"
```

Expected output:
```
All imports successful
Streamlit ready
Model loading ready
```

---

## Need Help?

- **Documentation**: See `docs/` folder
- **Project Structure**: `PROJECT_STRUCTURE.md`
- **Deployment**: `docs/DEPLOYMENT_GUIDE.md`
- **Paper**: `paper/main.md` (comprehensive methodology)
