"""
Simple Tree-Based Ensemble (No Neural Networks)
Uses only XGBoost and LightGBM for reliability
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from pathlib import Path
import json

def load_data(filepath='data/featured.csv'):
    """Load featured dataset"""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Convert date column if exists
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
    
    return df

def prepare_features(df, target_col='sales'):
    """Prepare features and target"""
    # Drop non-feature columns
    exclude_cols = ['date', target_col, 'InvoiceDate', 'InvoiceNo', 
                    'CustomerID', 'Country', 'Description', 'StockCode']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Handle missing values
    X = X.fillna(method='ffill').fillna(0)
    
    # Remove any infinite values
    X = X.replace([np.inf, -np.inf], 0)
    
    print(f"Features: {len(feature_cols)}")
    print(f"Samples: {len(X)}")
    
    return X, y, feature_cols

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost model"""
    print("\nTraining XGBoost...")
    
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_mape = mean_absolute_percentage_error(y_train, train_pred) * 100
    test_mape = mean_absolute_percentage_error(y_test, test_pred) * 100
    test_r2 = r2_score(y_test, test_pred)
    
    print(f"  Train MAPE: {train_mape:.2f}%")
    print(f"  Test MAPE: {test_mape:.2f}%")
    print(f"  Test R²: {test_r2:.3f}")
    
    return model, test_pred

def train_lightgbm(X_train, y_train, X_test, y_test):
    """Train LightGBM model"""
    print("\nTraining LightGBM...")
    
    model = lgb.LGBMRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )
    
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_mape = mean_absolute_percentage_error(y_train, train_pred) * 100
    test_mape = mean_absolute_percentage_error(y_test, test_pred) * 100
    test_r2 = r2_score(y_test, test_pred)
    
    print(f"  Train MAPE: {train_mape:.2f}%")
    print(f"  Test MAPE: {test_mape:.2f}%")
    print(f"  Test R²: {test_r2:.3f}")
    
    return model, test_pred

def create_ensemble(predictions_dict, y_test):
    """Create weighted ensemble"""
    print("\n" + "="*70)
    print("ENSEMBLE RESULTS")
    print("="*70)
    
    # Try different weight combinations
    weight_combinations = [
        {'xgb': 0.5, 'lgb': 0.5, 'name': 'Equal Weight'},
        {'xgb': 0.6, 'lgb': 0.4, 'name': 'XGB Favored'},
        {'xgb': 0.4, 'lgb': 0.6, 'name': 'LGB Favored'},
    ]
    
    best_mape = float('inf')
    best_ensemble = None
    best_weights = None
    
    for weights in weight_combinations:
        ensemble_pred = (
            weights['xgb'] * predictions_dict['xgb'] +
            weights['lgb'] * predictions_dict['lgb']
        )
        
        mape = mean_absolute_percentage_error(y_test, ensemble_pred) * 100
        r2 = r2_score(y_test, ensemble_pred)
        mae = mean_absolute_error(y_test, ensemble_pred)
        
        print(f"\n{weights['name']}:")
        print(f"  Weights: XGB={weights['xgb']:.1f}, LGB={weights['lgb']:.1f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  R²: {r2:.3f}")
        print(f"  MAE: ${mae:,.2f}")
        
        if mape < best_mape:
            best_mape = mape
            best_ensemble = ensemble_pred
            best_weights = weights
    
    return best_ensemble, best_weights, best_mape

def calculate_direction_accuracy(y_true, y_pred):
    """Calculate direction accuracy"""
    y_true_diff = np.diff(y_true)
    y_pred_diff = np.diff(y_pred)
    
    correct_direction = np.sum((y_true_diff * y_pred_diff) > 0)
    total = len(y_true_diff)
    
    return (correct_direction / total) * 100

def main():
    print("="*70)
    print("TREE-BASED ENSEMBLE MODEL")
    print("="*70)
    
    # Load data
    df = load_data('data/featured.csv')
    
    # Prepare features
    X, y, feature_cols = prepare_features(df)
    
    # Split data (75/25 for time series)
    split_idx = int(len(X) * 0.75)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\nTrain samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train models
    xgb_model, xgb_pred = train_xgboost(X_train, y_train, X_test, y_test)
    lgb_model, lgb_pred = train_lightgbm(X_train, y_train, X_test, y_test)
    
    # Create ensemble
    predictions = {
        'xgb': xgb_pred,
        'lgb': lgb_pred
    }
    
    ensemble_pred, best_weights, best_mape = create_ensemble(predictions, y_test)
    
    # Calculate final metrics
    print("\n" + "="*70)
    print("FINAL ENSEMBLE PERFORMANCE")
    print("="*70)
    
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    direction_acc = calculate_direction_accuracy(y_test.values, ensemble_pred)
    
    print(f"\nBest Configuration: {best_weights['name']}")
    print(f"Weights: XGB={best_weights['xgb']:.1f}, LGB={best_weights['lgb']:.1f}")
    print(f"\nPerformance Metrics:")
    print(f"  MAPE: {best_mape:.2f}%")
    print(f"  R²: {ensemble_r2:.3f}")
    print(f"  MAE: ${ensemble_mae:,.2f}")
    print(f"  Direction Accuracy: {direction_acc:.1f}%")
    
    # Compare with baseline
    print("\n" + "="*70)
    print("IMPROVEMENT OVER BASELINE")
    print("="*70)
    
    baseline_mape = 19.3
    baseline_r2 = 0.840
    
    mape_improvement = ((baseline_mape - best_mape) / baseline_mape) * 100
    r2_improvement = ((ensemble_r2 - baseline_r2) / baseline_r2) * 100
    
    print(f"\nBaseline (Prophet + LSTM):")
    print(f"  MAPE: {baseline_mape:.1f}%")
    print(f"  R²: {baseline_r2:.3f}")
    
    print(f"\nNew Ensemble (XGBoost + LightGBM):")
    print(f"  MAPE: {best_mape:.2f}% (↓ {baseline_mape - best_mape:.1f} points)")
    print(f"  R²: {ensemble_r2:.3f} (↑ {ensemble_r2 - baseline_r2:.3f})")
    
    print(f"\nRelative Improvement:")
    print(f"  MAPE: {mape_improvement:.1f}% better")
    print(f"  R²: {r2_improvement:.1f}% better")
    
    # Save results
    results_dir = Path('results/tree_ensemble')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'model': 'XGBoost + LightGBM Ensemble',
        'weights': best_weights,
        'metrics': {
            'mape': float(best_mape),
            'r2': float(ensemble_r2),
            'mae': float(ensemble_mae),
            'direction_accuracy': float(direction_acc)
        },
        'baseline_comparison': {
            'baseline_mape': baseline_mape,
            'baseline_r2': baseline_r2,
            'mape_improvement_pct': float(mape_improvement),
            'r2_improvement_pct': float(r2_improvement)
        },
        'individual_models': {
            'xgboost_mape': float(mean_absolute_percentage_error(y_test, xgb_pred) * 100),
            'lightgbm_mape': float(mean_absolute_percentage_error(y_test, lgb_pred) * 100)
        }
    }
    
    with open(results_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {results_dir / 'results.json'}")
    
    # Feature importance
    print("\n" + "="*70)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("="*70)
    
    # Get feature importance from XGBoost
    importance = xgb_model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("\nXGBoost Feature Importance:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:<30} {row['importance']:.4f}")
    
    feature_importance.to_csv(results_dir / 'feature_importance.csv', index=False)
    print(f"\n✅ Feature importance saved to {results_dir / 'feature_importance.csv'}")
    
    print("\n" + "="*70)
    print("✅ TREE ENSEMBLE TRAINING COMPLETE!")
    print("="*70)
    
    return results

if __name__ == '__main__':
    main()
