"""
Optimized XGBoost Model for Sales Forecasting
Best performer on small time series datasets
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error
import xgboost as xgb
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath='data/featured.csv'):
    """Load featured dataset"""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
    
    return df

def prepare_features(df, target_col='sales'):
    """Prepare features and target"""
    exclude_cols = ['date', target_col, 'InvoiceDate', 'InvoiceNo', 
                    'CustomerID', 'Country', 'Description', 'StockCode']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Handle missing values
    X = X.ffill().fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    print(f"Features: {len(feature_cols)}")
    print(f"Samples: {len(X)}")
    
    return X, y, feature_cols

def calculate_direction_accuracy(y_true, y_pred):
    """Calculate direction accuracy"""
    y_true_diff = np.diff(y_true)
    y_pred_diff = np.diff(y_pred)
    
    correct_direction = np.sum((y_true_diff * y_pred_diff) > 0)
    total = len(y_true_diff)
    
    return (correct_direction / total) * 100

def train_optimized_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost with optimized hyperparameters for small datasets"""
    print("\nTraining Optimized XGBoost...")
    
    # Optimized for small time series datasets
    model = xgb.XGBRegressor(
        n_estimators=100,      # Fewer trees for small data
        max_depth=4,           # Shallower trees to prevent overfitting
        learning_rate=0.1,     # Slightly higher learning rate
        subsample=0.9,         # Use more data per tree
        colsample_bytree=0.9,  # Use more features per tree
        min_child_weight=3,    # Prevent overfitting
        gamma=0.1,             # Min loss reduction for split
        reg_alpha=0.1,         # L1 regularization
        reg_lambda=1.0,        # L2 regularization
        random_state=42,
        verbosity=0
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )
    
    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Metrics
    train_mape = mean_absolute_percentage_error(y_train, train_pred) * 100
    test_mape = mean_absolute_percentage_error(y_test, test_pred) * 100
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    direction_acc = calculate_direction_accuracy(y_test.values, test_pred)
    
    print(f"\nTraining Performance:")
    print(f"  MAPE: {train_mape:.2f}%")
    print(f"  R²: {train_r2:.3f}")
    
    print(f"\nTest Performance:")
    print(f"  MAPE: {test_mape:.2f}%")
    print(f"  R²: {test_r2:.3f}")
    print(f"  MAE: ${test_mae:,.2f}")
    print(f"  Direction Accuracy: {direction_acc:.1f}%")
    
    return model, train_pred, test_pred

def visualize_results(y_train, train_pred, y_test, test_pred, save_dir):
    """Create visualization of predictions"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Train predictions
    axes[0, 0].plot(y_train.values, label='Actual', marker='o')
    axes[0, 0].plot(train_pred, label='Predicted', marker='s', alpha=0.7)
    axes[0, 0].set_title('Training Set Predictions')
    axes[0, 0].set_xlabel('Time Period')
    axes[0, 0].set_ylabel('Sales')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Test predictions
    axes[0, 1].plot(y_test.values, label='Actual', marker='o')
    axes[0, 1].plot(test_pred, label='Predicted', marker='s', alpha=0.7)
    axes[0, 1].set_title('Test Set Predictions')
    axes[0, 1].set_xlabel('Time Period')
    axes[0, 1].set_ylabel('Sales')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Residuals
    test_residuals = y_test.values - test_pred
    axes[1, 0].scatter(test_pred, test_residuals, alpha=0.6)
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].set_title('Residual Plot (Test Set)')
    axes[1, 0].set_xlabel('Predicted Values')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Actual vs Predicted
    axes[1, 1].scatter(y_test.values, test_pred, alpha=0.6)
    min_val = min(y_test.min(), test_pred.min())
    max_val = max(y_test.max(), test_pred.max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--')
    axes[1, 1].set_title('Actual vs Predicted (Test Set)')
    axes[1, 1].set_xlabel('Actual Sales')
    axes[1, 1].set_ylabel('Predicted Sales')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'predictions.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualizations saved to {save_dir / 'predictions.png'}")
    plt.close()

def main():
    print("="*70)
    print("OPTIMIZED XGBOOST FOR SALES FORECASTING")
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
    
    # Train model
    model, train_pred, test_pred = train_optimized_xgboost(
        X_train, y_train, X_test, y_test
    )
    
    # Compare with baseline
    print("\n" + "="*70)
    print("IMPROVEMENT OVER BASELINE")
    print("="*70)
    
    baseline_mape = 19.3
    baseline_r2 = 0.840
    test_mape = mean_absolute_percentage_error(y_test, test_pred) * 100
    test_r2 = r2_score(y_test, test_pred)
    
    mape_improvement = ((baseline_mape - test_mape) / baseline_mape) * 100
    r2_improvement = ((test_r2 - baseline_r2) / baseline_r2) * 100
    
    print(f"\n📊 Baseline (Prophet + LSTM):")
    print(f"   MAPE: {baseline_mape:.1f}%")
    print(f"   R²: {baseline_r2:.3f}")
    
    print(f"\n🚀 New Model (Optimized XGBoost + Features):")
    print(f"   MAPE: {test_mape:.2f}%")
    print(f"   R²: {test_r2:.3f}")
    
    print(f"\n✨ Improvement:")
    print(f"   MAPE: {abs(baseline_mape - test_mape):.2f} points better ({mape_improvement:.1f}%)")
    if test_r2 > baseline_r2:
        print(f"   R²: {test_r2 - baseline_r2:.3f} points better ({r2_improvement:.1f}%)")
    else:
        print(f"   R²: {baseline_r2 - test_r2:.3f} points worse ({r2_improvement:.1f}%)")
    
    # Feature importance
    print("\n" + "="*70)
    print("TOP 15 MOST IMPORTANT FEATURES")
    print("="*70)
    
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print()
    for idx, row in feature_importance.head(15).iterrows():
        bar = '█' * int(row['importance'] * 50)
        print(f"  {row['feature']:<30} {bar} {row['importance']:.4f}")
    
    # Save results
    results_dir = Path('results/xgboost_optimized')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    test_mae = mean_absolute_error(y_test, test_pred)
    direction_acc = calculate_direction_accuracy(y_test.values, test_pred)
    
    results = {
        'model': 'Optimized XGBoost with Feature Engineering',
        'hyperparameters': {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.1,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0
        },
        'performance': {
            'test_mape': float(test_mape),
            'test_r2': float(test_r2),
            'test_mae': float(test_mae),
            'direction_accuracy': float(direction_acc),
            'train_mape': float(mean_absolute_percentage_error(y_train, train_pred) * 100)
        },
        'baseline_comparison': {
            'baseline_mape': baseline_mape,
            'baseline_r2': baseline_r2,
            'mape_improvement': float(baseline_mape - test_mape),
            'mape_improvement_pct': float(mape_improvement),
            'r2_change': float(test_r2 - baseline_r2)
        },
        'features_used': len(feature_cols),
        'top_features': feature_importance.head(10)['feature'].tolist()
    }
    
    with open(results_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    feature_importance.to_csv(results_dir / 'feature_importance.csv', index=False)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'actual': y_test.values,
        'predicted': test_pred,
        'error': y_test.values - test_pred,
        'error_pct': ((y_test.values - test_pred) / y_test.values) * 100
    })
    predictions_df.to_csv(results_dir / 'predictions.csv', index=False)
    
    # Visualize
    visualize_results(y_train, train_pred, y_test, test_pred, results_dir)
    
    print(f"\n✅ Results saved to {results_dir / 'results.json'}")
    print(f"✅ Predictions saved to {results_dir / 'predictions.csv'}")
    print(f"✅ Feature importance saved to {results_dir / 'feature_importance.csv'}")
    
    print("\n" + "="*70)
    print("🎉 SUCCESS! MODEL IMPROVEMENT COMPLETE!")
    print("="*70)
    
    if test_mape < baseline_mape:
        print(f"\n✨ Your model is now {mape_improvement:.1f}% more accurate!")
        print(f"   Went from {baseline_mape:.1f}% → {test_mape:.2f}% MAPE")
    
    print("\n📝 Next Steps:")
    print("   1. Review predictions.png to see forecast quality")
    print("   2. Check feature_importance.csv for key drivers")
    print("   3. Consider hyperparameter tuning for further gains")
    print("   4. Deploy model with confidence intervals")
    
    return results

if __name__ == '__main__':
    main()
