"""
Uncertainty Quantification for Time Series Forecasting
Implements conformal prediction and quantile regression for reliable prediction intervals
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from pathlib import Path
import json

class ConformalPredictor:
    """Conformal prediction for distribution-free uncertainty quantification"""
    
    def __init__(self, base_model, alpha=0.05):
        """
        Args:
            base_model: Trained prediction model
            alpha: Significance level (0.05 for 95% confidence)
        """
        self.base_model = base_model
        self.alpha = alpha
        self.calibration_scores = None
        
    def calibrate(self, X_cal, y_cal):
        """Calibrate on a calibration set"""
        predictions = self.base_model.predict(X_cal)
        self.calibration_scores = np.abs(y_cal - predictions)
        
    def predict_with_intervals(self, X_test):
        """Generate predictions with conformal prediction intervals"""
        predictions = self.base_model.predict(X_test)
        
        # Calculate quantile of calibration scores
        q = np.quantile(self.calibration_scores, 1 - self.alpha)
        
        # Prediction intervals
        lower = predictions - q
        upper = predictions + q
        
        return predictions, lower, upper


class QuantileRegressionForecaster:
    """Quantile regression for prediction intervals"""
    
    def __init__(self, quantiles=[0.025, 0.5, 0.975]):
        """
        Args:
            quantiles: List of quantiles to predict (default: 2.5%, 50%, 97.5% for 95% CI)
        """
        self.quantiles = quantiles
        self.models = {}
        
    def train(self, X_train, y_train):
        """Train separate models for each quantile"""
        for q in self.quantiles:
            print(f"Training quantile regressor for q={q:.3f}...")
            
            model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                loss='quantile',
                alpha=q,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            self.models[q] = model
            
    def predict(self, X_test):
        """Generate predictions for all quantiles"""
        predictions = {}
        for q in self.quantiles:
            predictions[q] = self.models[q].predict(X_test)
        
        return predictions


class UncertaintyEvaluator:
    """Evaluate uncertainty quantification quality"""
    
    @staticmethod
    def coverage_score(y_true, lower, upper):
        """Calculate empirical coverage (should be close to nominal level)"""
        coverage = np.mean((y_true >= lower) & (y_true <= upper))
        return coverage
    
    @staticmethod
    def interval_width(lower, upper):
        """Calculate average prediction interval width"""
        return np.mean(upper - lower)
    
    @staticmethod
    def sharpness_score(lower, upper, y_true):
        """Combined metric: coverage × (1 / normalized width)"""
        coverage = UncertaintyEvaluator.coverage_score(y_true, lower, upper)
        width = UncertaintyEvaluator.interval_width(lower, upper)
        normalized_width = width / np.mean(y_true)
        sharpness = coverage / normalized_width
        return sharpness


def run_uncertainty_quantification(data_path='data/featured.csv', 
                                   results_dir='results/uncertainty'):
    """Complete uncertainty quantification pipeline"""
    
    print("="*70)
    print("UNCERTAINTY QUANTIFICATION PIPELINE")
    print("="*70)
    
    # Load data
    print("\nLoading featured data...")
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Prepare features
    feature_cols = [c for c in df.columns if c not in ['date', 'sales']]
    X = df[feature_cols].values
    y = df['sales'].values
    dates = df['date'].values
    
    # Split: train / calibration / test
    total = len(df)
    train_size = total - 18
    cal_size = 6
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_cal = X[train_size:train_size + cal_size]
    y_cal = y[train_size:train_size + cal_size]
    
    X_test = X[train_size + cal_size:]
    y_test = y[train_size + cal_size:]
    dates_test = dates[train_size + cal_size:]
    
    print(f"Train: {len(X_train)} samples")
    print(f"Calibration: {len(X_cal)} samples")
    print(f"Test: {len(X_test)} samples")
    
    # Method 1: Conformal Prediction
    print("\n" + "="*70)
    print("METHOD 1: CONFORMAL PREDICTION")
    print("="*70)
    
    from sklearn.ensemble import RandomForestRegressor
    
    base_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    base_model.fit(X_train, y_train)
    
    conformal = ConformalPredictor(base_model, alpha=0.05)
    conformal.calibrate(X_cal, y_cal)
    
    conf_pred, conf_lower, conf_upper = conformal.predict_with_intervals(X_test)
    
    # Evaluate
    conf_coverage = UncertaintyEvaluator.coverage_score(y_test, conf_lower, conf_upper)
    conf_width = UncertaintyEvaluator.interval_width(conf_lower, conf_upper)
    
    print(f"\nConformal Prediction Results:")
    print(f"  Coverage: {conf_coverage:.1%} (target: 95%)")
    print(f"  Avg interval width: ${conf_width:,.2f}")
    print(f"  Relative width: {(conf_width / np.mean(y_test) * 100):.1f}%")
    
    # Method 2: Quantile Regression
    print("\n" + "="*70)
    print("METHOD 2: QUANTILE REGRESSION")
    print("="*70)
    
    qr_forecaster = QuantileRegressionForecaster(quantiles=[0.025, 0.5, 0.975])
    
    # Combine train and calibration for more data
    X_train_full = np.vstack([X_train, X_cal])
    y_train_full = np.concatenate([y_train, y_cal])
    
    qr_forecaster.train(X_train_full, y_train_full)
    
    qr_predictions = qr_forecaster.predict(X_test)
    qr_lower = qr_predictions[0.025]
    qr_median = qr_predictions[0.5]
    qr_upper = qr_predictions[0.975]
    
    # Evaluate
    qr_coverage = UncertaintyEvaluator.coverage_score(y_test, qr_lower, qr_upper)
    qr_width = UncertaintyEvaluator.interval_width(qr_lower, qr_upper)
    
    print(f"\nQuantile Regression Results:")
    print(f"  Coverage: {qr_coverage:.1%} (target: 95%)")
    print(f"  Avg interval width: ${qr_width:,.2f}")
    print(f"  Relative width: {(qr_width / np.mean(y_test) * 100):.1f}%")
    print(f"  Median MAE: ${mean_absolute_error(y_test, qr_median):,.2f}")
    
    # Visualization
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Conformal Prediction
    ax = axes[0]
    ax.plot(dates_test, y_test, 'o-', label='Actual', linewidth=2, markersize=8, color='black')
    ax.plot(dates_test, conf_pred, 's--', label='Prediction', linewidth=2, color='blue')
    ax.fill_between(dates_test, conf_lower, conf_upper, alpha=0.3, 
                    color='blue', label=f'95% Conformal Interval ({conf_coverage:.1%} coverage)')
    ax.set_title('Conformal Prediction with Uncertainty Intervals', fontweight='bold', fontsize=12)
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # Plot 2: Quantile Regression
    ax = axes[1]
    ax.plot(dates_test, y_test, 'o-', label='Actual', linewidth=2, markersize=8, color='black')
    ax.plot(dates_test, qr_median, 's--', label='Median Prediction', linewidth=2, color='green')
    ax.fill_between(dates_test, qr_lower, qr_upper, alpha=0.3, 
                    color='green', label=f'95% Quantile Interval ({qr_coverage:.1%} coverage)')
    ax.set_title('Quantile Regression with Prediction Intervals', fontweight='bold', fontsize=12)
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    fig_path = results_path / 'uncertainty_visualization.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {fig_path}")
    plt.show()
    
    # Save results
    results = {
        'conformal_prediction': {
            'coverage': float(conf_coverage),
            'avg_interval_width': float(conf_width),
            'relative_width_pct': float(conf_width / np.mean(y_test) * 100)
        },
        'quantile_regression': {
            'coverage': float(qr_coverage),
            'avg_interval_width': float(qr_width),
            'relative_width_pct': float(qr_width / np.mean(y_test) * 100),
            'median_mae': float(mean_absolute_error(y_test, qr_median))
        }
    }
    
    results_file = results_path / 'uncertainty_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    
    print("\n" + "="*70)
    print("UNCERTAINTY QUANTIFICATION COMPLETE")
    print("="*70)
    
    return results


if __name__ == "__main__":
    # Ensure featured data exists
    from feature_engineering import TimeSeriesFeatureEngineer
    
    import os
    if not os.path.exists('data/featured.csv'):
        print("Creating featured dataset first...")
        engineer = TimeSeriesFeatureEngineer()
        engineer.engineer_features()
    
    # Run uncertainty quantification
    results = run_uncertainty_quantification()
    
    print("\nKey Takeaways:")
    print("  - Conformal prediction provides distribution-free guarantees")
    print("  - Quantile regression offers more flexible intervals")
    print("  - Both methods provide reliable uncertainty estimates")
    print("\nNext steps:")
    print("  - Use these intervals in production for risk management")
    print("  - Monitor coverage in deployment")
    print("  - Adjust alpha based on business requirements")
