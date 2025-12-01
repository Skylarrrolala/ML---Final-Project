"""
Comprehensive Baseline Evaluation Pipeline
Reproduces current model performance and establishes improvement baseline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score,
    mean_absolute_percentage_error
)
import tensorflow as tf
from tensorflow import keras
import warnings
import json
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

class BaselineEvaluator:
    """Comprehensive evaluation of current Prophet + LSTM ensemble"""
    
    def __init__(self, data_path='data/cleaned.csv', results_dir='results/metrics'):
        self.data_path = data_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Model components
        self.prophet_model = None
        self.lstm_model = None
        self.scaler = MinMaxScaler()
        
        # Weights
        self.prophet_weight = 0.6
        self.lstm_weight = 0.4
        
        # Results storage
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {},
            'predictions': {},
            'config': {
                'prophet_weight': self.prophet_weight,
                'lstm_weight': self.lstm_weight,
                'sequence_length': 12
            }
        }
    
    def load_and_prepare_data(self):
        """Load data and create train/test split"""
        print("Loading and preparing data...")
        
        df = pd.read_csv(self.data_path)
        df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')
        
        # Aggregate to monthly
        monthly_sales = df.groupby(pd.Grouper(key='Order Date', freq='M'))['Sales'].sum()
        
        # Train/test split (last 12 months for test)
        train_size = len(monthly_sales) - 12
        self.train_data = monthly_sales[:train_size]
        self.test_data = monthly_sales[train_size:]
        
        print(f"Training samples: {len(self.train_data)} months")
        print(f"Test samples: {len(self.test_data)} months")
        print(f"Date range: {monthly_sales.index.min()} to {monthly_sales.index.max()}")
        
        return monthly_sales
    
    def train_prophet(self):
        """Train Prophet model on training data"""
        print("\nTraining Prophet model...")
        
        prophet_df = self.train_data.reset_index()
        prophet_df.columns = ['ds', 'y']
        
        self.prophet_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05
        )
        self.prophet_model.fit(prophet_df)
        print("Prophet training complete")
    
    def train_lstm(self, seq_length=12):
        """Train LSTM model on training data"""
        print("\nTraining LSTM model...")
        
        # Scale data
        scaled_train = self.scaler.fit_transform(
            self.train_data.values.reshape(-1, 1)
        )
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_train) - seq_length):
            X.append(scaled_train[i:(i + seq_length)])
            y.append(scaled_train[i + seq_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # Build LSTM model
        self.lstm_model = keras.Sequential([
            keras.layers.LSTM(50, activation='relu', input_shape=(seq_length, 1)),
            keras.layers.Dense(1)
        ])
        
        self.lstm_model.compile(optimizer='adam', loss='mse')
        
        # Train
        self.lstm_model.fit(
            X, y, 
            epochs=100, 
            batch_size=32, 
            verbose=0,
            validation_split=0.1
        )
        
        print("LSTM training complete")
        return seq_length
    
    def predict_prophet(self, test_dates):
        """Generate Prophet predictions"""
        test_df = pd.DataFrame({'ds': test_dates})
        forecast = self.prophet_model.predict(test_df)
        return forecast['yhat'].values, forecast[['yhat_lower', 'yhat_upper']].values
    
    def predict_lstm(self, monthly_sales, seq_length=12):
        """Generate LSTM predictions on test set"""
        scaled_all = self.scaler.transform(monthly_sales.values.reshape(-1, 1))
        
        train_size = len(self.train_data)
        lstm_preds = []
        
        for i in range(len(self.test_data)):
            start_idx = train_size - seq_length + i
            sequence = scaled_all[start_idx:start_idx + seq_length]
            sequence = sequence.reshape(1, seq_length, 1)
            
            pred = self.lstm_model.predict(sequence, verbose=0)
            lstm_preds.append(pred[0, 0])
        
        # Inverse transform
        lstm_preds = self.scaler.inverse_transform(
            np.array(lstm_preds).reshape(-1, 1)
        ).flatten()
        
        return lstm_preds
    
    def create_ensemble(self, prophet_preds, lstm_preds):
        """Create weighted ensemble predictions"""
        ensemble = (self.prophet_weight * prophet_preds + 
                   self.lstm_weight * lstm_preds)
        return ensemble
    
    def calculate_metrics(self, y_true, y_pred, model_name="Model"):
        """Calculate comprehensive metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Direction accuracy
        if len(y_true) > 1:
            y_true_diff = np.diff(y_true)
            y_pred_diff = np.diff(y_pred)
            direction_acc = np.mean((y_true_diff * y_pred_diff) > 0) * 100
        else:
            direction_acc = 0
        
        return {
            'model': model_name,
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'r2': float(r2),
            'mape': float(mape),
            'direction_accuracy': float(direction_acc)
        }
    
    def evaluate(self):
        """Run complete evaluation pipeline"""
        print("="*70)
        print("BASELINE MODEL EVALUATION")
        print("="*70)
        
        # Load data
        monthly_sales = self.load_and_prepare_data()
        
        # Train models
        self.train_prophet()
        seq_length = self.train_lstm()
        
        # Generate predictions
        print("\nGenerating predictions on test set...")
        prophet_preds, prophet_intervals = self.predict_prophet(self.test_data.index)
        lstm_preds = self.predict_lstm(monthly_sales, seq_length)
        ensemble_preds = self.create_ensemble(prophet_preds, lstm_preds)
        
        # Calculate metrics
        y_true = self.test_data.values
        
        print("\n" + "="*70)
        print("PERFORMANCE METRICS")
        print("="*70)
        
        metrics_prophet = self.calculate_metrics(y_true, prophet_preds, "Prophet")
        metrics_lstm = self.calculate_metrics(y_true, lstm_preds, "LSTM")
        metrics_ensemble = self.calculate_metrics(y_true, ensemble_preds, "Ensemble")
        
        # Display results
        metrics_df = pd.DataFrame([metrics_prophet, metrics_lstm, metrics_ensemble])
        print("\n" + metrics_df.to_string(index=False))
        
        # Store results
        self.results['metrics'] = {
            'prophet': metrics_prophet,
            'lstm': metrics_lstm,
            'ensemble': metrics_ensemble
        }
        
        self.results['predictions'] = {
            'dates': [d.isoformat() for d in self.test_data.index],
            'actual': y_true.tolist(),
            'prophet': prophet_preds.tolist(),
            'lstm': lstm_preds.tolist(),
            'ensemble': ensemble_preds.tolist(),
            'prophet_lower': prophet_intervals[:, 0].tolist(),
            'prophet_upper': prophet_intervals[:, 1].tolist()
        }
        
        # Visualize
        self.visualize_results(y_true, prophet_preds, lstm_preds, ensemble_preds)
        
        # Save results
        self.save_results()
        
        return self.results
    
    def visualize_results(self, y_true, prophet_preds, lstm_preds, ensemble_preds):
        """Create visualization of predictions vs actuals"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        dates = self.test_data.index
        
        # Plot 1: All models comparison
        ax = axes[0, 0]
        ax.plot(dates, y_true, 'o-', label='Actual', linewidth=2, markersize=8)
        ax.plot(dates, prophet_preds, 's--', label='Prophet', alpha=0.7)
        ax.plot(dates, lstm_preds, '^--', label='LSTM', alpha=0.7)
        ax.plot(dates, ensemble_preds, 'D-', label='Ensemble', linewidth=2, color='red')
        ax.set_title('Model Predictions Comparison', fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sales ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        # Plot 2: Residuals
        ax = axes[0, 1]
        residuals = y_true - ensemble_preds
        ax.plot(dates, residuals, 'o-', color='purple', linewidth=2)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.fill_between(dates, -np.std(residuals), np.std(residuals), 
                        alpha=0.2, color='green', label='±1 Std Dev')
        ax.set_title('Ensemble Residuals', fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Residual ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        # Plot 3: Error distribution
        ax = axes[1, 0]
        pct_errors = ((y_true - ensemble_preds) / y_true) * 100
        ax.bar(dates, pct_errors, color=['red' if x < 0 else 'green' for x in pct_errors], 
               alpha=0.6)
        ax.axhline(y=0, color='black', linewidth=1)
        ax.set_title('Percentage Errors', fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Error (%)')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        # Plot 4: Metrics comparison
        ax = axes[1, 1]
        metrics_data = self.results['metrics']
        models = ['Prophet', 'LSTM', 'Ensemble']
        mapes = [metrics_data['prophet']['mape'], 
                metrics_data['lstm']['mape'], 
                metrics_data['ensemble']['mape']]
        
        bars = ax.bar(models, mapes, color=['blue', 'orange', 'red'], alpha=0.7)
        ax.set_title('MAPE Comparison', fontweight='bold')
        ax.set_ylabel('MAPE (%)')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        fig_path = self.results_dir / 'baseline_evaluation.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {fig_path}")
        plt.show()
    
    def save_results(self):
        """Save evaluation results to JSON"""
        output_path = self.results_dir / 'baseline_results.json'
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        
        # Also save summary
        summary_path = self.results_dir / 'baseline_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("BASELINE MODEL EVALUATION SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Timestamp: {self.results['timestamp']}\n\n")
            
            f.write("METRICS:\n")
            f.write("-" * 70 + "\n")
            for model_name, metrics in self.results['metrics'].items():
                f.write(f"\n{model_name.upper()}:\n")
                f.write(f"  MAPE: {metrics['mape']:.2f}%\n")
                f.write(f"  RMSE: ${metrics['rmse']:,.2f}\n")
                f.write(f"  MAE: ${metrics['mae']:,.2f}\n")
                f.write(f"  R²: {metrics['r2']:.4f}\n")
                f.write(f"  Direction Accuracy: {metrics['direction_accuracy']:.1f}%\n")
        
        print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    evaluator = BaselineEvaluator()
    results = evaluator.evaluate()
    
    print("\n" + "="*70)
    print("BASELINE EVALUATION COMPLETE")
    print("="*70)
    print(f"\nEnsemble MAPE: {results['metrics']['ensemble']['mape']:.2f}%")
    print(f"Ensemble R²: {results['metrics']['ensemble']['r2']:.4f}")
    print("\nNext steps:")
    print("  1. Review baseline_results.json for detailed metrics")
    print("  2. Run feature engineering improvements")
    print("  3. Execute hyperparameter optimization")
