"""
Hyperparameter Tuning with Optuna for Prophet and LSTM
Tracks experiments with MLflow for reproducibility
"""

import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
import mlflow
import mlflow.sklearn
import warnings
from pathlib import Path
import json

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


class ProphetTuner:
    """Tune Prophet hyperparameters with Optuna"""
    
    def __init__(self, train_data, val_data):
        self.train_data = train_data
        self.val_data = val_data
        self.best_params = None
        self.best_model = None
    
    def objective(self, trial):
        """Optuna objective function for Prophet"""
        
        # Sample hyperparameters
        params = {
            'changepoint_prior_scale': trial.suggest_float(
                'changepoint_prior_scale', 0.001, 0.5, log=True
            ),
            'seasonality_prior_scale': trial.suggest_float(
                'seasonality_prior_scale', 0.01, 10.0, log=True
            ),
            'seasonality_mode': trial.suggest_categorical(
                'seasonality_mode', ['additive', 'multiplicative']
            ),
            'changepoint_range': trial.suggest_float(
                'changepoint_range', 0.8, 0.95
            )
        }
        
        # Prepare data
        prophet_df = self.train_data.reset_index()
        prophet_df.columns = ['ds', 'y']
        
        # Train model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            **params
        )
        model.fit(prophet_df)
        
        # Validate
        val_df = pd.DataFrame({'ds': self.val_data.index})
        forecast = model.predict(val_df)
        
        # Calculate MAPE
        mape = mean_absolute_percentage_error(
            self.val_data.values,
            forecast['yhat'].values
        ) * 100
        
        return mape
    
    def tune(self, n_trials=50):
        """Run hyperparameter tuning"""
        print("Tuning Prophet hyperparameters...")
        
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42)
        )
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params = study.best_params
        
        print(f"\nBest MAPE: {study.best_value:.2f}%")
        print("Best parameters:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")
        
        return self.best_params
    
    def train_best_model(self):
        """Train Prophet with best parameters"""
        prophet_df = self.train_data.reset_index()
        prophet_df.columns = ['ds', 'y']
        
        self.best_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            **self.best_params
        )
        self.best_model.fit(prophet_df)
        
        return self.best_model


class LSTMTuner:
    """Tune LSTM hyperparameters with Optuna"""
    
    def __init__(self, train_data, val_data):
        self.train_data = train_data
        self.val_data = val_data
        self.scaler = MinMaxScaler()
        self.best_params = None
        self.best_model = None
    
    def create_sequences(self, data, seq_length):
        """Create LSTM sequences"""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)
    
    def objective(self, trial):
        """Optuna objective function for LSTM"""
        
        # Sample hyperparameters
        seq_length = trial.suggest_int('seq_length', 6, 18)
        units = trial.suggest_int('units', 32, 128, step=32)
        dropout = trial.suggest_float('dropout', 0.0, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        
        # Prepare data
        scaled_train = self.scaler.fit_transform(
            self.train_data.values.reshape(-1, 1)
        )
        
        X_train, y_train = self.create_sequences(scaled_train, seq_length)
        
        if len(X_train) < 10:
            return float('inf')
        
        # Build model
        model = keras.Sequential([
            keras.layers.LSTM(units, activation='relu', 
                            input_shape=(seq_length, 1), 
                            return_sequences=False),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(1)
        ])
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        
        # Train
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=batch_size,
            validation_split=0.15,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Validate on separate validation set
        # Prepare validation data
        all_data = np.concatenate([
            self.train_data.values,
            self.val_data.values
        ])
        scaled_all = self.scaler.transform(all_data.reshape(-1, 1))
        
        train_size = len(self.train_data)
        val_preds = []
        
        for i in range(len(self.val_data)):
            start_idx = train_size - seq_length + i
            if start_idx < 0:
                continue
            
            sequence = scaled_all[start_idx:start_idx + seq_length]
            if len(sequence) < seq_length:
                continue
            
            sequence = sequence.reshape(1, seq_length, 1)
            pred = model.predict(sequence, verbose=0)
            val_preds.append(pred[0, 0])
        
        if len(val_preds) == 0:
            return float('inf')
        
        # Inverse transform
        val_preds = self.scaler.inverse_transform(
            np.array(val_preds).reshape(-1, 1)
        ).flatten()
        
        # Calculate MAPE
        actual = self.val_data.values[-len(val_preds):]
        mape = np.mean(np.abs((actual - val_preds) / actual)) * 100
        
        # Clear session to prevent memory leak
        keras.backend.clear_session()
        
        return mape
    
    def tune(self, n_trials=30):
        """Run hyperparameter tuning"""
        print("\nTuning LSTM hyperparameters...")
        
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42)
        )
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params = study.best_params
        
        print(f"\nBest MAPE: {study.best_value:.2f}%")
        print("Best parameters:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")
        
        return self.best_params
    
    def train_best_model(self):
        """Train LSTM with best parameters"""
        seq_length = self.best_params['seq_length']
        units = self.best_params['units']
        dropout = self.best_params['dropout']
        learning_rate = self.best_params['learning_rate']
        batch_size = self.best_params['batch_size']
        
        # Prepare data
        scaled_train = self.scaler.fit_transform(
            self.train_data.values.reshape(-1, 1)
        )
        
        X_train, y_train = self.create_sequences(scaled_train, seq_length)
        
        # Build model
        self.best_model = keras.Sequential([
            keras.layers.LSTM(units, activation='relu', 
                            input_shape=(seq_length, 1)),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(1)
        ])
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.best_model.compile(optimizer=optimizer, loss='mse')
        
        # Train
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True
        )
        
        self.best_model.fit(
            X_train, y_train,
            epochs=200,
            batch_size=batch_size,
            validation_split=0.15,
            callbacks=[early_stop],
            verbose=1
        )
        
        return self.best_model, self.scaler


def run_tuning_pipeline(data_path='data/cleaned.csv', 
                       prophet_trials=50, 
                       lstm_trials=30,
                       results_dir='results/tuning'):
    """Complete hyperparameter tuning pipeline"""
    
    print("="*70)
    print("HYPERPARAMETER TUNING PIPELINE")
    print("="*70)
    
    # Setup MLflow
    mlflow.set_experiment("sales_forecasting_tuning")
    
    # Load and split data
    print("\nLoading data...")
    df = pd.read_csv(data_path)
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')
    monthly_sales = df.groupby(pd.Grouper(key='Order Date', freq='M'))['Sales'].sum()
    
    # Train/val/test split
    total_size = len(monthly_sales)
    train_size = total_size - 18  # Leave last 18 months
    val_size = 6  # 6 months for validation
    
    train_data = monthly_sales[:train_size]
    val_data = monthly_sales[train_size:train_size + val_size]
    test_data = monthly_sales[train_size + val_size:]
    
    print(f"Train: {len(train_data)} months")
    print(f"Validation: {len(val_data)} months")
    print(f"Test: {len(test_data)} months")
    
    # Tune Prophet
    with mlflow.start_run(run_name="prophet_tuning"):
        prophet_tuner = ProphetTuner(train_data, val_data)
        prophet_params = prophet_tuner.tune(n_trials=prophet_trials)
        
        mlflow.log_params(prophet_params)
        mlflow.log_param("model_type", "Prophet")
    
    # Tune LSTM
    with mlflow.start_run(run_name="lstm_tuning"):
        lstm_tuner = LSTMTuner(train_data, val_data)
        lstm_params = lstm_tuner.tune(n_trials=lstm_trials)
        
        mlflow.log_params(lstm_params)
        mlflow.log_param("model_type", "LSTM")
    
    # Save results
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        'prophet': prophet_params,
        'lstm': lstm_params
    }
    
    output_file = results_path / 'tuned_hyperparameters.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nTuned hyperparameters saved to: {output_file}")
    print("\n" + "="*70)
    print("TUNING COMPLETE")
    print("="*70)
    
    return results


if __name__ == "__main__":
    results = run_tuning_pipeline(
        prophet_trials=50,
        lstm_trials=30
    )
    
    print("\nNext steps:")
    print("  1. Review MLflow UI: mlflow ui")
    print("  2. Use tuned_hyperparameters.json for final model training")
    print("  3. Evaluate on test set")
