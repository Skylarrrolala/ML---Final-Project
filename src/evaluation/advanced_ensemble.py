"""
Advanced Ensemble Methods: Stacking, Blending, and Meta-Learning
Combines Prophet, LSTM, XGBoost, and LightGBM
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras
import warnings
from pathlib import Path
import json

warnings.filterwarnings('ignore')


class AdvancedEnsemble:
    """Stacking ensemble with multiple base models and meta-learner"""
    
    def __init__(self, featured_data_path='data/featured.csv'):
        self.featured_data_path = featured_data_path
        self.base_models = {}
        self.meta_learner = None
        self.scalers = {}
        
    def load_featured_data(self):
        """Load feature-engineered dataset"""
        df = pd.read_csv(self.featured_data_path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    def prepare_data(self, df, target='sales', test_months=12):
        """Prepare train/test split with features"""
        
        # Separate features and target
        feature_cols = [c for c in df.columns if c not in ['date', target]]
        
        X = df[feature_cols].values
        y = df[target].values
        dates = df['date'].values
        
        # Train/test split
        train_size = len(df) - test_months
        
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        dates_train, dates_test = dates[:train_size], dates[train_size:]
        
        return X_train, X_test, y_train, y_test, dates_train, dates_test, feature_cols
    
    def train_prophet(self, dates_train, y_train, tuned_params=None):
        """Train Prophet model"""
        print("Training Prophet...")
        
        prophet_df = pd.DataFrame({
            'ds': dates_train,
            'y': y_train
        })
        
        if tuned_params is None:
            tuned_params = {
                'yearly_seasonality': True,
                'weekly_seasonality': False,
                'daily_seasonality': False,
                'seasonality_mode': 'multiplicative'
            }
        
        model = Prophet(**tuned_params)
        model.fit(prophet_df)
        
        self.base_models['prophet'] = model
        return model
    
    def train_lstm(self, y_train, tuned_params=None, seq_length=12):
        """Train LSTM model"""
        print("Training LSTM...")
        
        # Scale data
        scaler = MinMaxScaler()
        scaled_train = scaler.fit_transform(y_train.reshape(-1, 1))
        self.scalers['lstm'] = scaler
        
        # Create sequences
        X_seq, y_seq = [], []
        for i in range(len(scaled_train) - seq_length):
            X_seq.append(scaled_train[i:(i + seq_length)])
            y_seq.append(scaled_train[i + seq_length])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # Default params if not tuned
        if tuned_params is None:
            tuned_params = {
                'units': 64,
                'dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32
            }
        
        # Build model
        model = keras.Sequential([
            keras.layers.LSTM(tuned_params['units'], activation='relu',
                            input_shape=(seq_length, 1)),
            keras.layers.Dropout(tuned_params['dropout']),
            keras.layers.Dense(1)
        ])
        
        optimizer = keras.optimizers.Adam(learning_rate=tuned_params['learning_rate'])
        model.compile(optimizer=optimizer, loss='mse')
        
        # Train
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True
        )
        
        model.fit(
            X_seq, y_seq,
            epochs=150,
            batch_size=tuned_params['batch_size'],
            validation_split=0.15,
            callbacks=[early_stop],
            verbose=0
        )
        
        self.base_models['lstm'] = model
        self.base_models['lstm_seq_length'] = seq_length
        return model
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model"""
        print("Training XGBoost...")
        
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train, verbose=False)
        self.base_models['xgboost'] = model
        return model
    
    def train_lightgbm(self, X_train, y_train):
        """Train LightGBM model"""
        print("Training LightGBM...")
        
        model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        self.base_models['lightgbm'] = model
        return model
    
    def generate_base_predictions(self, X_test, dates_test, y_full):
        """Generate predictions from all base models"""
        predictions = {}
        
        # Prophet
        if 'prophet' in self.base_models:
            prophet_df = pd.DataFrame({'ds': dates_test})
            forecast = self.base_models['prophet'].predict(prophet_df)
            predictions['prophet'] = forecast['yhat'].values
        
        # LSTM
        if 'lstm' in self.base_models:
            seq_length = self.base_models['lstm_seq_length']
            scaler = self.scalers['lstm']
            
            scaled_all = scaler.transform(y_full.reshape(-1, 1))
            train_size = len(y_full) - len(dates_test)
            
            lstm_preds = []
            for i in range(len(dates_test)):
                start_idx = train_size - seq_length + i
                sequence = scaled_all[start_idx:start_idx + seq_length]
                sequence = sequence.reshape(1, seq_length, 1)
                
                pred = self.base_models['lstm'].predict(sequence, verbose=0)
                lstm_preds.append(pred[0, 0])
            
            lstm_preds = scaler.inverse_transform(
                np.array(lstm_preds).reshape(-1, 1)
            ).flatten()
            predictions['lstm'] = lstm_preds
        
        # XGBoost
        if 'xgboost' in self.base_models:
            predictions['xgboost'] = self.base_models['xgboost'].predict(X_test)
        
        # LightGBM
        if 'lightgbm' in self.base_models:
            predictions['lightgbm'] = self.base_models['lightgbm'].predict(X_test)
        
        return predictions
    
    def train_meta_learner(self, base_preds_train, y_train, method='ridge'):
        """Train meta-learner on base model predictions"""
        print(f"\nTraining meta-learner ({method})...")
        
        # Stack predictions
        X_meta = np.column_stack([base_preds_train[name] for name in sorted(base_preds_train.keys())])
        
        # Train meta-learner
        if method == 'ridge':
            meta_model = Ridge(alpha=1.0)
        elif method == 'lasso':
            meta_model = Lasso(alpha=0.1)
        elif method == 'rf':
            meta_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        meta_model.fit(X_meta, y_train)
        self.meta_learner = meta_model
        
        # Print feature importance (weights)
        if hasattr(meta_model, 'coef_'):
            print("\nMeta-learner weights:")
            for name, weight in zip(sorted(base_preds_train.keys()), meta_model.coef_):
                print(f"  {name}: {weight:.4f}")
        
        return meta_model
    
    def predict_ensemble(self, base_predictions):
        """Generate ensemble predictions using meta-learner"""
        X_meta = np.column_stack([base_predictions[name] for name in sorted(base_predictions.keys())])
        ensemble_preds = self.meta_learner.predict(X_meta)
        return ensemble_preds
    
    def evaluate(self, y_true, predictions_dict):
        """Evaluate all models"""
        results = {}
        
        for name, preds in predictions_dict.items():
            mape = mean_absolute_percentage_error(y_true, preds) * 100
            mae = mean_absolute_error(y_true, preds)
            r2 = r2_score(y_true, preds)
            
            results[name] = {
                'mape': float(mape),
                'mae': float(mae),
                'r2': float(r2)
            }
        
        return results
    
    def run_stacking_pipeline(self, test_months=12, meta_method='ridge'):
        """Complete stacking ensemble pipeline"""
        
        print("="*70)
        print("STACKING ENSEMBLE PIPELINE")
        print("="*70)
        
        # Load data
        df = self.load_featured_data()
        X_train, X_test, y_train, y_test, dates_train, dates_test, feature_cols = \
            self.prepare_data(df, test_months=test_months)
        
        print(f"\nFeatures: {len(feature_cols)}")
        print(f"Train samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # Train base models
        print("\n" + "="*70)
        print("TRAINING BASE MODELS")
        print("="*70)
        
        self.train_prophet(dates_train, y_train)
        self.train_lstm(y_train)
        self.train_xgboost(X_train, y_train)
        self.train_lightgbm(X_train, y_train)
        
        # Generate out-of-fold predictions for meta-learner training
        # For simplicity, we'll use validation set approach
        val_size = 6
        train_size_meta = len(y_train) - val_size
        
        y_train_meta = y_train[:train_size_meta]
        y_val_meta = y_train[train_size_meta:]
        
        X_train_meta = X_train[:train_size_meta]
        X_val_meta = X_train[train_size_meta:]
        
        dates_train_meta = dates_train[:train_size_meta]
        dates_val_meta = dates_train[train_size_meta:]
        
        # Re-train base models on smaller training set
        print("\nRe-training base models for meta-learner...")
        temp_models = {}
        
        # Prophet
        prophet_df_meta = pd.DataFrame({'ds': dates_train_meta, 'y': y_train_meta})
        prophet_meta = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                              daily_seasonality=False, seasonality_mode='multiplicative')
        prophet_meta.fit(prophet_df_meta)
        temp_models['prophet'] = prophet_meta
        
        # Get validation predictions
        val_preds_meta = {}
        
        prophet_val_df = pd.DataFrame({'ds': dates_val_meta})
        val_preds_meta['prophet'] = prophet_meta.predict(prophet_val_df)['yhat'].values
        
        # For XGBoost and LightGBM
        xgb_meta = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
        xgb_meta.fit(X_train_meta, y_train_meta, verbose=False)
        val_preds_meta['xgboost'] = xgb_meta.predict(X_val_meta)
        
        lgb_meta = lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8, random_state=42,
                                     n_jobs=-1, verbose=-1)
        lgb_meta.fit(X_train_meta, y_train_meta)
        val_preds_meta['lightgbm'] = lgb_meta.predict(X_val_meta)
        
        # Train meta-learner
        self.train_meta_learner(val_preds_meta, y_val_meta, method=meta_method)
        
        # Generate test predictions from final base models
        print("\n" + "="*70)
        print("GENERATING TEST PREDICTIONS")
        print("="*70)
        
        y_full = np.concatenate([y_train, y_test])
        base_preds_test = self.generate_base_predictions(X_test, dates_test, y_full)
        
        # Ensemble prediction
        ensemble_preds = self.predict_ensemble(base_preds_test)
        
        # Add ensemble to predictions
        all_preds = {**base_preds_test, 'ensemble_stacking': ensemble_preds}
        
        # Evaluate
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        
        results = self.evaluate(y_test, all_preds)
        
        # Display
        results_df = pd.DataFrame(results).T
        results_df = results_df.sort_values('mape')
        print("\n" + results_df.to_string())
        
        # Save results
        output_dir = Path('results/ensemble')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'stacking_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n\nResults saved to: {output_dir / 'stacking_results.json'}")
        
        return results, all_preds, y_test


if __name__ == "__main__":
    # First, make sure featured data exists
    from feature_engineering import TimeSeriesFeatureEngineer
    
    print("Step 1: Creating featured dataset...")
    engineer = TimeSeriesFeatureEngineer()
    engineer.engineer_features()
    
    # Run stacking ensemble
    print("\n\nStep 2: Training stacking ensemble...")
    ensemble = AdvancedEnsemble()
    results, predictions, y_test = ensemble.run_stacking_pipeline(meta_method='ridge')
    
    print("\n" + "="*70)
    print("STACKING ENSEMBLE COMPLETE")
    print("="*70)
    print(f"\nBest Model: {min(results.items(), key=lambda x: x[1]['mape'])[0]}")
    print(f"Best MAPE: {min(r['mape'] for r in results.values()):.2f}%")
