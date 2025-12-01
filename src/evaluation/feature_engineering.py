"""
Advanced Feature Engineering for Time Series Forecasting
Adds lag features, rolling statistics, date features, and more
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class TimeSeriesFeatureEngineer:
    """Create advanced features for time series forecasting"""
    
    def __init__(self, data_path='data/cleaned.csv'):
        self.data_path = data_path
        self.scaler = StandardScaler()
    
    def load_data(self):
        """Load and aggregate data to monthly level"""
        print("Loading data...")
        df = pd.read_csv(self.data_path)
        df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')
        
        # Aggregate to monthly with additional stats
        monthly = df.groupby(pd.Grouper(key='Order Date', freq='M')).agg({
            'Sales': ['sum', 'mean', 'std', 'count'],
            'Row ID': 'count'  # transaction count
        })
        
        monthly.columns = ['sales_sum', 'sales_mean', 'sales_std', 
                          'sales_count', 'transaction_count']
        monthly = monthly.reset_index()
        monthly.columns = ['date', 'sales', 'avg_order_value', 'sales_volatility',
                          'num_orders', 'num_transactions']
        
        # Fill missing volatility with 0
        monthly['sales_volatility'] = monthly['sales_volatility'].fillna(0)
        
        return monthly
    
    def add_lag_features(self, df, lags=[1, 2, 3, 6, 12]):
        """Add lagged sales features"""
        print(f"Adding lag features: {lags}")
        
        for lag in lags:
            df[f'sales_lag_{lag}'] = df['sales'].shift(lag)
        
        return df
    
    def add_rolling_features(self, df, windows=[3, 6, 12]):
        """Add rolling statistics features"""
        print(f"Adding rolling features with windows: {windows}")
        
        for window in windows:
            # Rolling mean
            df[f'sales_rolling_mean_{window}'] = (
                df['sales'].rolling(window=window, min_periods=1).mean()
            )
            
            # Rolling std
            df[f'sales_rolling_std_{window}'] = (
                df['sales'].rolling(window=window, min_periods=1).std().fillna(0)
            )
            
            # Rolling min/max
            df[f'sales_rolling_min_{window}'] = (
                df['sales'].rolling(window=window, min_periods=1).min()
            )
            df[f'sales_rolling_max_{window}'] = (
                df['sales'].rolling(window=window, min_periods=1).max()
            )
        
        return df
    
    def add_date_features(self, df):
        """Extract date-based features"""
        print("Adding date features...")
        
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year_month'] = df['date'].dt.year * 100 + df['date'].dt.month
        
        # Cyclical encoding for month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Cyclical encoding for quarter
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
        
        # Time index (months since start)
        df['time_index'] = np.arange(len(df))
        
        return df
    
    def add_growth_features(self, df):
        """Add growth and momentum features"""
        print("Adding growth features...")
        
        # Month-over-month growth
        df['mom_growth'] = df['sales'].pct_change()
        
        # Year-over-year growth (12 months)
        df['yoy_growth'] = df['sales'].pct_change(periods=12)
        
        # Moving average convergence divergence (MACD-like)
        ema_12 = df['sales'].ewm(span=12, adjust=False).mean()
        ema_26 = df['sales'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        
        # Momentum (rate of change)
        df['momentum_3'] = df['sales'] - df['sales'].shift(3)
        df['momentum_6'] = df['sales'] - df['sales'].shift(6)
        
        return df
    
    def add_statistical_features(self, df):
        """Add statistical features"""
        print("Adding statistical features...")
        
        # Difference from rolling mean
        for window in [3, 6, 12]:
            df[f'diff_from_mean_{window}'] = (
                df['sales'] - df[f'sales_rolling_mean_{window}']
            )
        
        # Z-score (standardized sales)
        df['sales_zscore'] = (
            (df['sales'] - df['sales'].mean()) / df['sales'].std()
        )
        
        # Percentile rank
        df['sales_percentile'] = df['sales'].rank(pct=True)
        
        return df
    
    def add_interaction_features(self, df):
        """Add interaction features"""
        print("Adding interaction features...")
        
        # Month × time index interaction
        df['month_time_interaction'] = df['month'] * df['time_index']
        
        # Quarter × year interaction
        df['quarter_year_interaction'] = df['quarter'] * df['year']
        
        # Volatility × momentum
        if 'sales_volatility' in df.columns and 'momentum_3' in df.columns:
            df['volatility_momentum'] = df['sales_volatility'] * df['momentum_3']
        
        return df
    
    def engineer_features(self, save_path='data/featured.csv'):
        """Run complete feature engineering pipeline"""
        print("="*70)
        print("FEATURE ENGINEERING PIPELINE")
        print("="*70)
        
        # Load data
        df = self.load_data()
        print(f"\nOriginal features: {df.shape[1]}")
        
        # Add features
        df = self.add_lag_features(df)
        df = self.add_rolling_features(df)
        df = self.add_date_features(df)
        df = self.add_growth_features(df)
        df = self.add_statistical_features(df)
        df = self.add_interaction_features(df)
        
        # Fill remaining NaNs (from lags and rolling)
        # Forward fill first, then backward fill, then fill with 0
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        print(f"\nTotal features after engineering: {df.shape[1]}")
        print(f"New features added: {df.shape[1] - 6}")
        
        # Save
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"\nFeatured dataset saved to: {save_path}")
        
        # Display feature summary
        print("\n" + "="*70)
        print("FEATURE SUMMARY")
        print("="*70)
        
        feature_groups = {
            'Original': ['sales', 'avg_order_value', 'sales_volatility', 
                        'num_orders', 'num_transactions'],
            'Lag Features': [c for c in df.columns if 'lag' in c],
            'Rolling Features': [c for c in df.columns if 'rolling' in c],
            'Date Features': [c for c in df.columns if c in [
                'year', 'month', 'quarter', 'month_sin', 'month_cos',
                'quarter_sin', 'quarter_cos', 'time_index'
            ]],
            'Growth Features': [c for c in df.columns if any(
                x in c for x in ['growth', 'macd', 'momentum']
            )],
            'Statistical Features': [c for c in df.columns if any(
                x in c for x in ['zscore', 'percentile', 'diff_from_mean']
            )],
            'Interaction Features': [c for c in df.columns if 'interaction' in c]
        }
        
        for group_name, features in feature_groups.items():
            print(f"\n{group_name} ({len(features)}):")
            for feat in features[:5]:  # Show first 5
                print(f"  - {feat}")
            if len(features) > 5:
                print(f"  ... and {len(features) - 5} more")
        
        print("\n" + "="*70)
        print("FEATURE ENGINEERING COMPLETE")
        print("="*70)
        
        return df


if __name__ == "__main__":
    engineer = TimeSeriesFeatureEngineer()
    featured_df = engineer.engineer_features()
    
    print("\nNext steps:")
    print("  1. Use featured.csv for advanced modeling")
    print("  2. Apply feature selection to reduce dimensionality")
    print("  3. Train XGBoost/LightGBM with these features")
