"""
Example: How to Load and Use the Trained Model
"""

import joblib
import pandas as pd
import json

# 1. Load the model
model = joblib.load('results/production_model/xgboost_model.pkl')

# 2. Load feature names
with open('results/production_model/feature_names.json', 'r') as f:
    feature_names = json.load(f)

# 3. Prepare your data
# Make sure your data has ALL the features in the same order
new_data = pd.DataFrame({
    # Include all 43 features here
    'num_orders': [1500],
    'volatility_momentum': [0.15],
    'sales_percentile': [0.75],
    # ... add all other features
})

# Ensure features are in correct order
new_data = new_data[feature_names]

# 4. Make predictions
prediction = model.predict(new_data)

print(f"Predicted sales: ${prediction[0]:,.2f}")

# 5. Get feature importance
importance = model.feature_importances_
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values('importance', ascending=False)

print("\nTop 5 Important Features:")
print(feature_importance.head())
