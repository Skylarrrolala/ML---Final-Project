"""
Visual Comparison Dashboard: Baseline vs Improved Model
Creates comprehensive comparison visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

def create_performance_comparison():
    """Create performance comparison chart"""
    
    baseline = {
        'model': 'Prophet + LSTM\n(Baseline)',
        'mape': 19.3,
        'r2': 0.840,
        'direction': 83.3
    }
    
    improved = {
        'model': 'XGBoost + Features\n(Improved)',
        'mape': 11.6,
        'r2': 0.856,
        'direction': 72.7
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # MAPE Comparison
    models = [baseline['model'], improved['model']]
    mapes = [baseline['mape'], improved['mape']]
    colors = ['#e74c3c', '#27ae60']
    
    bars1 = axes[0].bar(models, mapes, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[0].set_ylabel('MAPE (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('Forecast Accuracy\n(Lower is Better)', fontsize=13, fontweight='bold')
    axes[0].set_ylim(0, 25)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, mapes)):
        height = bar.get_height()
        improvement = ''
        if i == 1:
            improvement = f'\n(↓{baseline["mape"] - val:.1f}%)'
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{val:.1f}%{improvement}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add improvement annotation
    axes[0].annotate('', xy=(1, improved['mape']), xytext=(1, baseline['mape']),
                    arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    axes[0].text(1.15, (baseline['mape'] + improved['mape'])/2, 
                '40%\nbetter', fontsize=10, color='green', fontweight='bold')
    
    # R² Comparison
    r2s = [baseline['r2'], improved['r2']]
    bars2 = axes[1].bar(models, r2s, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[1].set_ylabel('R² Score', fontsize=12, fontweight='bold')
    axes[1].set_title('Model Fit\n(Higher is Better)', fontsize=13, fontweight='bold')
    axes[1].set_ylim(0, 1)
    axes[1].axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Good Threshold')
    
    for bar, val in zip(bars2, r2s):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.3f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Direction Accuracy Comparison
    directions = [baseline['direction'], improved['direction']]
    bars3 = axes[2].bar(models, directions, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[2].set_ylabel('Direction Accuracy (%)', fontsize=12, fontweight='bold')
    axes[2].set_title('Trend Prediction\n(Higher is Better)', fontsize=13, fontweight='bold')
    axes[2].set_ylim(0, 100)
    axes[2].axhline(y=75, color='gray', linestyle='--', alpha=0.5, label='Good Threshold')
    
    for bar, val in zip(bars3, directions):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{val:.1f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_feature_importance():
    """Create feature importance visualization"""
    
    features = [
        ('num_orders', 0.4848),
        ('volatility_momentum', 0.1220),
        ('sales_percentile', 0.0980),
        ('sales_zscore', 0.0774),
        ('sales_lag_12', 0.0358),
        ('month', 0.0297),
        ('diff_from_mean_3', 0.0200),
        ('momentum_6', 0.0190),
        ('sales_rolling_mean_3', 0.0155),
        ('sales_rolling_mean_12', 0.0121)
    ]
    
    df = pd.DataFrame(features, columns=['Feature', 'Importance'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors_gradient = plt.cm.viridis(np.linspace(0.3, 0.9, len(df)))
    bars = ax.barh(df['Feature'], df['Importance'], color=colors_gradient, 
                   edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title('Top 10 Most Important Features\nXGBoost Feature Importance', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.invert_yaxis()
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, df['Importance'])):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
               f'{val:.4f}',
               ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_prediction_comparison():
    """Create actual vs predicted comparison"""
    
    # Load predictions
    predictions = pd.read_csv('results/xgboost_optimized/predictions.csv')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Time series comparison
    time_periods = range(1, len(predictions) + 1)
    axes[0, 0].plot(time_periods, predictions['actual'], 'o-', 
                   label='Actual', linewidth=2, markersize=8, color='#3498db')
    axes[0, 0].plot(time_periods, predictions['predicted'], 's--', 
                   label='Predicted', linewidth=2, markersize=8, color='#e74c3c', alpha=0.7)
    axes[0, 0].set_xlabel('Time Period', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Sales ($)', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Test Set: Actual vs Predicted Sales', fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot
    axes[0, 1].scatter(predictions['actual'], predictions['predicted'], 
                      s=100, alpha=0.6, color='#9b59b6', edgecolor='black', linewidth=1)
    min_val = predictions[['actual', 'predicted']].min().min()
    max_val = predictions[['actual', 'predicted']].max().max()
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    axes[0, 1].set_xlabel('Actual Sales ($)', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Predicted Sales ($)', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Actual vs Predicted Scatter', fontsize=13, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Error distribution
    axes[1, 0].hist(predictions['error_pct'], bins=15, color='#e67e22', 
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[1, 0].set_xlabel('Error (%)', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Prediction Error Distribution', fontsize=13, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Residuals
    axes[1, 1].scatter(predictions['predicted'], predictions['error'], 
                      s=100, alpha=0.6, color='#1abc9c', edgecolor='black', linewidth=1)
    axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Predicted Sales ($)', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Residual ($)', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Residual Plot', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_improvement_summary():
    """Create summary infographic"""
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'MODEL IMPROVEMENT SUMMARY', 
           ha='center', va='top', fontsize=24, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#3498db', alpha=0.3))
    
    # Baseline section
    ax.text(0.15, 0.80, 'BASELINE MODEL', ha='center', fontsize=16, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='#e74c3c', alpha=0.3))
    ax.text(0.15, 0.72, 'Prophet (60%) + LSTM (40%)', ha='center', fontsize=11)
    ax.text(0.15, 0.66, 'MAPE: 19.3%', ha='center', fontsize=12, fontweight='bold')
    ax.text(0.15, 0.61, 'R²: 0.840', ha='center', fontsize=12, fontweight='bold')
    ax.text(0.15, 0.56, 'Direction: 83.3%', ha='center', fontsize=12, fontweight='bold')
    
    # Arrow
    ax.annotate('', xy=(0.65, 0.68), xytext=(0.35, 0.68),
               arrowprops=dict(arrowstyle='->', lw=3, color='green'))
    ax.text(0.5, 0.72, '40% Better\nAccuracy!', ha='center', fontsize=13, 
           fontweight='bold', color='green',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    # Improved section
    ax.text(0.85, 0.80, 'IMPROVED MODEL', ha='center', fontsize=16, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='#27ae60', alpha=0.3))
    ax.text(0.85, 0.72, 'XGBoost + 43 Features', ha='center', fontsize=11)
    ax.text(0.85, 0.66, 'MAPE: 11.6%', ha='center', fontsize=12, fontweight='bold', color='green')
    ax.text(0.85, 0.61, 'R²: 0.856', ha='center', fontsize=12, fontweight='bold', color='green')
    ax.text(0.85, 0.56, 'Direction: 72.7%', ha='center', fontsize=12, fontweight='bold')
    
    # Improvements section
    ax.text(0.5, 0.45, 'KEY IMPROVEMENTS', ha='center', fontsize=16, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='#f39c12', alpha=0.3))
    
    improvements = [
        '✓ Feature Engineering: 43 engineered features',
        '✓ Model Change: XGBoost (better for tabular data)',
        '✓ Regularization: Prevents overfitting on small data',
        '✓ Feature Selection: Focus on top predictive features',
        '✓ Hyperparameter Optimization: Tuned for time series'
    ]
    
    y_pos = 0.38
    for improvement in improvements:
        ax.text(0.5, y_pos, improvement, ha='center', fontsize=11,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.2))
        y_pos -= 0.06
    
    # Metrics comparison
    ax.text(0.5, 0.05, 'Result: MAPE improved from 19.3% → 11.6% (7.7 percentage points, 40% relative improvement)',
           ha='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.4))
    
    return fig

def main():
    print("="*70)
    print("CREATING VISUAL COMPARISON DASHBOARD")
    print("="*70)
    
    results_dir = Path('results/visual_comparison')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create all visualizations
    print("\n1. Creating performance comparison...")
    fig1 = create_performance_comparison()
    fig1.savefig(results_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved to {results_dir / 'performance_comparison.png'}")
    plt.close()
    
    print("\n2. Creating feature importance chart...")
    fig2 = create_feature_importance()
    fig2.savefig(results_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved to {results_dir / 'feature_importance.png'}")
    plt.close()
    
    print("\n3. Creating prediction comparison...")
    fig3 = create_prediction_comparison()
    fig3.savefig(results_dir / 'prediction_comparison.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved to {results_dir / 'prediction_comparison.png'}")
    plt.close()
    
    print("\n4. Creating improvement summary infographic...")
    fig4 = create_improvement_summary()
    fig4.savefig(results_dir / 'improvement_summary.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved to {results_dir / 'improvement_summary.png'}")
    plt.close()
    
    # Create summary report
    summary = {
        'baseline': {
            'model': 'Prophet (60%) + LSTM (40%)',
            'mape': 19.3,
            'r2': 0.840,
            'direction_accuracy': 83.3
        },
        'improved': {
            'model': 'XGBoost with Feature Engineering',
            'mape': 11.6,
            'r2': 0.856,
            'direction_accuracy': 72.7,
            'features': 43
        },
        'improvement': {
            'mape_reduction': 7.7,
            'mape_improvement_pct': 39.9,
            'r2_increase': 0.016
        },
        'visualizations': [
            'performance_comparison.png',
            'feature_importance.png',
            'prediction_comparison.png',
            'improvement_summary.png'
        ]
    }
    
    with open(results_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n   ✓ Summary saved to {results_dir / 'summary.json'}")
    
    print("\n" + "="*70)
    print("✅ VISUAL COMPARISON DASHBOARD COMPLETE!")
    print("="*70)
    print(f"\nAll visualizations saved to: {results_dir}/")
    print("\nFiles created:")
    print("  1. performance_comparison.png   - Side-by-side metric comparison")
    print("  2. feature_importance.png       - Top 10 predictive features")
    print("  3. prediction_comparison.png    - Actual vs predicted analysis")
    print("  4. improvement_summary.png      - One-page summary infographic")
    print("  5. summary.json                 - Data for presentations")
    
    print("\n💡 Use these in your:")
    print("  - Presentation slides")
    print("  - Research paper")
    print("  - Project documentation")
    print("  - GitHub README")

if __name__ == '__main__':
    main()
