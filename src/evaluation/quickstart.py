#!/usr/bin/env python3
"""
Quick Start Script - Run Baseline Evaluation
This script provides a fast way to evaluate your current model performance
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.baseline_evaluation import BaselineEvaluator

def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║   SALES FORECASTING MODEL - BASELINE EVALUATION              ║
    ║   Quick performance assessment of your current model          ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    print("This will:")
    print("  1. Load your cleaned sales data")
    print("  2. Train Prophet and LSTM models")
    print("  3. Evaluate ensemble performance")
    print("  4. Generate visualizations and metrics")
    print()
    
    evaluator = BaselineEvaluator(
        data_path='../../data/cleaned.csv',
        results_dir='../../results/metrics'
    )
    
    results = evaluator.evaluate()
    
    # Print key findings
    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + " "*22 + "KEY FINDINGS" + " "*34 + "║")
    print("╚" + "═"*68 + "╝")
    
    ensemble = results['metrics']['ensemble']
    
    print(f"""
    📊 PERFORMANCE METRICS:
       • MAPE:      {ensemble['mape']:.2f}%
       • RMSE:      ${ensemble['rmse']:,.2f}
       • MAE:       ${ensemble['mae']:,.2f}
       • R² Score:  {ensemble['r2']:.4f}
       • Direction: {ensemble['direction_accuracy']:.1f}%
    
    ✅ QUALITY ASSESSMENT:
    """)
    
    # Quality assessment
    if ensemble['mape'] < 15:
        quality = "EXCELLENT"
        emoji = "🌟"
    elif ensemble['mape'] < 20:
        quality = "GOOD"
        emoji = "✅"
    elif ensemble['mape'] < 25:
        quality = "ACCEPTABLE"
        emoji = "👍"
    else:
        quality = "NEEDS IMPROVEMENT"
        emoji = "⚠️"
    
    print(f"       {emoji} Model Quality: {quality} (MAPE = {ensemble['mape']:.2f}%)")
    
    # Recommendations
    print("\n    💡 RECOMMENDATIONS:")
    
    if ensemble['mape'] > 15:
        print("       → Run feature engineering to improve accuracy")
        print("       → Consider hyperparameter tuning")
        print("       → Try advanced ensemble methods (stacking)")
    
    if ensemble['r2'] < 0.85:
        print("       → Add more features to capture variance")
        print("       → Try XGBoost or LightGBM models")
    
    if ensemble['direction_accuracy'] < 80:
        print("       → Focus on trend-capturing features")
        print("       → Tune LSTM sequence length")
    
    print("\n    📁 RESULTS SAVED TO:")
    print(f"       • results/metrics/baseline_results.json")
    print(f"       • results/metrics/baseline_summary.txt")
    print(f"       • results/metrics/baseline_evaluation.png")
    
    print("\n    🚀 NEXT STEPS:")
    print("       1. Review the visualization (baseline_evaluation.png)")
    print("       2. Run full improvement pipeline:")
    print("          python src/evaluation/run_improvement_pipeline.py")
    print()


if __name__ == "__main__":
    main()
