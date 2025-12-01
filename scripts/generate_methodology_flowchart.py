"""
Generate detailed research methodology flowchart
Shows complete workflow from raw data to validation
"""

from graphviz import Digraph

def create_methodology_flowchart():
    dot = Digraph(comment='Research Methodology Flowchart', format='png')
    dot.attr(rankdir='TB', size='14,20', dpi='300')
    dot.attr('node', shape='box', style='rounded,filled', 
             fontname='Arial', fontsize='11', margin='0.3,0.2')
    
    # ==================== DATA INPUT ====================
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='Data Input', style='filled', color='lightgrey', fontsize='13', labelloc='t')
        c.node('A1', 'Raw E-Commerce Data\n(2014-2018, 48 months)\n~10,000 transactions', 
               fillcolor='#E8F4F8', shape='cylinder')
    
    # ==================== PREPROCESSING ====================
    with dot.subgraph(name='cluster_preprocess') as c:
        c.attr(label='Data Preprocessing', style='filled', color='lightgrey', fontsize='13')
        c.node('B1', 'Missing Value Treatment\n(Postal Code → Unknown)', fillcolor='#FFF4E6')
        c.node('B2', 'Outlier Detection\n(Z-score method, threshold=3)', fillcolor='#FFF4E6')
        c.node('B3', 'Data Validation\n(Type checking, duplicates)', fillcolor='#FFF4E6')
        c.node('B4', 'Monthly Aggregation\n(Daily → Monthly sales)', fillcolor='#FFF4E6')
    
    # ==================== FEATURE ENGINEERING ====================
    with dot.subgraph(name='cluster_features') as c:
        c.attr(label='Feature Engineering (43 Features)', style='filled', color='lightgrey', fontsize='13')
        c.node('C1', 'Lag Features (12)\nLags: 1, 3, 6, 12 months\nSales, Orders, Profit', 
               fillcolor='#E8F8F5')
        c.node('C2', 'Rolling Statistics (12)\nMean, Std, Min, Max\nWindows: 3, 6, 12', 
               fillcolor='#E8F8F5')
        c.node('C3', 'Date Features (7)\nMonth, Quarter, Year\nSin/Cos encoding', 
               fillcolor='#E8F8F5')
        c.node('C4', 'Growth Metrics (6)\nMoM, YoY growth\nMomentum, Volatility', 
               fillcolor='#E8F8F5')
        c.node('C5', 'Statistical Features (6)\nZ-score, Percentile\nDeviation from mean', 
               fillcolor='#E8F8F5')
    
    # ==================== TRAIN/TEST SPLIT ====================
    dot.node('D1', 'Train/Test Split\nTrain: 36 months (Jan 2015 - Dec 2017)\nTest: 12 months (Jan 2018 - Dec 2018)', 
             fillcolor='#FCE4EC', style='filled,bold', shape='box')
    
    # ==================== PARALLEL MODEL DEVELOPMENT ====================
    with dot.subgraph(name='cluster_models') as c:
        c.attr(label='Parallel Model Development', style='filled', color='lightgrey', fontsize='13')
        c.node('E1', 'Model 1: Baseline\nLinear Regression\nTrend + Seasonal', 
               fillcolor='#FFEBEE')
        c.node('E2', 'Model 2: Statistical\nFacebook Prophet\nAdditive decomposition\nMCMC sampling', 
               fillcolor='#C8E6C9')
        c.node('E3', 'Model 3: Deep Learning\nLSTM Network\n50 units, 100 epochs\nSequence length: 12', 
               fillcolor='#FFEBEE')
        c.node('E4', 'Model 5: Gradient Boosting\nXGBoost\n100 trees, depth=4\nL1/L2 regularization', 
               fillcolor='#FFF9C4')
    
    # ==================== ENSEMBLE ====================
    dot.node('F1', 'Model 4: Weighted Ensemble\n60% Prophet + 40% LSTM\nLinear combination', 
             fillcolor='#C8E6C9', style='filled,bold')
    
    # ==================== INITIAL EVALUATION ====================
    dot.node('G1', 'Test Set Evaluation\nMetrics: MAPE, R², MAE, RMSE\nDirection Accuracy', 
             fillcolor='#E1BEE7')
    
    # ==================== COMPREHENSIVE VALIDATION ====================
    with dot.subgraph(name='cluster_validation') as c:
        c.attr(label='Comprehensive Validation', style='filled', color='lightgrey', fontsize='13')
        c.node('H1', 'Cross-Validation\n24-fold Walk-Forward\nExpanding window', 
               fillcolor='#B3E5FC')
        c.node('H2', 'Statistical Tests\nPaired t-test\nFriedman test\nα = 0.05', 
               fillcolor='#B3E5FC')
        c.node('H3', 'Diagnostic Tests\nNormality (Shapiro-Wilk)\nBias (One-sample t-test)\nAutocorrelation', 
               fillcolor='#B3E5FC')
        c.node('H4', 'Feature Importance\nSHAP values\nGain-based importance', 
               fillcolor='#B3E5FC')
    
    # ==================== RESULTS ====================
    with dot.subgraph(name='cluster_results') as c:
        c.attr(label='Model Performance Results', style='filled', color='lightgrey', fontsize='13')
        c.node('I1', 'Linear: MAPE 25.3%', fillcolor='#FFCDD2')
        c.node('I2', 'Prophet: MAPE 19.6%\nR² 0.865', fillcolor='#A5D6A7')
        c.node('I3', 'LSTM: MAPE 30.3%', fillcolor='#FFCDD2')
        c.node('I4', 'Ensemble: MAPE 15.2%\nR² 0.826', fillcolor='#A5D6A7')
        c.node('I5', 'XGBoost: MAPE 11.6%\nR² 0.856 ★ BEST', fillcolor='#FFF59D', style='filled,bold')
    
    # ==================== FINAL OUTPUT ====================
    dot.node('J1', 'Production Model\nXGBoost Deployment\nMonitoring Framework\nRetraining Pipeline', 
             fillcolor='#81C784', style='filled,bold', shape='box3d')
    
    # ==================== EDGES ====================
    
    # Input to Preprocessing
    dot.edge('A1', 'B1')
    dot.edge('B1', 'B2')
    dot.edge('B2', 'B3')
    dot.edge('B3', 'B4')
    
    # Preprocessing to Feature Engineering (parallel)
    dot.edge('B4', 'C1')
    dot.edge('B4', 'C2')
    dot.edge('B4', 'C3')
    dot.edge('B4', 'C4')
    dot.edge('B4', 'C5')
    
    # Feature Engineering to Split
    dot.edge('C1', 'D1')
    dot.edge('C2', 'D1')
    dot.edge('C3', 'D1')
    dot.edge('C4', 'D1')
    dot.edge('C5', 'D1')
    
    # Split to Models (parallel development)
    dot.edge('D1', 'E1')
    dot.edge('D1', 'E2')
    dot.edge('D1', 'E3')
    dot.edge('D1', 'E4')
    
    # Models to Ensemble
    dot.edge('E2', 'F1', label='60%', fontsize='9')
    dot.edge('E3', 'F1', label='40%', fontsize='9')
    
    # Models to Evaluation
    dot.edge('E1', 'G1')
    dot.edge('E2', 'G1')
    dot.edge('E3', 'G1')
    dot.edge('E4', 'G1')
    dot.edge('F1', 'G1')
    
    # Evaluation to Validation (parallel)
    dot.edge('G1', 'H1')
    dot.edge('G1', 'H2')
    dot.edge('G1', 'H3')
    dot.edge('G1', 'H4')
    
    # Validation to Results
    dot.edge('H1', 'I1')
    dot.edge('H1', 'I2')
    dot.edge('H1', 'I3')
    dot.edge('H1', 'I4')
    dot.edge('H1', 'I5')
    
    dot.edge('H2', 'I1')
    dot.edge('H2', 'I2')
    dot.edge('H2', 'I3')
    dot.edge('H2', 'I4')
    dot.edge('H2', 'I5')
    
    dot.edge('H3', 'I1')
    dot.edge('H3', 'I2')
    dot.edge('H3', 'I3')
    dot.edge('H3', 'I4')
    dot.edge('H3', 'I5')
    
    dot.edge('H4', 'I5')
    
    # Best model to Production
    dot.edge('I5', 'J1', style='bold', color='green', penwidth='2')
    
    return dot

if __name__ == '__main__':
    # Generate methodology flowchart
    flowchart = create_methodology_flowchart()
    
    # Save as high-resolution PNG and PDF
    flowchart.render('docs/figures/methodology_flowchart', format='png', cleanup=True)
    flowchart.render('docs/figures/methodology_flowchart', format='pdf', cleanup=True)
    flowchart.save('docs/figures/methodology_flowchart.gv')
    
    print("✅ Methodology flowchart generated successfully!")
    print("📁 Files created:")
    print("   - docs/figures/methodology_flowchart.png")
    print("   - docs/figures/methodology_flowchart.pdf")
    print("   - docs/figures/methodology_flowchart.gv (source)")
    print("\n📊 Flowchart shows:")
    print("   ✓ Raw data input (48 months)")
    print("   ✓ 4-step preprocessing pipeline")
    print("   ✓ 5 categories of feature engineering (43 features)")
    print("   ✓ Train/test split (36/12 months)")
    print("   ✓ 5 parallel models + ensemble")
    print("   ✓ 4-level validation framework")
    print("   ✓ Performance comparison")
    print("   ✓ Production deployment")
