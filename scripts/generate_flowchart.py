"""
Generate project flowchart using graphviz
Install: pip install graphviz
"""

from graphviz import Digraph

def create_project_flowchart():
    # Create a new directed graph
    dot = Digraph(comment='ML Sales Forecasting Project', format='png')
    dot.attr(rankdir='TB', size='12,16')
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue', 
             fontname='Arial', fontsize='10')
    
    # Data Collection & Preprocessing
    dot.node('A', 'Raw Data\n(raw.csv)\n~10K transactions')
    dot.node('B', 'EDA Notebook\n(eda.ipynb)')
    dot.node('C', 'Data Cleaning\n- Handle missing values\n- Outlier detection (Z-score)\n- Feature validation')
    dot.node('D', 'Cleaned Data\n(cleaned.csv)')
    
    # Feature Engineering
    dot.node('E', 'Feature Engineering\n(predictive.ipynb)')
    dot.node('F', '43 Features Created:\n- Lag features (12)\n- Rolling stats (12)\n- Date features (7)\n- Growth metrics (6)\n- Statistical features (6)')
    dot.node('G', 'Featured Data\n(featured.csv)')
    
    # Model Development
    dot.node('H1', 'Model 1:\nLinear Regression\nMAPE: 25.3%', fillcolor='lightyellow')
    dot.node('H2', 'Model 2:\nFacebook Prophet\nMAPE: 19.6%', fillcolor='lightgreen')
    dot.node('H3', 'Model 3:\nLSTM Neural Network\nMAPE: 30.3%', fillcolor='lightyellow')
    dot.node('H4', 'Model 4:\nEnsemble (P+L)\nMAPE: 15.2%', fillcolor='lightgreen')
    dot.node('H5', 'Model 5:\nXGBoost + Features\nMAPE: 11.6%', fillcolor='gold')
    
    # Validation
    dot.node('I', 'Validation Framework:\n- Train/Test Split (36/12)\n- Cross-Validation (24-fold)\n- Statistical Tests\n- Diagnostic Tests')
    
    # Results
    dot.node('J', 'Best Model:\nXGBoost\nR²: 0.856\nMAE: $6,016')
    
    # Additional Analysis
    dot.node('K1', 'K-Means\nCustomer Segmentation\n(k_means_customer_segmentation.ipynb)')
    dot.node('K2', 'Linear Regression\nAnalysis\n(linear_regression.ipynb)')
    
    # Production
    dot.node('L', 'Production Model\n(results/production_model/)')
    dot.node('M', 'Streamlit Dashboard\n(app/streamlit_app.py)\n6 Pages')
    
    # Documentation
    dot.node('N1', 'Research Paper\n(paper/main_updated.md)', fillcolor='lightcyan')
    dot.node('N2', 'Presentation Slides\n(presentation/slides_updated.md)', fillcolor='lightcyan')
    
    # Define edges (workflow)
    dot.edge('A', 'B')
    dot.edge('B', 'C')
    dot.edge('C', 'D')
    dot.edge('D', 'E')
    dot.edge('E', 'F')
    dot.edge('F', 'G')
    
    # Model training paths
    dot.edge('G', 'H1')
    dot.edge('G', 'H2')
    dot.edge('G', 'H3')
    dot.edge('H2', 'H4', label='60%')
    dot.edge('H3', 'H4', label='40%')
    dot.edge('G', 'H5')
    
    # Validation
    dot.edge('H1', 'I')
    dot.edge('H2', 'I')
    dot.edge('H3', 'I')
    dot.edge('H4', 'I')
    dot.edge('H5', 'I')
    
    # Best model selection
    dot.edge('I', 'J')
    dot.edge('J', 'L')
    
    # Additional analysis paths
    dot.edge('D', 'K1')
    dot.edge('D', 'K2')
    
    # Production deployment
    dot.edge('L', 'M')
    
    # Documentation
    dot.edge('J', 'N1')
    dot.edge('J', 'N2')
    
    # Add legend
    with dot.subgraph(name='cluster_legend') as legend:
        legend.attr(label='Legend', fontsize='12', style='dashed')
        legend.node('L1', 'Weak Performance', fillcolor='lightyellow')
        legend.node('L2', 'Good Performance', fillcolor='lightgreen')
        legend.node('L3', 'Best Performance', fillcolor='gold')
        legend.node('L4', 'Documentation', fillcolor='lightcyan')
    
    return dot

if __name__ == '__main__':
    # Generate flowchart
    flowchart = create_project_flowchart()
    
    # Save as PNG and PDF
    flowchart.render('docs/figures/project_flowchart', format='png', cleanup=True)
    flowchart.render('docs/figures/project_flowchart', format='pdf', cleanup=True)
    
    # Also save the source code
    flowchart.save('docs/figures/project_flowchart.gv')
    
    print("✅ Flowchart generated successfully!")
    print("📁 Files created:")
    print("   - docs/figures/project_flowchart.png")
    print("   - docs/figures/project_flowchart.pdf")
    print("   - docs/figures/project_flowchart.gv (source)")
