import streamlit as st

def setup_page_config():
    st.set_page_config(
        page_title="Sales Analytics Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed"
    )


def apply_custom_css():
    """Apply professional custom CSS styling"""
    st.markdown(
        """
        <style>
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stAppDeployButton {visibility: hidden;}
        .stAppHeader {visibility: hidden;}
        
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global Typography */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Main Container Styling */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }
        
        /* Header Styling */
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem 2rem 2.5rem 2rem;
            border-radius: 16px;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2);
        }
        
        .main-title {
            color: white;
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
            text-align: center;
            letter-spacing: -0.5px;
        }
        
        .main-subtitle {
            color: rgba(255, 255, 255, 0.9);
            font-size: 1.1rem;
            font-weight: 400;
            text-align: center;
            margin-top: 0.5rem;
        }
        
        /* Tab Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #f8f9fa;
            padding: 0.5rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            background-color: white;
            border-radius: 8px;
            color: #495057;
            font-weight: 500;
            font-size: 0.95rem;
            border: none;
            padding: 0 1.5rem;
            transition: all 0.3s ease;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #e9ecef;
            transform: translateY(-2px);
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }
        
        .stTabs [data-baseweb="tab-panel"] {
            padding-top: 1.5rem;
        }
        
        /* Metric Cards */
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 700;
            color: #2d3748;
        }
        
        [data-testid="stMetricLabel"] {
            font-size: 0.9rem;
            font-weight: 600;
            color: #718096;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        [data-testid="stMetricDelta"] {
            font-weight: 600;
        }
        
        /* Container Cards */
        [data-testid="stHorizontalBlock"] > div[data-testid="column"] > div {
            background-color: white;
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            transition: all 0.3s ease;
        }
        
        [data-testid="stHorizontalBlock"] > div[data-testid="column"] > div:hover {
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
            transform: translateY(-2px);
        }
        
        /* Expander Styling */
        .streamlit-expanderHeader {
            background-color: #f8f9fa;
            border-radius: 8px;
            font-weight: 600;
            color: #495057;
            border: 1px solid #e9ecef;
        }
        
        .streamlit-expanderHeader:hover {
            background-color: #e9ecef;
        }
        
        /* Subheaders */
        h2, h3 {
            color: #2d3748;
            font-weight: 700;
            margin-top: 0.5rem !important;
            margin-bottom: 1rem !important;
        }
        
        h3 {
            font-size: 1.5rem;
        }
        
        /* Chart Containers */
        [data-testid="stContainer"] {
            background-color: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            border: 1px solid #e9ecef;
        }
        
        /* Button Styling */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 2rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        
        /* Checkbox Styling */
        .stCheckbox {
            font-weight: 500;
        }
        
        /* Slider Styling */
        .stSlider [data-baseweb="slider"] {
            background-color: #e9ecef;
        }
        
        /* Dataframe Styling */
        [data-testid="stDataFrame"] {
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }
        
        /* Plotly Chart Container */
        .js-plotly-plot {
            border-radius: 8px;
        }
        
        /* Scrollbar Styling */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #555;
        }
        
        /* Success/Info Messages */
        .element-container .stAlert {
            border-radius: 8px;
            border-left: 4px solid;
        }
        
        /* Spacing utilities */
        .space-small {
            margin-top: 1rem;
        }
        
        .space-medium {
            margin-top: 2rem;
        }
        
        .space-large {
            margin-top: 3rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def hide_streamlit_elements():
    """Deprecated - use apply_custom_css() instead"""
    apply_custom_css()


def create_header():
    """Create professional dashboard header"""
    st.markdown(
        """
        <div class="main-header">
            <h1 class="main-title">📊 Sales Analytics Dashboard</h1>
            <p class="main-subtitle">Comprehensive insights and AI-powered analytics for data-driven decisions</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def create_navigation():
    """Create professional navigation tabs"""
    tabs = st.tabs([
        "📈 Overview", 
        "⏱️ Time Analysis", 
        "🗺️ Geographic", 
        "🛍️ Products", 
        "👥 Customers",
        "🤖 AI Insights"
    ])
    
    return tabs
