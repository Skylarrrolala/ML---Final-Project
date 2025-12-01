import streamlit as st

def setup_page_config():
    st.set_page_config(
        page_title="Sales Dashboard",
        page_icon="📊",
        layout="wide",
    )


def hide_streamlit_elements():
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stAppDeployButton {visibility: hidden;}
        
        /* Reduce vertical spacing */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        
        /* Reduce space between title and content */
        h1, h2, h3 {
            margin-top: 0.2rem !important;
            margin-bottom: 0.2rem !important;
        }
        
        /* Reduce tab padding */
        .st-emotion-cache-1y4p8pa {
            padding-top: 0.2rem !important;
        }
        
        /* Tighten up overall spacing */
        .stTabs [data-baseweb="tab-panel"] {
            padding-top: 0.5rem !important;
        }

        .stAppHeader {
            visibility: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def create_navigation():
    st.markdown("<h3 style='text-align: center; margin: 0.5rem 0 0.8rem 0; padding-top: 0.5rem;'>Sales Dashboard 📊</h3>", unsafe_allow_html=True)
    
    tabs = st.tabs([
        "📈 Overview", 
        "⏱️ Time Analysis", 
        "🗺️ Geographic Analysis", 
        "🛍️ Product Analysis", 
        "👥 Customer Analysis",
        "🤖 AI-Powered Insights"
    ])
    
    return tabs
