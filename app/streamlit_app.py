from config import setup_page_config, hide_streamlit_elements, create_navigation
from data_loader import load_data, load_cleaned_data
from pages.overview import render_overview_page
from pages.time_analysis import render_time_analysis_page
from pages.geographic_analysis import render_geographic_analysis_page
from pages.product_analysis import render_product_analysis_page
from pages.customer_analysis import render_customer_analysis_page
from pages.ai_insights import render_ai_insights_page

def main():
    setup_page_config()
    hide_streamlit_elements()
    
    df = load_data()
    df_cleaned = load_cleaned_data()
    tabs = create_navigation()
    
    with tabs[0]: 
        render_overview_page(df)
    with tabs[1]:
        render_time_analysis_page(df)
    with tabs[2]:
        render_geographic_analysis_page(df)
    with tabs[3]:
        render_product_analysis_page(df)
    with tabs[4]:
        render_customer_analysis_page(df)
    with tabs[5]:
        render_ai_insights_page(df_cleaned)

if __name__ == "__main__":
    main()