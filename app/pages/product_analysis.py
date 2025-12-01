"""
Product analysis page for Sales Dashboard
"""
import streamlit as st
import plotly.express as px

from utils import create_date_range_filter


def render_product_analysis_page(df):
    """Render the product analysis page"""
    st.subheader("Product Analysis")

    with st.expander("🔍 Filter by Date Range", expanded=False):
        _, _, filtered_df = create_date_range_filter(df, key_suffix="product_analysis", label="Analysis Date Range")
        use_filter = st.checkbox("Apply date filter to charts", value=False, key="use_product_analysis_filter")
        
    analysis_df = filtered_df if use_filter else df
    
    # Top 10 products by sales
    with st.container(border=True):
        st.subheader("Top 10 Products by Sales")
        product_sales = analysis_df.groupby("Product Name")["Sales"].sum().reset_index().sort_values(by="Sales", ascending=False).head(10)
        fig = px.bar(product_sales, x="Sales", y="Product Name", orientation='h', color="Sales")
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig)
    
    # Product count by category
    with st.container(border=True):
        st.subheader("Product Count by Category")
        category_counts = analysis_df.groupby(["Category", "Sub-Category"]).size().reset_index(name="Count")
        fig = px.treemap(category_counts, path=["Category", "Sub-Category"], values="Count")
        st.plotly_chart(fig)
