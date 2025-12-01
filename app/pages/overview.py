"""
Overview dashboard page for Sales Dashboard
"""
import streamlit as st
import plotly.express as px
from utils import create_date_range_filter


def render_overview_page(df):
    """Render the overview dashboard page"""
    st.subheader("Overview")
    
    
    # Optional: Add date filter for time analysis
    with st.expander("🔍 Filter by Date Range", expanded=False):
        _, _, filtered_df = create_date_range_filter(df, key_suffix="overview", label="Analysis Date Range")
        use_filter = st.checkbox("Apply date filter to charts", value=False, key="use_overview_filter")
        
    # Use filtered data if checkbox is selected, otherwise use full dataset
    analysis_df = filtered_df if use_filter else df
       
    # Key metrics using filtered data
    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Sales", f"${analysis_df['Sales'].sum():,.2f}")
        with col2:
            st.metric("Total Orders", analysis_df["Order ID"].nunique())
        with col3:
            st.metric("Total Customers", analysis_df["Customer ID"].nunique())
        with col4:
            st.metric("Total Products", analysis_df["Product Name"].nunique())
    
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            # Sales by Category
            st.subheader("Sales by Category")
            category_sales = analysis_df.groupby("Category")["Sales"].sum().reset_index()
            fig = px.bar(category_sales, x="Category", y="Sales", color="Category")
            st.plotly_chart(fig)
    with col2:
        with st.container(border=True):
            # Sales by Segment
            st.subheader("Sales by Segment")
            fig = px.pie(analysis_df, names="Segment", values="Sales", hole=0.4)
            st.plotly_chart(fig)

    with st.container(border=True):
        st.subheader("Sales by Subcategory")
        sub_category_sales = analysis_df.groupby("Sub-Category")["Sales"].sum().reset_index()
        fig = px.bar(sub_category_sales, x="Sub-Category", y="Sales", color="Sub-Category")
        st.plotly_chart(fig)
    
    # Sample data
    with st.expander("View Sample Data"):
        st.dataframe(df.head(10))
