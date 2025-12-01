"""
Customer analysis page for Sales Dashboard
"""
import streamlit as st
import plotly.express as px

from utils import create_date_range_filter


def render_customer_analysis_page(df):
    st.subheader("Customer Analysis")

    with st.expander("🔍 Filter by Date Range", expanded=False):
        _, _, filtered_df = create_date_range_filter(df, key_suffix="customer_analysis", label="Analysis Date Range")
        use_filter = st.checkbox("Apply date filter to charts", value=False, key="use_customer_analysis_filter")
        
    analysis_df = filtered_df if use_filter else df
    
    # Order frequency distribution
    with st.container(border=True):
        st.subheader("Customer Order Frequency")
        customer_orders = analysis_df.groupby("Customer ID").size().reset_index(name="Order Count")
        order_dist = customer_orders["Order Count"].value_counts().reset_index()
        order_dist.columns = ["Orders Made", "Number of Customers"]
        fig = px.bar(order_dist, x="Orders Made", y="Number of Customers")
        st.plotly_chart(fig)

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            # Sales by customer segment
            st.subheader("Sales by Customer Segment")
            segment_sales = analysis_df.groupby("Segment")["Sales"].sum().reset_index()
            fig = px.pie(segment_sales, names="Segment", values="Sales")
            st.plotly_chart(fig)

    with col2:
        with st.container(border=True):
            # Average sales per customer by segment
            st.subheader("Average Sales per Customer by Segment")
            avg_sales = analysis_df.groupby(["Segment", "Customer ID"])["Sales"].sum().reset_index()
            avg_sales = avg_sales.groupby("Segment")["Sales"].mean().reset_index()
            fig = px.bar(avg_sales, x="Segment", y="Sales", color="Segment")
            st.plotly_chart(fig)
