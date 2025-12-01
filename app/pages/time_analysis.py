"""
Time analysis page for Sales Dashboard
"""
import streamlit as st
import plotly.express as px
from utils import create_date_range_filter


def render_time_analysis_page(df):
    """Render the time analysis page"""
    st.subheader("Time Analysis")
    
    with st.expander("🔍 Filter by Date Range", expanded=False):
        _, _, filtered_df = create_date_range_filter(df, key_suffix="time_analysis", label="Analysis Date Range")
        use_filter = st.checkbox("Apply date filter to charts", value=False, key="use_time_analysis_filter")
        
    analysis_df = filtered_df if use_filter else df
    
    # Monthly sales trend
    with st.container(border=True):
        st.subheader("Monthly Sales Trend")
        monthly_sales = analysis_df.groupby(analysis_df["Order Date"].dt.to_period("M"))["Sales"].sum().reset_index()
        monthly_sales["Order Date"] = monthly_sales["Order Date"].astype(str)
        fig = px.line(monthly_sales, x="Order Date", y="Sales", markers=True)
        fig.update_layout(xaxis_title="Month", yaxis_title="Sales ($)")
        st.plotly_chart(fig)
    
    # Order frequency by day of week
    with st.container(border=True):
        st.subheader("Order Frequency by Day of Week")
        analysis_df_copy = analysis_df.copy()
        analysis_df_copy["Day of Week"] = analysis_df_copy["Order Date"].dt.day_name()
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_counts = analysis_df_copy["Day of Week"].value_counts().reindex(day_order)
        fig = px.bar(
            x=day_counts.index,
            y=day_counts.values,
            color=day_counts.index,  # Assign color by day
            color_discrete_sequence=px.colors.qualitative.Set2  # Use a qualitative color set
        )
        fig.update_layout(xaxis_title="Day of Week", yaxis_title="Number of Orders", showlegend=False)
        st.plotly_chart(fig)
