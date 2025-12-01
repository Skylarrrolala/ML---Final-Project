"""
Geographic analysis page for Sales Dashboard
"""
import streamlit as st
import plotly.express as px

from utils import create_date_range_filter


def render_geographic_analysis_page(df):
    """Render the geographic analysis page"""
    st.subheader("Geographic Analysis")

    with st.expander("🔍 Filter by Date Range", expanded=False):
        _, _, filtered_df = create_date_range_filter(df, key_suffix="geographic_analysis", label="Analysis Date Range")
        use_filter = st.checkbox("Apply date filter to charts", value=False, key="use_geographic_analysis_filter")
        
    analysis_df = filtered_df if use_filter else df

    with st.container(border=True):
        st.markdown("#### Key Metrics")
        col1, col2, col3, _ = st.columns(4)
        with col1:
            st.metric("Unique Regions", analysis_df["Region"].nunique())
        with col2:
            st.metric("Unique States", analysis_df["State"].nunique())
        with col3:
            st.metric("Unique Cities", analysis_df["City"].nunique())

    with st.container(border=True):
        st.markdown("#### Sales by Region")
        region_sales = analysis_df.groupby("Region", as_index=False)["Sales"].sum()
        fig = px.pie(
            region_sales,
            names="Region",
            values="Sales",
            color_discrete_sequence=px.colors.sequential.RdBu,
            hole=0.4
        )
        fig.update_traces(textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    # Top 10 states by sales
    with st.container(border=True):
        st.subheader("Top 10 States by Sales")
        state_sales = analysis_df.groupby("State")["Sales"].sum().reset_index().sort_values(by="Sales", ascending=False).head(10)
        fig = px.bar(state_sales, x="State", y="Sales", color="State")
        st.plotly_chart(fig)
    
    # Top 10 cities by sales
    with st.container(border=True):
        st.subheader("Top 10 Cities by Sales")
        city_sales = analysis_df.groupby("City")["Sales"].sum().reset_index().sort_values(by="Sales", ascending=False).head(10)
        fig = px.bar(city_sales, x="City", y="Sales", color="City")
        st.plotly_chart(fig)
    