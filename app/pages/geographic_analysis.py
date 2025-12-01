"""
Geographic analysis page for Sales Dashboard
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils import create_date_range_filter, format_currency


def render_geographic_analysis_page(df):
    """Render the geographic analysis page with enhanced visualizations"""
    
    with st.expander("🔍 Filter by Date Range", expanded=False):
        _, _, filtered_df = create_date_range_filter(df, key_suffix="geographic_analysis", label="Analysis Date Range")
        use_filter = st.checkbox("Apply date filter to charts", value=False, key="use_geographic_analysis_filter")
        
    analysis_df = filtered_df if use_filter else df

    # Geographic KPIs
    st.markdown("### 🗺️ Geographic Distribution")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🌎 Regions", analysis_df["Region"].nunique())
    
    with col2:
        st.metric("📍 States", analysis_df["State"].nunique())
    
    with col3:
        st.metric("🏙️ Cities", analysis_df["City"].nunique())
    
    with col4:
        top_region = analysis_df.groupby("Region")["Sales"].sum().idxmax()
        st.metric("🏆 Top Region", top_region)
    
    st.markdown("<div class='space-medium'></div>", unsafe_allow_html=True)
    
    # Regional Analysis
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📊 Sales by Region")
        
        region_sales = analysis_df.groupby("Region", as_index=False)["Sales"].sum()
        region_sales = region_sales.sort_values("Sales", ascending=False)
        
        fig = go.Figure(data=[go.Pie(
            labels=region_sales["Region"],
            values=region_sales["Sales"],
            hole=0.5,
            marker=dict(
                colors=['#667eea', '#764ba2', '#f093fb', '#4facfe']
            ),
            textinfo='label+percent',
            textfont_size=13,
            hovertemplate='<b>%{label}</b><br>Sales: $%{value:,.0f}<br>Share: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            height=400,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=20, r=20, t=20, b=60)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Regional Performance Metrics")
        
        region_metrics = analysis_df.groupby("Region").agg({
            "Sales": "sum",
            "Order ID": "nunique",
            "Customer ID": "nunique"
        }).reset_index()
        
        region_metrics["Avg Order Value"] = region_metrics["Sales"] / region_metrics["Order ID"]
        region_metrics = region_metrics.sort_values("Sales", ascending=False)
        
        # Display as styled metrics
        for _, row in region_metrics.iterrows():
            with st.container():
                st.markdown(f"**{row['Region']}**")
                subcol1, subcol2, subcol3 = st.columns(3)
                with subcol1:
                    st.caption("Total Sales")
                    st.write(format_currency(row['Sales']))
                with subcol2:
                    st.caption("Orders")
                    st.write(f"{row['Order ID']:,}")
                with subcol3:
                    st.caption("Avg Order")
                    st.write(format_currency(row['Avg Order Value']))
                st.divider()
    
    st.markdown("<div class='space-medium'></div>", unsafe_allow_html=True)
    
    # Top States Analysis
    st.markdown("### 🏛️ Top 10 States by Sales")
    
    state_sales = analysis_df.groupby("State")["Sales"].sum().reset_index()
    state_sales = state_sales.sort_values(by="Sales", ascending=False).head(10)
    
    fig = go.Figure(go.Bar(
        y=state_sales["State"],
        x=state_sales["Sales"],
        orientation='h',
        marker=dict(
            color=state_sales["Sales"],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Sales ($)")
        ),
        text=state_sales["Sales"].apply(lambda x: f'${x/1000:.0f}K'),
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Sales: $%{x:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        xaxis_title="Sales ($)",
        yaxis_title="",
        height=450,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis={'categoryorder':'total ascending'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<div class='space-medium'></div>", unsafe_allow_html=True)
    
    # Top Cities Analysis
    st.markdown("### 🏙️ Top 10 Cities by Sales")
    
    city_sales = analysis_df.groupby("City")["Sales"].sum().reset_index()
    city_sales = city_sales.sort_values(by="Sales", ascending=False).head(10)
    
    fig = go.Figure(go.Bar(
        x=city_sales["City"],
        y=city_sales["Sales"],
        marker=dict(
            color=city_sales["Sales"],
            colorscale='Portland',
            showscale=False
        ),
        text=city_sales["Sales"].apply(lambda x: f'${x/1000:.0f}K'),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Sales: $%{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Sales ($)",
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis={'categoryorder':'total descending'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Geographic Insights
    st.markdown("<div class='space-medium'></div>", unsafe_allow_html=True)
    st.markdown("### 💡 Geographic Insights")
    
    col1, col2, col3 = st.columns(3)
    
    top_state = state_sales.iloc[0]
    top_city = city_sales.iloc[0]
    
    with col1:
        st.success(f"**🏆 Top State:** {top_state['State']}")
        st.caption(f"Sales: {format_currency(top_state['Sales'])}")
    
    with col2:
        st.info(f"**🌟 Top City:** {top_city['City']}")
        st.caption(f"Sales: {format_currency(top_city['Sales'])}")
    
    with col3:
        total_locations = analysis_df["City"].nunique()
        st.warning(f"**📍 Total Locations:** {total_locations}")
        st.caption(f"Across {analysis_df['State'].nunique()} states")
    