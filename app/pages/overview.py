"""
Overview dashboard page for Sales Dashboard
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils import create_date_range_filter, format_currency, create_metric_card


def render_overview_page(df):
    """Render the overview dashboard page with professional design"""
    
    # Date filter section
    with st.expander("🔍 Filter by Date Range", expanded=False):
        _, _, filtered_df = create_date_range_filter(df, key_suffix="overview", label="Analysis Date Range")
        use_filter = st.checkbox("Apply date filter to charts", value=False, key="use_overview_filter")
        
    # Use filtered data if checkbox is selected, otherwise use full dataset
    analysis_df = filtered_df if use_filter else df
       
    # Key Performance Indicators
    st.markdown("### 📊 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_sales = analysis_df['Sales'].sum()
        st.markdown(create_metric_card(
            "💰 Total Revenue",
            format_currency(total_sales),
            "success"
        ), unsafe_allow_html=True)
    
    with col2:
        total_orders = analysis_df["Order ID"].nunique()
        st.markdown(create_metric_card(
            "📦 Total Orders",
            f"{total_orders:,}",
            "info"
        ), unsafe_allow_html=True)
    
    with col3:
        total_customers = analysis_df["Customer ID"].nunique()
        st.markdown(create_metric_card(
            "👥 Unique Customers",
            f"{total_customers:,}",
            "warning"
        ), unsafe_allow_html=True)
    
    with col4:
        avg_order_value = total_sales / total_orders if total_orders > 0 else 0
        st.markdown(create_metric_card(
            "💳 Avg Order Value",
            format_currency(avg_order_value),
            "primary"
        ), unsafe_allow_html=True)
    
    st.markdown("<div class='space-medium'></div>", unsafe_allow_html=True)
    
    # Sales Distribution Section
    st.markdown("### 📈 Sales Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sales by Category - Enhanced Bar Chart
        category_sales = analysis_df.groupby("Category")["Sales"].sum().reset_index()
        category_sales = category_sales.sort_values("Sales", ascending=True)
        
        fig = go.Figure(go.Bar(
            x=category_sales["Sales"],
            y=category_sales["Category"],
            orientation='h',
            marker=dict(
                color=category_sales["Sales"],
                colorscale='Viridis',
                showscale=False
            ),
            text=category_sales["Sales"].apply(lambda x: f'${x:,.0f}'),
            textposition='auto',
        ))
        
        fig.update_layout(
            title="Sales by Category",
            xaxis_title="Sales ($)",
            yaxis_title="",
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sales by Segment - Enhanced Donut Chart
        segment_sales = analysis_df.groupby("Segment")["Sales"].sum().reset_index()
        
        fig = go.Figure(data=[go.Pie(
            labels=segment_sales["Segment"],
            values=segment_sales["Sales"],
            hole=0.5,
            marker=dict(colors=['#667eea', '#764ba2', '#f093fb']),
            textinfo='label+percent',
            textfont_size=12,
            pull=[0.05, 0, 0]
        )])
        
        fig.update_layout(
            title="Sales by Customer Segment",
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<div class='space-medium'></div>", unsafe_allow_html=True)
    
    # Product Performance Section
    st.markdown("### 🛍️ Product Performance")
    
    # Sales by Subcategory - Enhanced Horizontal Bar
    sub_category_sales = analysis_df.groupby("Sub-Category")["Sales"].sum().reset_index()
    sub_category_sales = sub_category_sales.sort_values("Sales", ascending=True)
    
    fig = go.Figure(go.Bar(
        x=sub_category_sales["Sales"],
        y=sub_category_sales["Sub-Category"],
        orientation='h',
        marker=dict(
            color=sub_category_sales["Sales"],
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="Sales ($)")
        ),
        hovertemplate='<b>%{y}</b><br>Sales: $%{x:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Sales by Sub-Category",
        xaxis_title="Sales ($)",
        yaxis_title="",
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Additional Insights Section
    st.markdown("<div class='space-medium'></div>", unsafe_allow_html=True)
    st.markdown("### 📋 Additional Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        top_category = category_sales.iloc[-1]
        st.info(f"**🏆 Top Category:** {top_category['Category']}")
        st.caption(f"Sales: {format_currency(top_category['Sales'])}")
    
    with col2:
        top_segment = segment_sales.sort_values("Sales", ascending=False).iloc[0]
        st.success(f"**👔 Leading Segment:** {top_segment['Segment']}")
        st.caption(f"Sales: {format_currency(top_segment['Sales'])}")
    
    with col3:
        top_subcategory = sub_category_sales.iloc[-1]
        st.warning(f"**⭐ Top Sub-Category:** {top_subcategory['Sub-Category']}")
        st.caption(f"Sales: {format_currency(top_subcategory['Sales'])}")
    
    # Sample data viewer
    with st.expander("📊 View Sample Data", expanded=False):
        st.dataframe(
            analysis_df.head(100).style.format({
                'Sales': '${:,.2f}',
                'Order Date': lambda x: x.strftime('%Y-%m-%d'),
                'Ship Date': lambda x: x.strftime('%Y-%m-%d')
            }),
            use_container_width=True,
            height=400
        )
