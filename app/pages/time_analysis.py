"""
Time analysis page for Sales Dashboard
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from utils import create_date_range_filter, format_currency


def render_time_analysis_page(df):
    """Render the time analysis page with enhanced visualizations"""
    
    with st.expander("🔍 Filter by Date Range", expanded=False):
        _, _, filtered_df = create_date_range_filter(df, key_suffix="time_analysis", label="Analysis Date Range")
        use_filter = st.checkbox("Apply date filter to charts", value=False, key="use_time_analysis_filter")
        
    analysis_df = filtered_df if use_filter else df
    
    # Time-based KPIs
    st.markdown("### ⏱️ Temporal Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate metrics
    monthly_avg = analysis_df.groupby(analysis_df["Order Date"].dt.to_period("M"))["Sales"].sum().mean()
    yearly_sales = analysis_df.groupby(analysis_df["Order Date"].dt.year)["Sales"].sum()
    yoy_growth = ((yearly_sales.iloc[-1] - yearly_sales.iloc[-2]) / yearly_sales.iloc[-2] * 100) if len(yearly_sales) > 1 else 0
    
    with col1:
        st.metric("Avg Monthly Sales", format_currency(monthly_avg))
    
    with col2:
        st.metric("YoY Growth", f"{yoy_growth:.1f}%", delta=f"{yoy_growth:.1f}%")
    
    with col3:
        peak_month = analysis_df.groupby(analysis_df["Order Date"].dt.to_period("M"))["Sales"].sum().idxmax()
        st.metric("Peak Month", str(peak_month))
    
    with col4:
        total_days = (analysis_df["Order Date"].max() - analysis_df["Order Date"].min()).days
        st.metric("Analysis Period", f"{total_days} days")
    
    st.markdown("<div class='space-medium'></div>", unsafe_allow_html=True)
    
    # Monthly sales trend with area chart
    st.markdown("### 📈 Monthly Sales Trend")
    
    monthly_sales = analysis_df.groupby(analysis_df["Order Date"].dt.to_period("M"))["Sales"].sum().reset_index()
    monthly_sales["Order Date"] = monthly_sales["Order Date"].astype(str)
    monthly_sales["Month"] = pd.to_datetime(monthly_sales["Order Date"]).dt.strftime('%b %Y')
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=monthly_sales["Order Date"],
        y=monthly_sales["Sales"],
        mode='lines+markers',
        name='Monthly Sales',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8, color='#764ba2'),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.1)',
        hovertemplate='<b>%{customdata}</b><br>Sales: $%{y:,.0f}<extra></extra>',
        customdata=monthly_sales["Month"]
    ))
    
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Sales ($)",
        height=400,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<div class='space-medium'></div>", unsafe_allow_html=True)
    
    # Yearly comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📅 Year-over-Year Comparison")
        
        yearly_sales_df = analysis_df.groupby(analysis_df["Order Date"].dt.year)["Sales"].sum().reset_index()
        yearly_sales_df.columns = ["Year", "Sales"]
        yearly_sales_df["Year"] = yearly_sales_df["Year"].astype(str)
        
        fig = go.Figure(go.Bar(
            x=yearly_sales_df["Year"],
            y=yearly_sales_df["Sales"],
            marker=dict(
                color=yearly_sales_df["Sales"],
                colorscale='Purples',
                showscale=False
            ),
            text=yearly_sales_df["Sales"].apply(lambda x: f'${x/1000:.0f}K'),
            textposition='auto',
            hovertemplate='<b>Year %{x}</b><br>Sales: $%{y:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Sales ($)",
            height=350,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Quarterly Performance")
        
        quarterly_sales = analysis_df.copy()
        quarterly_sales["Quarter"] = quarterly_sales["Order Date"].dt.to_period("Q").astype(str)
        quarterly_data = quarterly_sales.groupby("Quarter")["Sales"].sum().reset_index()
        
        # Take last 8 quarters for better visibility
        quarterly_data = quarterly_data.tail(8)
        
        fig = go.Figure(go.Bar(
            x=quarterly_data["Quarter"],
            y=quarterly_data["Sales"],
            marker=dict(
                color=quarterly_data["Sales"],
                colorscale='Teal',
                showscale=False
            ),
            text=quarterly_data["Sales"].apply(lambda x: f'${x/1000:.0f}K'),
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Sales: $%{y:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            xaxis_title="Quarter",
            yaxis_title="Sales ($)",
            height=350,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<div class='space-medium'></div>", unsafe_allow_html=True)
    
    # Day of week analysis
    st.markdown("### 📆 Weekly Pattern Analysis")
    
    analysis_df_copy = analysis_df.copy()
    analysis_df_copy["Day of Week"] = analysis_df_copy["Order Date"].dt.day_name()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    # Sales by day of week
    day_sales = analysis_df_copy.groupby("Day of Week")["Sales"].sum().reindex(day_order)
    day_counts = analysis_df_copy["Day of Week"].value_counts().reindex(day_order)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure(go.Bar(
            x=day_order,
            y=day_sales.values,
            marker=dict(
                color=['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe', '#43e97b', '#38f9d7'],
            ),
            text=day_sales.values,
            texttemplate='$%{text:,.0f}',
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Total Sales: $%{y:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Total Sales by Day of Week",
            xaxis_title="",
            yaxis_title="Sales ($)",
            height=350,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure(go.Bar(
            x=day_order,
            y=day_counts.values,
            marker=dict(
                color=['#fa709a', '#fee140', '#30cfd0', '#330867', '#fa8bff', '#2af598', '#009efd'],
            ),
            text=day_counts.values,
            texttemplate='%{text}',
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Order Count: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Order Frequency by Day of Week",
            xaxis_title="",
            yaxis_title="Number of Orders",
            height=350,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Insights
    st.markdown("<div class='space-medium'></div>", unsafe_allow_html=True)
    st.markdown("### 💡 Key Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        best_day = day_sales.idxmax()
        st.info(f"**📅 Best Day:** {best_day}")
        st.caption(f"Total Sales: {format_currency(day_sales[best_day])}")
    
    with col2:
        busiest_day = day_counts.idxmax()
        st.success(f"**🔥 Busiest Day:** {busiest_day}")
        st.caption(f"Orders: {day_counts[busiest_day]:,}")
    
    with col3:
        best_month = monthly_sales.loc[monthly_sales["Sales"].idxmax()]
        st.warning(f"**⭐ Peak Month:** {best_month['Month']}")
        st.caption(f"Sales: {format_currency(best_month['Sales'])}")
