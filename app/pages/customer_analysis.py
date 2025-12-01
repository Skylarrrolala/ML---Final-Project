"""
Customer analysis page for Sales Dashboard
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils import create_date_range_filter, format_currency


def render_customer_analysis_page(df):
    """Render the customer analysis page with enhanced visualizations"""
    
    with st.expander("🔍 Filter by Date Range", expanded=False):
        _, _, filtered_df = create_date_range_filter(df, key_suffix="customer_analysis", label="Analysis Date Range")
        use_filter = st.checkbox("Apply date filter to charts", value=False, key="use_customer_analysis_filter")
        
    analysis_df = filtered_df if use_filter else df
    
    # Customer KPIs
    st.markdown("### 👥 Customer Metrics Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_customers = analysis_df["Customer ID"].nunique()
    total_orders = analysis_df["Order ID"].nunique()
    avg_orders_per_customer = total_orders / total_customers if total_customers > 0 else 0
    customer_lifetime_value = analysis_df.groupby("Customer ID")["Sales"].sum().mean()
    
    with col1:
        st.metric("👥 Total Customers", f"{total_customers:,}")
    
    with col2:
        st.metric("📦 Total Orders", f"{total_orders:,}")
    
    with col3:
        st.metric("📊 Avg Orders/Customer", f"{avg_orders_per_customer:.1f}")
    
    with col4:
        st.metric("💰 Avg Customer Value", format_currency(customer_lifetime_value))
    
    st.markdown("<div class='space-medium'></div>", unsafe_allow_html=True)
    
    # Customer Behavior Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Customer Order Frequency Distribution")
        
        customer_orders = analysis_df.groupby("Customer ID").size().reset_index(name="Order Count")
        order_dist = customer_orders["Order Count"].value_counts().sort_index().reset_index()
        order_dist.columns = ["Orders Made", "Number of Customers"]
        
        # Limit to first 15 for better visualization
        order_dist = order_dist.head(15)
        
        fig = go.Figure(go.Bar(
            x=order_dist["Orders Made"],
            y=order_dist["Number of Customers"],
            marker=dict(
                color=order_dist["Number of Customers"],
                colorscale='Blues',
                showscale=False
            ),
            text=order_dist["Number of Customers"],
            textposition='outside',
            hovertemplate='<b>%{x} Orders</b><br>Customers: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            xaxis_title="Number of Orders Made",
            yaxis_title="Number of Customers",
            height=350,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 💎 Customer Value Distribution")
        
        customer_value = analysis_df.groupby("Customer ID")["Sales"].sum().reset_index()
        customer_value.columns = ["Customer ID", "Total Spent"]
        
        # Create value segments
        customer_value["Value Segment"] = pd.cut(
            customer_value["Total Spent"],
            bins=[0, 500, 2000, 5000, float('inf')],
            labels=["Low ($0-500)", "Medium ($500-2K)", "High ($2K-5K)", "Premium ($5K+)"]
        )
        
        segment_dist = customer_value["Value Segment"].value_counts().reset_index()
        segment_dist.columns = ["Segment", "Count"]
        
        fig = go.Figure(data=[go.Pie(
            labels=segment_dist["Segment"],
            values=segment_dist["Count"],
            hole=0.5,
            marker=dict(colors=['#fa709a', '#fee140', '#30cfd0', '#667eea']),
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Customers: %{value}<br>Share: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
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
    
    # Segment Analysis
    st.markdown("### 🎯 Customer Segment Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sales by segment with enhanced styling
        segment_sales = analysis_df.groupby("Segment")["Sales"].sum().reset_index()
        segment_sales = segment_sales.sort_values("Sales", ascending=True)
        
        fig = go.Figure(data=[go.Pie(
            labels=segment_sales["Segment"],
            values=segment_sales["Sales"],
            hole=0.4,
            marker=dict(colors=['#667eea', '#764ba2', '#f093fb']),
            textinfo='label+percent',
            textfont_size=13,
            pull=[0.05, 0, 0],
            hovertemplate='<b>%{label}</b><br>Sales: $%{value:,.0f}<br>Share: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title="Sales Distribution by Segment",
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Average sales per customer by segment
        customer_avg_sales = analysis_df.groupby(["Segment", "Customer ID"])["Sales"].sum().reset_index()
        avg_sales = customer_avg_sales.groupby("Segment")["Sales"].mean().reset_index()
        avg_sales = avg_sales.sort_values("Sales", ascending=False)
        
        fig = go.Figure(go.Bar(
            x=avg_sales["Segment"],
            y=avg_sales["Sales"],
            marker=dict(
                color=avg_sales["Sales"],
                colorscale='Purples',
                showscale=False
            ),
            text=avg_sales["Sales"].apply(lambda x: f'${x:,.0f}'),
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Avg Spending: $%{y:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Average Customer Spending by Segment",
            xaxis_title="",
            yaxis_title="Average Sales per Customer ($)",
            height=350,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<div class='space-medium'></div>", unsafe_allow_html=True)
    
    # Top Customers
    st.markdown("### 🏆 Top 15 Customers by Revenue")
    
    top_customers = analysis_df.groupby("Customer Name").agg({
        "Sales": "sum",
        "Order ID": "nunique"
    }).reset_index()
    top_customers.columns = ["Customer", "Total Sales", "Orders"]
    top_customers = top_customers.sort_values("Total Sales", ascending=False).head(15)
    
    fig = go.Figure(go.Bar(
        y=top_customers["Customer"][::-1],
        x=top_customers["Total Sales"][::-1],
        orientation='h',
        marker=dict(
            color=top_customers["Total Sales"][::-1],
            colorscale='Sunset',
            showscale=True,
            colorbar=dict(title="Sales ($)")
        ),
        text=top_customers["Total Sales"][::-1].apply(lambda x: f'${x/1000:.1f}K'),
        textposition='auto',
        customdata=top_customers["Orders"][::-1],
        hovertemplate='<b>%{y}</b><br>Total Sales: $%{x:,.0f}<br>Orders: %{customdata}<extra></extra>'
    ))
    
    fig.update_layout(
        xaxis_title="Total Sales ($)",
        yaxis_title="",
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Customer Insights
    st.markdown("<div class='space-medium'></div>", unsafe_allow_html=True)
    st.markdown("### 💡 Customer Insights")
    
    col1, col2, col3 = st.columns(3)
    
    top_segment = segment_sales.iloc[-1]
    top_customer = top_customers.iloc[0]
    repeat_customers = (customer_orders["Order Count"] > 1).sum()
    repeat_rate = (repeat_customers / total_customers * 100) if total_customers > 0 else 0
    
    with col1:
        st.success(f"**🏆 Top Segment:** {top_segment['Segment']}")
        st.caption(f"Sales: {format_currency(top_segment['Sales'])}")
    
    with col2:
        st.info(f"**👑 Top Customer:**")
        st.caption(top_customer['Customer'])
        st.caption(f"Spent: {format_currency(top_customer['Total Sales'])}")
    
    with col3:
        st.warning(f"**🔁 Repeat Customer Rate:** {repeat_rate:.1f}%")
        st.caption(f"{repeat_customers:,} of {total_customers:,} customers")


import pandas as pd
