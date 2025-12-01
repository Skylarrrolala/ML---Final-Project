"""
Product analysis page for Sales Dashboard
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils import create_date_range_filter, format_currency


def render_product_analysis_page(df):
    """Render the product analysis page with enhanced visualizations"""

    with st.expander("🔍 Filter by Date Range", expanded=False):
        _, _, filtered_df = create_date_range_filter(df, key_suffix="product_analysis", label="Analysis Date Range")
        use_filter = st.checkbox("Apply date filter to charts", value=False, key="use_product_analysis_filter")
        
    analysis_df = filtered_df if use_filter else df
    
    # Product KPIs
    st.markdown("### 🛍️ Product Performance Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_products = analysis_df["Product Name"].nunique()
    total_categories = analysis_df["Category"].nunique()
    total_subcategories = analysis_df["Sub-Category"].nunique()
    top_product = analysis_df.groupby("Product Name")["Sales"].sum().idxmax()
    
    with col1:
        st.metric("📦 Total Products", f"{total_products:,}")
    
    with col2:
        st.metric("📂 Categories", total_categories)
    
    with col3:
        st.metric("📋 Sub-Categories", total_subcategories)
    
    with col4:
        top_product_short = top_product[:20] + "..." if len(top_product) > 20 else top_product
        st.metric("🏆 Best Seller", "")
        st.caption(top_product_short)
    
    st.markdown("<div class='space-medium'></div>", unsafe_allow_html=True)
    
    # Top Products
    st.markdown("### 🌟 Top 15 Products by Sales")
    
    product_sales = analysis_df.groupby("Product Name")["Sales"].sum().reset_index()
    product_sales = product_sales.sort_values(by="Sales", ascending=False).head(15)
    
    # Truncate long product names for display
    product_sales["Display Name"] = product_sales["Product Name"].apply(
        lambda x: x[:40] + "..." if len(x) > 40 else x
    )
    
    fig = go.Figure(go.Bar(
        y=product_sales["Display Name"][::-1],
        x=product_sales["Sales"][::-1],
        orientation='h',
        marker=dict(
            color=product_sales["Sales"][::-1],
            colorscale='Sunset',
            showscale=True,
            colorbar=dict(title="Sales ($)")
        ),
        text=product_sales["Sales"][::-1].apply(lambda x: f'${x/1000:.1f}K'),
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Sales: $%{x:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        xaxis_title="Sales ($)",
        yaxis_title="",
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<div class='space-medium'></div>", unsafe_allow_html=True)
    
    # Category Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Category Distribution")
        
        category_sales = analysis_df.groupby("Category").agg({
            "Sales": "sum",
            "Product Name": "nunique"
        }).reset_index()
        category_sales.columns = ["Category", "Sales", "Products"]
        
        fig = go.Figure(data=[go.Pie(
            labels=category_sales["Category"],
            values=category_sales["Sales"],
            hole=0.4,
            marker=dict(colors=['#667eea', '#764ba2', '#f093fb']),
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Sales: $%{value:,.0f}<br>Products: %{customdata}<extra></extra>',
            customdata=category_sales["Products"]
        )])
        
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
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
        st.markdown("### 📈 Sub-Category Performance")
        
        subcategory_sales = analysis_df.groupby("Sub-Category")["Sales"].sum().reset_index()
        subcategory_sales = subcategory_sales.sort_values("Sales", ascending=False).head(10)
        
        fig = go.Figure(go.Bar(
            x=subcategory_sales["Sub-Category"],
            y=subcategory_sales["Sales"],
            marker=dict(
                color=subcategory_sales["Sales"],
                colorscale='Tealgrn',
                showscale=False
            ),
            text=subcategory_sales["Sales"].apply(lambda x: f'${x/1000:.0f}K'),
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Sales: $%{y:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            xaxis_title="",
            yaxis_title="Sales ($)",
            height=350,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis={'tickangle': -45}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<div class='space-medium'></div>", unsafe_allow_html=True)
    
    # Product Hierarchy Treemap
    st.markdown("### 🗂️ Product Hierarchy View")
    
    category_counts = analysis_df.groupby(["Category", "Sub-Category"]).agg({
        "Sales": "sum",
        "Product Name": "nunique"
    }).reset_index()
    category_counts.columns = ["Category", "Sub-Category", "Sales", "Products"]
    
    fig = px.treemap(
        category_counts,
        path=["Category", "Sub-Category"],
        values="Sales",
        color="Sales",
        color_continuous_scale='Purples',
        hover_data={"Products": True},
        custom_data=["Products"]
    )
    
    fig.update_traces(
        textinfo="label+value+percent parent",
        hovertemplate='<b>%{label}</b><br>Sales: $%{value:,.0f}<br>Products: %{customdata[0]}<extra></extra>'
    )
    
    fig.update_layout(
        height=450,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Product Insights
    st.markdown("<div class='space-medium'></div>", unsafe_allow_html=True)
    st.markdown("### 💡 Product Insights")
    
    col1, col2, col3 = st.columns(3)
    
    top_category = category_sales.sort_values("Sales", ascending=False).iloc[0]
    top_subcategory = subcategory_sales.iloc[0]
    top_product_sales = product_sales.iloc[0]
    
    with col1:
        st.success(f"**🏆 Top Category:** {top_category['Category']}")
        st.caption(f"Sales: {format_currency(top_category['Sales'])}")
        st.caption(f"Products: {top_category['Products']}")
    
    with col2:
        st.info(f"**⭐ Top Sub-Category:** {top_subcategory['Sub-Category']}")
        st.caption(f"Sales: {format_currency(top_subcategory['Sales'])}")
    
    with col3:
        st.warning(f"**🌟 Best Product:**")
        st.caption(top_product[:30] + "..." if len(top_product) > 30 else top_product)
        st.caption(f"Sales: {format_currency(top_product_sales['Sales'])}")
