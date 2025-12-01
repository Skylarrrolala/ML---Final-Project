"""
Utility functions for the Sales Dashboard
"""
import streamlit as st
import pandas as pd
from datetime import datetime


def create_date_range_filter(df, key_suffix="", label="Data View Range"):
    """Create a date range filter with professional styling"""
    latest_date = df["Order Date"].max()
    latest_month = latest_date.month
    latest_year = latest_date.year

    first_day_of_month = datetime(latest_year, latest_month, 1).date()
    
    if latest_month == 12:
        last_day_of_month = datetime(latest_year + 1, 1, 1).date()
    else:
        last_day_of_month = datetime(latest_year, latest_month + 1, 1).date()
    last_day_of_month = (last_day_of_month - pd.Timedelta(days=1))

    date_key = f"date_range_{key_suffix}" if key_suffix else "date_range_default"
    
    date_range = st.date_input(
        label,
        value=(first_day_of_month, last_day_of_month),
        format="MM/DD/YYYY",
        key=date_key
    )
    
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range
    
    filtered_data = df[(df["Order Date"].dt.date >= start_date) & 
                        (df["Order Date"].dt.date <= end_date)]
    
    return start_date, end_date, filtered_data


def format_currency(value):
    """Format number as currency with proper formatting"""
    if value >= 1000000:
        return f"${value/1000000:.2f}M"
    elif value >= 1000:
        return f"${value/1000:.1f}K"
    else:
        return f"${value:,.2f}"


def format_number(value):
    """Format large numbers with K/M suffixes"""
    if value >= 1000000:
        return f"{value/1000000:.2f}M"
    elif value >= 1000:
        return f"{value/1000:.1f}K"
    else:
        return f"{value:,.0f}"


def create_metric_card(title, value, color="primary"):
    """
    Create a styled metric card with custom colors
    
    Args:
        title: The metric title
        value: The metric value
        color: Color theme (primary, success, info, warning, danger)
    """
    colors = {
        "primary": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        "success": "linear-gradient(135deg, #48c774 0%, #3eab63 100%)",
        "info": "linear-gradient(135deg, #3273dc 0%, #2366d1 100%)",
        "warning": "linear-gradient(135deg, #ffdd57 0%, #ffc107 100%)",
        "danger": "linear-gradient(135deg, #f14668 0%, #e01e37 100%)"
    }
    
    text_colors = {
        "primary": "#ffffff",
        "success": "#ffffff",
        "info": "#ffffff",
        "warning": "#000000",
        "danger": "#ffffff"
    }
    
    bg_color = colors.get(color, colors["primary"])
    text_color = text_colors.get(color, "#ffffff")
    
    return f"""
    <div style="
        background: {bg_color};
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: all 0.3s ease;
    ">
        <div style="
            color: {text_color};
            opacity: 0.9;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.5rem;
        ">{title}</div>
        <div style="
            color: {text_color};
            font-size: 2rem;
            font-weight: 700;
            line-height: 1.2;
        ">{value}</div>
    </div>
    """


def create_info_box(title, content, icon="ℹ️"):
    """Create a styled information box"""
    return f"""
    <div style="
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: 4px solid #667eea;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    ">
        <div style="
            font-size: 1.1rem;
            font-weight: 600;
            color: #2d3748;
            margin-bottom: 0.5rem;
        ">{icon} {title}</div>
        <div style="
            color: #4a5568;
            font-size: 0.95rem;
            line-height: 1.6;
        ">{content}</div>
    </div>
    """
