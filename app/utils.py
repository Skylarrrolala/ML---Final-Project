"""
Utility functions for the Sales Dashboard
"""
import streamlit as st
import pandas as pd
from datetime import datetime


def create_date_range_filter(df, key_suffix="", label="Data View Range"):
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
