import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    df = pd.read_csv('data/raw.csv')
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d/%m/%Y")
    df["Month-Year"] = df["Order Date"].dt.to_period("M")
    return df

@st.cache_data
def load_cleaned_data():
    df = pd.read_csv('data/cleaned.csv')
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d/%m/%Y")
    df["Month-Year"] = df["Order Date"].dt.to_period("M")
    return df
