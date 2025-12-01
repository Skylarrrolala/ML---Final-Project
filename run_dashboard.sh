#!/bin/bash

# 🚀 Quick Start Script for Sales Analytics Dashboard
# This script helps you launch the professional Streamlit dashboard

echo "📊 Sales Analytics Dashboard - Quick Start"
echo "==========================================="
echo ""

# Check if we're in the correct directory
if [ ! -f "streamlit_app.py" ]; then
    echo "⚠️  Not in the app directory. Navigating..."
    cd app 2>/dev/null || {
        echo "❌ Error: Could not find app directory"
        echo "Please run this script from the project root or app directory"
        exit 1
    }
fi

echo "✅ Found app directory"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    echo "Please install Python 3.9 or higher"
    exit 1
fi

echo "✅ Python is installed: $(python3 --version)"
echo ""

# Check if streamlit is installed
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "⚠️  Streamlit is not installed"
    echo "Installing required packages..."
    pip3 install streamlit pandas numpy matplotlib seaborn plotly scikit-learn
    echo ""
fi

echo "✅ All dependencies are ready"
echo ""

# Launch the dashboard
echo "🚀 Launching Sales Analytics Dashboard..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "The dashboard will open in your browser at:"
echo "👉 http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run streamlit
streamlit run streamlit_app.py
