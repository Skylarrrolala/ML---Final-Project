# 📊 Professional Sales Analytics Dashboard

A modern, professional-grade sales analytics dashboard built with Streamlit, featuring comprehensive data visualizations, AI-powered insights, and an intuitive user interface.

## ✨ Features

### 🎨 Professional Design
- **Modern UI/UX**: Clean, professional interface with gradient accents and smooth transitions
- **Responsive Layout**: Optimized for various screen sizes
- **Custom Styling**: Beautiful color schemes and typography using Inter font
- **Interactive Charts**: Enhanced Plotly visualizations with custom styling

### 📊 Dashboard Pages

1. **Overview**
   - Key Performance Indicators (KPIs)
   - Sales distribution by category and segment
   - Product performance metrics
   - Interactive filters

2. **Time Analysis**
   - Monthly and yearly sales trends
   - Quarterly performance comparison
   - Day-of-week pattern analysis
   - Year-over-year growth metrics

3. **Geographic Analysis**
   - Regional sales distribution
   - Top states and cities by revenue
   - Geographic performance metrics
   - Interactive maps and charts

4. **Product Analysis**
   - Top-selling products
   - Category and sub-category performance
   - Product hierarchy treemap
   - Revenue distribution

5. **Customer Analysis**
   - Customer segmentation
   - Order frequency distribution
   - Customer lifetime value
   - Repeat customer analytics

6. **AI Insights**
   - Linear regression sales forecasting
   - K-means customer segmentation
   - RFM (Recency, Frequency, Monetary) analysis
   - Predictive analytics

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

1. **Clone the repository** (if not already done):
   ```bash
   cd "/Users/skylarrrr/Documents/Skylar's Space/School/AUPP/Machine Learning/Final Project/ML---Final-Project"
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

   Or install the core dependencies manually:
   ```bash
   pip install streamlit pandas numpy matplotlib seaborn plotly scikit-learn
   ```

### Running the Dashboard

1. **Navigate to the app directory**:
   ```bash
   cd app
   ```

2. **Launch the Streamlit app**:
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Open in browser**:
   - The app will automatically open in your default browser
   - Default URL: `http://localhost:8501`
   - If it doesn't open automatically, copy the URL from the terminal

### Alternative Run Methods

**From project root**:
```bash
streamlit run app/streamlit_app.py
```

**With custom port**:
```bash
streamlit run app/streamlit_app.py --server.port 8502
```

**With auto-reload disabled**:
```bash
streamlit run app/streamlit_app.py --server.runOnSave false
```

## 📁 Project Structure

```
app/
├── streamlit_app.py        # Main application entry point
├── config.py               # Configuration and styling
├── data_loader.py          # Data loading utilities
├── utils.py                # Helper functions and utilities
├── pages/                  # Dashboard pages
│   ├── __init__.py
│   ├── overview.py         # Overview dashboard
│   ├── time_analysis.py    # Time-based analysis
│   ├── geographic_analysis.py  # Geographic insights
│   ├── product_analysis.py # Product performance
│   ├── customer_analysis.py    # Customer analytics
│   └── ai_insights.py      # AI-powered predictions
└── README.md               # This file
```

## 🎯 Key Features Explained

### Professional Styling
- **Custom CSS**: Modern gradient backgrounds, smooth transitions
- **Color Scheme**: Purple gradient primary theme with semantic colors
- **Typography**: Inter font family for clean, professional look
- **Interactive Elements**: Hover effects, smooth animations

### Data Visualization
- **Plotly Charts**: Interactive, professional-grade visualizations
- **Custom Color Scales**: Carefully selected color palettes
- **Responsive Design**: Charts adapt to container width
- **Rich Tooltips**: Detailed hover information

### AI & Analytics
- **Sales Forecasting**: Linear regression with seasonal components
- **Customer Segmentation**: K-means clustering with RFM analysis
- **Performance Metrics**: R², MAPE, MAE for model evaluation
- **Outlier Detection**: IQR-based anomaly identification

## 🛠️ Customization

### Changing Color Theme

Edit `config.py` to modify the color scheme:

```python
# Primary gradient
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

# Update to your preferred colors
background: linear-gradient(135deg, #YOUR_COLOR1 0%, #YOUR_COLOR2 100%);
```

### Adding New Pages

1. Create a new file in `pages/` directory
2. Import in `streamlit_app.py`
3. Add to navigation tabs in `config.py`
4. Implement render function following existing patterns

### Modifying Metrics

Update metric calculations in respective page files:
- `pages/overview.py` - General KPIs
- `pages/time_analysis.py` - Time-based metrics
- And so on...

## 📊 Data Requirements

The dashboard expects CSV files in the `data/` directory:

- `raw.csv` - Main dataset with columns:
  - Order Date, Ship Date
  - Sales, Order ID
  - Customer ID, Customer Name
  - Product Name, Category, Sub-Category
  - Region, State, City
  - Segment, Ship Mode

- `cleaned.csv` - Preprocessed data for AI models

## 🐛 Troubleshooting

### Port Already in Use
```bash
streamlit run streamlit_app.py --server.port 8502
```

### Module Not Found
```bash
pip install -r requirements.txt
```

### Data File Not Found
Ensure you're running from the correct directory:
```bash
cd app
streamlit run streamlit_app.py
```

### Styling Not Applied
Clear Streamlit cache:
```bash
streamlit cache clear
```

## 💡 Tips for Best Experience

1. **Use a modern browser**: Chrome, Firefox, or Edge recommended
2. **Full screen mode**: Press F11 for immersive experience
3. **Explore filters**: Use date range filters for focused analysis
4. **Hover for details**: Charts show detailed tooltips on hover
5. **Responsive**: Works on tablets and large screens

## 📝 Performance Notes

- **Data Caching**: Uses `@st.cache_data` for faster load times
- **Lazy Loading**: Charts render only when tabs are active
- **Optimized Queries**: Efficient pandas operations
- **Minimal Re-renders**: Smart state management

## 🔒 Security Notes

For production deployment:
- Implement authentication (e.g., Streamlit Auth)
- Use environment variables for sensitive data
- Enable HTTPS/SSL
- Set appropriate CORS policies

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Chart Types](https://plotly.com/python/)
- [Color Palettes](https://colorhunt.co/)
- [UI/UX Best Practices](https://streamlit.io/gallery)

## 🤝 Contributing

Feel free to customize and extend this dashboard for your needs!

## 📄 License

This project is part of the ML Final Project coursework.

---

**Developed with ❤️ using Streamlit**
