"""
AI-powered insights page for Sales Dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import matplotlib.pyplot as plt


def create_features(data):
    """Create features for linear regression"""
    features_df = data.copy()
    features_df['month_sequence'] = range(1, len(features_df) + 1)
    features_df['month'] = features_df['Order Date'].dt.month
    month_dummies = pd.get_dummies(features_df['month'], prefix='month', drop_first=True)
    features_df = pd.concat([features_df, month_dummies], axis=1)
    return features_df


def generate_forecasts(model, last_date, data_length, feature_columns, num_months=3):
    """Generate future sales forecasts"""
    forecasts = []
    
    # Get the last month sequence number
    last_month_seq = data_length
    
    for i in range(1, num_months + 1):
        # Calculate future date
        future_date = last_date + pd.DateOffset(months=i)
        
        # Create feature vector
        month_seq = last_month_seq + i
        month_num = future_date.month
        
        # Create feature row as dictionary with proper feature names
        feature_dict = {'month_sequence': month_seq}
        
        # Add month dummies (months 2-12, since January is dropped)
        for month in range(2, 13):
            month_col = f'month_{month}'
            if month_col in feature_columns:
                feature_dict[month_col] = 1 if month_num == month else 0
        
        # Create DataFrame with proper feature names
        feature_df = pd.DataFrame([feature_dict])
        # Ensure all required columns are present
        for col in feature_columns:
            if col not in feature_df.columns:
                feature_df[col] = 0
        
        # Reorder columns to match training data
        feature_df = feature_df[feature_columns]
        
        # Make prediction
        prediction = model.predict(feature_df)[0]
        
        forecasts.append({
            'Date': future_date,
            'Predicted_Sales': prediction,
            'Month': future_date.strftime('%B %Y')
        })
    
    return pd.DataFrame(forecasts)


def detect_outliers_iqr(df, columns):
    """Detect outliers using Interquartile Range (IQR) method"""
    outliers_indices = set()
    
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Find outliers
        column_outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index
        outliers_indices.update(column_outliers)
    
    return list(outliers_indices)


def perform_kmeans_clustering(df):
    """Perform K-means clustering on customer data using RFM analysis"""
    # Calculate RFM metrics
    reference_date = df['Order Date'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('Customer ID').agg({
        'Order Date': lambda x: (reference_date - x.max()).days,  # Recency
        'Order ID': 'count',                                      # Frequency
        'Sales': 'sum'                                           # Monetary
    }).rename(columns={
        'Order Date': 'Recency',
        'Order ID': 'Frequency', 
        'Sales': 'Monetary'
    })
    
    # Detect and remove outliers
    outlier_indices = detect_outliers_iqr(rfm, ['Recency', 'Frequency', 'Monetary'])
    rfm_clean = rfm.drop(outlier_indices)
    
    # Standardize the data
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_clean)
    
    # Find optimal k using silhouette analysis
    k_range = range(2, 8)
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(rfm_scaled)
        silhouette_avg = silhouette_score(rfm_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    # Use optimal k
    optimal_k = k_range[np.argmax(silhouette_scores)]
    
    # Perform final clustering
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans_final.fit_predict(rfm_scaled)
    
    # Add cluster labels to RFM data
    rfm_clean['Cluster'] = clusters
    
    return rfm_clean, optimal_k, max(silhouette_scores), len(outlier_indices)


def interpret_clusters(rfm_data):
    """Interpret cluster characteristics and assign business-friendly names"""
    cluster_summary = rfm_data.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    }).round(2)
    
    cluster_sizes = rfm_data['Cluster'].value_counts().sort_index()
    interpretations = {}
    
    for cluster in cluster_summary.index:
        size = cluster_sizes[cluster]
        pct = (size / len(rfm_data)) * 100
        
        # Determine cluster characteristics based on RFM values
        recency = cluster_summary.loc[cluster, 'Recency']
        frequency = cluster_summary.loc[cluster, 'Frequency']
        monetary = cluster_summary.loc[cluster, 'Monetary']

        if cluster == 2:
            label = "Low-Value Dormant Customerss"
            description = "Low-Value and inactive customers"
        elif cluster == 1:
            label = "Low-Value Lost Customers"
            description = "Low-value and churned customers"
        else:
            label = "High-Value Dormant Customers"
            description = "High-value but inactive customers"
        
        
        interpretations[cluster] = {
            'label': label,
            'description': description,
            'size': size,
            'percentage': pct,
            'recency': recency,
            'frequency': frequency,
            'monetary': monetary
        }
    
    return interpretations


def render_ai_insights_page(df):
    """Render the AI-powered insights page"""
    st.subheader("AI-Powered Insights")
    
    # Sales Forecasting Section
    with st.container(border=True):
        st.markdown("#### 📈 Linear Regression Sales Forecast")
        forecast_months = st.slider(
            "Months to forecast:",
            min_value=1,
            max_value=24,
            value=3,
            step=1,
            help="Choose how many months ahead you want to predict sales"
        )
        
        # Prepare monthly sales data
        monthly_sales = df.groupby(df['Order Date'].dt.to_period('M'))['Sales'].sum().reset_index()
        monthly_sales['Order Date'] = monthly_sales['Order Date'].dt.to_timestamp()
        monthly_sales = monthly_sales.sort_values('Order Date').reset_index(drop=True)
        
        # Create features
        data_with_features = create_features(monthly_sales)
        
        # Prepare data for modeling
        feature_columns = [col for col in data_with_features.columns if col.startswith('month_')]
        feature_columns.insert(0, 'month_sequence')
        X = data_with_features[feature_columns]
        y = data_with_features['Sales']
        
        # Split data for training and testing (use last 12 months for testing)
        train_size = len(X) - 12
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train the linear regression model
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        
        # Make predictions on test set
        y_pred_test = lr_model.predict(X_test)
        
        # Calculate performance metrics
        mae = mean_absolute_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
        
        # Generate forecast based on slider value
        last_date = monthly_sales['Order Date'].max()
        forecast_df = generate_forecasts(lr_model, last_date, len(data_with_features), feature_columns, forecast_months)
        
        # Create visualization
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Prepare data for plotting
            historical_data = monthly_sales.copy()
            historical_data['Type'] = 'Historical'
            
            # Add test predictions
            test_dates = data_with_features['Order Date'][train_size:]
            test_data = pd.DataFrame({
                'Order Date': test_dates,
                'Sales': y_pred_test,
                'Type': 'Model Validation'
            })
            
            # Add future predictions
            future_data = pd.DataFrame({
                'Order Date': forecast_df['Date'],
                'Sales': forecast_df['Predicted_Sales'],
                'Type': 'Future Forecast'
            })
            
            # Combine all data
            plot_data = pd.concat([
                historical_data[['Order Date', 'Sales', 'Type']],
                test_data,
                future_data
            ], ignore_index=True)
            
            # Create the plot
            fig = go.Figure()
            
            # Historical data
            historical = plot_data[plot_data['Type'] == 'Historical']
            fig.add_trace(go.Scatter(
                x=historical['Order Date'],
                y=historical['Sales'],
                mode='lines+markers',
                name='Historical Sales',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=4)
            ))
            
            # Model validation
            validation = plot_data[plot_data['Type'] == 'Model Validation']
            fig.add_trace(go.Scatter(
                x=validation['Order Date'],
                y=validation['Sales'],
                mode='lines+markers',
                name='Model Validation',
                line=dict(color='#ff7f0e', width=2, dash='dash'),
                marker=dict(size=6, symbol='square')
            ))
            
            # Future forecast
            forecast = plot_data[plot_data['Type'] == 'Future Forecast']
            fig.add_trace(go.Scatter(
                x=forecast['Order Date'],
                y=forecast['Sales'],
                mode='lines+markers',
                name='Future Forecast',
                line=dict(color='#d62728', width=3),
                marker=dict(size=8, symbol='diamond')
            ))
            
            fig.update_layout(
                title=f"Sales Forecast - Next {forecast_months} Month{'s' if forecast_months != 1 else ''}",
                xaxis_title="Date",
                yaxis_title="Sales ($)",
                hovermode='x unified',
                showlegend=True,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("##### 📊 Model Performance")
            st.metric("R-squared (R²)", f"{r2:.3f}")
            st.metric("MAPE", f"{mape:.1f}%")
            st.metric("MAE", f"${mae:,.0f}")
            
            st.markdown(f"##### 🔮 {forecast_months}-Month Forecast")
            
            # Show individual months (limit display to first 6 for UI purposes)
            display_months = forecast_months
            for i, (_, row) in enumerate(forecast_df.head(display_months).iterrows()):
                st.metric(
                    row['Month'], 
                    f"${row['Predicted_Sales']:,.0f}"
                )

    # Customer Segmentation Section
    with st.container(border=True):
        st.markdown("#### 👥 K-Means Customer Segmentation")
        
        with st.spinner("Performing customer segmentation analysis..."):
            # Perform clustering
            rfm_data, optimal_k, silhouette_score_val, outliers_removed = perform_kmeans_clustering(df)
            cluster_interpretations = interpret_clusters(rfm_data)
        
        # Display clustering results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create RFM scatter plot colored by clusters
            # Add cluster labels to the data for better legend
            rfm_with_labels = rfm_data.copy()
            rfm_with_labels['Cluster_Label'] = rfm_with_labels['Cluster'].map(
                {cluster_id: info['label'] for cluster_id, info in cluster_interpretations.items()}
            )
            
            fig = px.scatter_3d(
                rfm_with_labels, 
                x='Recency', 
                y='Frequency', 
                z='Monetary',
                color='Cluster_Label',
                title="Customer Segments (RFM Analysis)",
                labels={
                    'Recency': 'Days Since Last Purchase',
                    'Frequency': 'Number of Orders',
                    'Monetary': 'Total Spent ($)',
                    'Cluster_Label': 'Customer Segment'
                }
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Cluster size distribution
            cluster_sizes = rfm_data['Cluster'].value_counts().sort_index()
            cluster_labels = [cluster_interpretations[i]['label'] for i in cluster_sizes.index]
            
            fig_pie = px.pie(
                values=cluster_sizes.values,
                names=cluster_labels,
                title="Customer Segment Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.markdown("##### 📈 Clustering Metrics")
            st.metric("Optimal Clusters", optimal_k)
            st.metric("Silhouette Score", f"{silhouette_score_val:.3f}")
            st.metric("Customers Analyzed", len(rfm_data))
            st.metric("Outliers Removed", outliers_removed)
            
            st.markdown("##### 🎯 Customer Segments")
            
            for cluster_id, info in cluster_interpretations.items():
                with st.expander(f"**{info['label']}** ({info['size']} customers)"):
                    st.write(f"**Description:** {info['description']}")
                    st.write(f"**Size:** {info['percentage']:.1f}% of customers")
                    st.write(f"**Avg Recency:** {info['recency']:.1f} days")
                    st.write(f"**Avg Frequency:** {info['frequency']:.1f} orders")
                    st.write(f"**Avg Monetary:** ${info['monetary']:,.2f}")
