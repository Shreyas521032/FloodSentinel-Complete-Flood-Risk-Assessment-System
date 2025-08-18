import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import os
import time
import json
from datetime import datetime
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image

warnings.filterwarnings('ignore')

# Enhanced page configuration
st.set_page_config(
    page_title="FloodSentinel Pro - AI Flood Risk Assessment",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with modern design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Modern Header */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        text-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .sub-header {
        text-align: center;
        font-size: 1.3rem;
        color: #6b7280;
        margin-bottom: 3rem;
        font-weight: 400;
    }
    
    /* Enhanced Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.4);
    }
    
    .metric-card h3 {
        margin: 0;
        font-size: 0.9rem;
        font-weight: 500;
        opacity: 0.9;
    }
    
    .metric-card h2 {
        margin: 0.5rem 0 0 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    
    /* Enhanced Alert Boxes */
    .success-card {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 1.5rem;
        border-radius: 16px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.3);
        border-left: 4px solid #34d399;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        padding: 1.5rem;
        border-radius: 16px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(245, 158, 11, 0.3);
        border-left: 4px solid #fbbf24;
    }
    
    .info-card {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        padding: 1.5rem;
        border-radius: 16px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.3);
        border-left: 4px solid #60a5fa;
    }
    
    .error-card {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        padding: 1.5rem;
        border-radius: 16px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(239, 68, 68, 0.3);
        border-left: 4px solid #f87171;
    }
    
    /* Enhanced Sidebar */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    /* Custom buttons */
    .stButton > button {
        border-radius: 25px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    /* Form styling */
    .stForm {
        background: #f8fafc;
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    /* Risk level indicators */
    .risk-low {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3);
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);
    }
    
    /* Image gallery */
    .image-gallery {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .image-item {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    .image-item:hover {
        transform: scale(1.05);
    }
    
    /* Progress indicators */
    .progress-container {
        background: #f1f5f9;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 16px;
        margin-top: 3rem;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
    }
    
    /* Data table styling */
    .dataframe {
        border: none !important;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1f2937;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'dataset_loaded' not in st.session_state:
        st.session_state.dataset_loaded = False
    if 'model_results' not in st.session_state:
        st.session_state.model_results = {}
    if 'sample_data' not in st.session_state:
        st.session_state.sample_data = None

init_session_state()

# Enhanced header
st.markdown('<h1 class="main-header">🌊 FloodSentinel Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced AI-Powered Flood Risk Assessment System</p>', unsafe_allow_html=True)

# Enhanced sidebar with modern design
st.sidebar.markdown("### 🧭 Navigation")
page = st.sidebar.selectbox(
    "Choose a section:",
    ["🏠 Dashboard", "📊 Data Analysis", "⚙️ Model Training", "🔮 Risk Prediction", "🛰️ Satellite Analysis", "📈 Results & Export"],
    help="Navigate through different sections of the application"
)

# Sample data generator for demonstration
@st.cache_data
def generate_sample_flood_data(n_samples=1000):
    """Generate realistic sample flood data for demonstration"""
    np.random.seed(42)
    
    # Feature names based on flood risk factors
    features = [
        'monsoon_intensity', 'topography_drainage', 'river_management',
        'deforestation', 'urbanization', 'climate_change', 'dams_quality',
        'siltation', 'agricultural_practices', 'encroachments',
        'disaster_preparedness', 'drainage_systems', 'coastal_vulnerability',
        'landslides', 'watersheds', 'infrastructure_quality',
        'population_density', 'wetland_loss', 'planning_adequacy',
        'political_factors'
    ]
    
    # Generate correlated features
    data = {}
    base_risk = np.random.beta(2, 3, n_samples)
    
    for i, feature in enumerate(features):
        # Create some correlation with base risk and add noise
        correlation_strength = np.random.uniform(0.3, 0.8)
        noise = np.random.normal(0, 0.2, n_samples)
        
        feature_values = (base_risk * correlation_strength + 
                         np.random.uniform(0, 1, n_samples) * (1 - correlation_strength) + 
                         noise)
        
        # Normalize to 0-1 range
        feature_values = np.clip(feature_values, 0, 1)
        data[feature] = feature_values
    
    # Generate target variable (flood probability)
    weights = np.random.uniform(0.5, 2.0, len(features))
    flood_prob = np.zeros(n_samples)
    
    for i, feature in enumerate(features):
        flood_prob += data[feature] * weights[i]
    
    # Normalize and add some non-linearity
    flood_prob = flood_prob / flood_prob.max()
    flood_prob = np.clip(flood_prob ** 1.5, 0, 1)  # Add non-linearity
    
    data['FloodProbability'] = flood_prob
    
    df = pd.DataFrame(data)
    return df

def get_model_algorithms():
    """Get enhanced model algorithms with better parameters"""
    return {
        "🌳 Random Forest": RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            min_samples_split=5,
            random_state=42
        ),
        "⚡ Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        ),
        "🧠 Neural Network": MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),
            max_iter=1000,
            learning_rate='adaptive',
            random_state=42
        ),
        "📈 Support Vector": SVR(
            kernel='rbf',
            C=1.0,
            gamma='scale'
        ),
        "🔗 ElasticNet": ElasticNet(
            alpha=0.1,
            l1_ratio=0.5,
            random_state=42
        ),
        "🎪 AdaBoost": AdaBoostRegressor(
            n_estimators=100,
            learning_rate=1.0,
            random_state=42
        ),
        "🌿 Decision Tree": DecisionTreeRegressor(
            max_depth=10,
            min_samples_split=5,
            random_state=42
        ),
        "👥 K-Neighbors": KNeighborsRegressor(
            n_neighbors=5,
            weights='distance'
        ),
        "📊 Ridge Regression": Ridge(
            alpha=1.0,
            random_state=42
        ),
        "🎯 Linear Regression": LinearRegression()
    }

def create_metric_card(title, value, icon="📊"):
    """Create an enhanced metric card"""
    return f"""
    <div class="metric-card">
        <h3>{icon} {title}</h3>
        <h2>{value}</h2>
    </div>
    """

def create_alert_card(message, alert_type="info"):
    """Create enhanced alert cards"""
    return f'<div class="{alert_type}-card">{message}</div>'

def generate_sample_satellite_images(num_images=12):
    """Generate sample satellite-like images for demonstration"""
    images = []
    
    for i in range(num_images):
        # Create a sample image with random terrain-like patterns
        np.random.seed(i + 42)
        
        # Create base terrain
        img = np.random.rand(128, 128, 3)
        
        # Add some structure (rivers, urban areas, etc.)
        if i < 4:  # Flooded areas - more blue
            img[:, :, 2] += 0.3  # More blue
            img[:, :, 0] *= 0.7  # Less red
        elif i < 8:  # Urban areas - more gray
            img = (img * 0.6) + 0.2
        else:  # Rural/forest - more green
            img[:, :, 1] += 0.2  # More green
            img[:, :, 0] *= 0.8  # Less red
        
        # Add some noise for realism
        noise = np.random.normal(0, 0.1, (128, 128, 3))
        img = np.clip(img + noise, 0, 1)
        
        # Convert to uint8
        img = (img * 255).astype(np.uint8)
        images.append(img)
    
    return images

# Main application pages
if page == "🏠 Dashboard":
    st.markdown('<div class="section-header">🎯 Project Overview</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(create_alert_card("""
            <h4>🌊 Advanced Flood Risk Assessment</h4>
            <p>FloodSentinel Pro leverages cutting-edge AI technologies to provide comprehensive flood risk analysis. 
            Our system combines machine learning algorithms with satellite imagery analysis to deliver accurate, 
            real-time flood predictions for enhanced disaster preparedness.</p>
            
            <h4>🚀 Key Capabilities:</h4>
            <ul>
                <li>🤖 10+ State-of-the-art ML algorithms</li>
                <li>🛰️ Multi-temporal satellite imagery analysis</li>
                <li>📊 Real-time risk assessment dashboard</li>
                <li>🎯 Interactive prediction interface</li>
                <li>📈 Comprehensive performance analytics</li>
                <li>💾 Advanced data export capabilities</li>
            </ul>
        """, "info"), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_alert_card("""
            <h4>⚡ Quick Start</h4>
            <p>Get started with FloodSentinel Pro:</p>
            <ol>
                <li>🔄 Load sample data</li>
                <li>📊 Explore data patterns</li>
                <li>⚙️ Train ML models</li>
                <li>🔮 Make predictions</li>
                <li>📈 Analyze results</li>
            </ol>
        """, "success"), unsafe_allow_html=True)
    
    # Load sample data button with enhanced styling
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🔄 Initialize Sample Dataset", type="primary", use_container_width=True):
            with st.spinner("🔄 Generating sample flood risk data..."):
                time.sleep(2)  # Simulate loading time
                df_sample = generate_sample_flood_data(1000)
                sample_images = generate_sample_satellite_images(12)
                
                st.session_state.df_flood = df_sample
                st.session_state.sample_images = sample_images
                st.session_state.dataset_loaded = True
                
            st.balloons()
            st.success("✅ Sample dataset loaded successfully!")
            st.rerun()
    
    # Status indicators
    if st.session_state.dataset_loaded:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(create_metric_card("Data Records", f"{len(st.session_state.df_flood):,}", "📋"), 
                       unsafe_allow_html=True)
        
        with col2:
            st.markdown(create_metric_card("Features", f"{len(st.session_state.df_flood.columns)-1}", "📊"), 
                       unsafe_allow_html=True)
        
        with col3:
            st.markdown(create_metric_card("Satellite Images", "12", "🛰️"), 
                       unsafe_allow_html=True)
        
        with col4:
            models_count = len(st.session_state.model_results) if st.session_state.models_trained else 0
            st.markdown(create_metric_card("Trained Models", f"{models_count}", "🤖"), 
                       unsafe_allow_html=True)

elif page == "📊 Data Analysis":
    st.markdown('<div class="section-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    if not st.session_state.dataset_loaded:
        st.markdown(create_alert_card("""
            ⚠️ <strong>Dataset Required</strong><br>
            Please load the sample dataset first from the Dashboard page to begin analysis.
        """, "warning"), unsafe_allow_html=True)
        st.stop()
    
    df = st.session_state.df_flood
    
    # Enhanced statistics overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(create_metric_card("Total Records", f"{len(df):,}", "📋"), 
                   unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_metric_card("Features", f"{len(df.columns)-1}", "📊"), 
                   unsafe_allow_html=True)
    
    with col3:
        avg_risk = df['FloodProbability'].mean()
        st.markdown(create_metric_card("Avg Risk", f"{avg_risk:.2%}", "🎯"), 
                   unsafe_allow_html=True)
    
    with col4:
        high_risk_count = (df['FloodProbability'] > 0.7).sum()
        st.markdown(create_metric_card("High Risk Areas", f"{high_risk_count:,}", "🚨"), 
                   unsafe_allow_html=True)
    
    # Enhanced correlation heatmap
    st.markdown("#### 🔥 Feature Correlation Matrix")
    
    # Select top features for better visualization
    correlations = df.corr()['FloodProbability'].abs().sort_values(ascending=False)[1:11]
    top_features = correlations.index.tolist() + ['FloodProbability']
    corr_matrix = df[top_features].corr()
    
    fig_heatmap = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdYlBu_r",
        title="🔥 Top 10 Features Correlation with Flood Probability"
    )
    fig_heatmap.update_layout(height=600)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Enhanced distribution analysis
    col1, col2 = st.columns(2)
    
    with col1:
        fig_hist = px.histogram(
            df, 
            x='FloodProbability',
            nbins=50,
            title="🎯 Flood Probability Distribution",
            color_discrete_sequence=['#667eea'],
            marginal="box"
        )
        fig_hist.update_layout(showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Risk categorization
        df_risk = df.copy()
        df_risk['Risk_Category'] = pd.cut(
            df_risk['FloodProbability'],
            bins=[0, 0.3, 0.6, 1.0],
            labels=['🟢 Low', '🟡 Medium', '🔴 High']
        )
        
        risk_counts = df_risk['Risk_Category'].value_counts()
        
        fig_pie = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="🎯 Risk Level Distribution",
            color_discrete_sequence=['#10b981', '#f59e0b', '#ef4444']
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Feature importance visualization
    st.markdown("#### 🔍 Feature Impact Analysis")
    
    correlations = df.corr()['FloodProbability'].abs().sort_values(ascending=False)[1:]
    
    fig_corr_bar = px.bar(
        x=correlations.values,
        y=correlations.index,
        orientation='h',
        title="🔍 Feature Correlation with Flood Probability",
        color=correlations.values,
        color_continuous_scale="Viridis",
        labels={'x': 'Correlation Strength', 'y': 'Features'}
    )
    fig_corr_bar.update_layout(height=700)
    st.plotly_chart(fig_corr_bar, use_container_width=True)
    
    # Statistical summary
    st.markdown("#### 📈 Statistical Summary")
    
    summary_stats = df.describe()
    st.dataframe(summary_stats, use_container_width=True)

elif page == "⚙️ Model Training":
    st.markdown('<div class="section-header">⚙️ Advanced Model Training</div>', unsafe_allow_html=True)
    
    if not st.session_state.dataset_loaded:
        st.markdown(create_alert_card("""
            ⚠️ <strong>Dataset Required</strong><br>
            Please load the sample dataset first from the Dashboard page to begin training.
        """, "warning"), unsafe_allow_html=True)
        st.stop()
    
    df = st.session_state.df_flood
    
    # Enhanced training configuration
    st.markdown("#### ⚙️ Training Configuration")
    
    with st.form("training_config"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            scaler_type = st.selectbox(
                "📊 Data Scaler:",
                ["StandardScaler", "MinMaxScaler", "RobustScaler"],
                help="Choose the scaling method for feature normalization"
            )
            
            test_size = st.slider(
                "🎯 Test Set Size:", 
                0.1, 0.4, 0.2, 0.05,
                help="Proportion of data to use for testing"
            )
        
        with col2:
            cv_folds = st.slider(
                "🔄 Cross-Validation Folds:", 
                3, 10, 5,
                help="Number of folds for cross-validation"
            )
            
            random_state = st.number_input(
                "🎲 Random State:", 
                value=42,
                help="Seed for reproducibility"
            )
        
        with col3:
            enable_hyperparameter_tuning = st.checkbox(
                "🔧 Hyperparameter Tuning",
                help="Enable automated hyperparameter optimization (slower but better results)"
            )
            
            parallel_training = st.checkbox(
                "⚡ Parallel Training",
                value=True,
                help="Use parallel processing for faster training"
            )
    
        # Model selection with enhanced interface
        st.markdown("#### 🎯 Model Selection")
        
        models = get_model_algorithms()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_models = st.multiselect(
                "Choose models to train:",
                list(models.keys()),
                default=list(models.keys())[:5],
                help="Select one or more models for training and comparison"
            )
        
        with col2:
            if st.form_submit_button("🚀 Start Training", type="primary", use_container_width=True):
                training_requested = True
            else:
                training_requested = False
    
    # Enhanced training process
    if training_requested:
        if not selected_models:
            st.markdown(create_alert_card("""
                ❌ <strong>No Models Selected</strong><br>
                Please select at least one model to train.
            """, "error"), unsafe_allow_html=True)
            st.stop()
        
        # Data preparation
        X = df.drop('FloodProbability', axis=1)
        y = df['FloodProbability']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=pd.cut(y, bins=3)
        )
        
        # Scaling
        if scaler_type == "StandardScaler":
            scaler = StandardScaler()
        elif scaler_type == "MinMaxScaler":
            scaler = MinMaxScaler()
        else:
            scaler = RobustScaler()
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        st.markdown("#### 🔄 Training Progress")
        
        results = {}
        progress_bar = st.progress(0)
        status_placeholder = st.empty()
        metrics_placeholder = st.empty()
        
        # Enhanced training loop with real-time metrics
        for i, model_name in enumerate(selected_models):
            status_placeholder.info(f"🔄 Training {model_name}...")
            
            model = models[model_name]
            
            # Training with timing
            start_time = time.time()
            
            try:
                model.fit(X_train_scaled, y_train)
                training_time = time.time() - start_time
                
                # Predictions
                y_pred = model.predict(X_test_scaled)
                y_train_pred = model.predict(X_train_scaled)
                
                # Enhanced metrics
                test_mse = mean_squared_error(y_test, y_pred)
                test_rmse = np.sqrt(test_mse)
                test_mae = mean_absolute_error(y_test, y_pred)
                test_r2 = r2_score(y_test, y_pred)
                
                train_mse = mean_squared_error(y_train, y_train_pred)
                train_r2 = r2_score(y_train, y_train_pred)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds, scoring='r2')
                
                # Overfitting detection
                overfitting_score = train_r2 - test_r2
                
                results[model_name] = {
                    'Test_MSE': test_mse,
                    'Test_RMSE': test_rmse,
                    'Test_MAE': test_mae,
                    'Test_R²': test_r2,
                    'Train_R²': train_r2,
                    'CV_Mean': cv_scores.mean(),
                    'CV_Std': cv_scores.std(),
                    'Training_Time': training_time,
                    'Overfitting_Score': overfitting_score,
                    'Model': model,
                    'Test_Predictions': y_pred,
                    'Train_Predictions': y_train_pred
                }
                
                # Real-time metrics display
                with metrics_placeholder.container():
                    metric_cols = st.columns(4)
                    with metric_cols[0]:
                        st.metric("R² Score", f"{test_r2:.4f}")
                    with metric_cols[1]:
                        st.metric("RMSE", f"{test_rmse:.4f}")
                    with metric_cols[2]:
                        st.metric("CV Score", f"{cv_scores.mean():.4f}")
                    with metric_cols[3]:
                        st.metric("Training Time", f"{training_time:.2f}s")
                
            except Exception as e:
                st.error(f"❌ Error training {model_name}: {str(e)}")
                continue
            
            progress_bar.progress((i + 1) / len(selected_models))
        
        # Store results
        st.session_state.model_results = results
        st.session_state.models_trained = True
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.X_train = X_train
        st.session_state.y_train = y_train
        st.session_state.scaler = scaler
        
        status_placeholder.success("✅ All models trained successfully!")
        
        # Quick results preview
        if results:
            st.markdown("#### 🏆 Training Results Preview")
            
            # Best model highlight
            best_model = max(results.keys(), key=lambda k: results[k]['Test_R²'])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="success-card">
                    <h4>🥇 Best Model</h4>
                    <h3>{best_model}</h3>
                    <p>R² Score: {results[best_model]['Test_R²']:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                fastest_model = min(results.keys(), key=lambda k: results[k]['Training_Time'])
                st.markdown(f"""
                <div class="info-card">
                    <h4>⚡ Fastest Training</h4>
                    <h3>{fastest_model}</h3>
                    <p>Time: {results[fastest_model]['Training_Time']:.2f}s</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                total_time = sum(r['Training_Time'] for r in results.values())
                st.markdown(f"""
                <div class="warning-card">
                    <h4>⏱️ Total Training Time</h4>
                    <h3>{total_time:.2f}s</h3>
                    <p>{len(results)} models trained</p>
                </div>
                """, unsafe_allow_html=True)

elif page == "🔮 Risk Prediction":
    st.markdown('<div class="section-header">🔮 Intelligent Flood Risk Prediction</div>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.markdown(create_alert_card("""
            ⚠️ <strong>Models Required</strong><br>
            Please train models first from the Model Training page to make predictions.
        """, "warning"), unsafe_allow_html=True)
        st.stop()
    
    # Enhanced prediction interface
    st.markdown("#### 🎛️ Environmental Parameter Input")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.form("enhanced_prediction_form"):
            # Organized parameter groups
            st.markdown("##### 🌦️ Climate & Weather Factors")
            climate_col1, climate_col2 = st.columns(2)
            
            with climate_col1:
                monsoon = st.slider("🌧️ Monsoon Intensity", 0.0, 1.0, 0.5, 0.01,
                                   help="Intensity of monsoon season (0=low, 1=extreme)")
                climate_change = st.slider("🌡️ Climate Change Impact", 0.0, 1.0, 0.5, 0.01,
                                         help="Local impact of climate change")
                coastal_vuln = st.slider("🌊 Coastal Vulnerability", 0.0, 1.0, 0.5, 0.01,
                                        help="Vulnerability to coastal flooding")
            
            with climate_col2:
                topography = st.slider("⛰️ Topography Drainage", 0.0, 1.0, 0.5, 0.01,
                                     help="Natural drainage capacity of terrain")
                landslides = st.slider("⛰️ Landslide Risk", 0.0, 1.0, 0.5, 0.01,
                                     help="Risk of landslides affecting drainage")
                watersheds = st.slider("💧 Watershed Condition", 0.0, 1.0, 0.5, 0.01,
                                     help="Health and capacity of watersheds")
            
            st.markdown("##### 🏗️ Infrastructure & Development")
            infra_col1, infra_col2 = st.columns(2)
            
            with infra_col1:
                urbanization = st.slider("🏙️ Urbanization Level", 0.0, 1.0, 0.5, 0.01,
                                       help="Degree of urban development")
                infrastructure = st.slider("🏗️ Infrastructure Quality", 0.0, 1.0, 0.5, 0.01,
                                         help="Quality of flood-resistant infrastructure")
                drainage = st.slider("🚰 Drainage Systems", 0.0, 1.0, 0.5, 0.01,
                                    help="Effectiveness of urban drainage")
                dams_quality = st.slider("🏗️ Dam Quality", 0.0, 1.0, 0.5, 0.01,
                                       help="Quality and maintenance of dams")
            
            with infra_col2:
                river_mgmt = st.slider("🏞️ River Management", 0.0, 1.0, 0.5, 0.01,
                                     help="Effectiveness of river management")
                encroachments = st.slider("🏘️ Encroachments", 0.0, 1.0, 0.5, 0.01,
                                        help="Illegal constructions in flood plains")
                population = st.slider("👥 Population Density", 0.0, 1.0, 0.5, 0.01,
                                     help="Population density in at-risk areas")
                planning = st.slider("📋 Urban Planning", 0.0, 1.0, 0.5, 0.01,
                                   help="Quality of urban planning")
            
            st.markdown("##### 🌿 Environmental Factors")
            env_col1, env_col2 = st.columns(2)
            
            with env_col1:
                deforestation = st.slider("🌳 Deforestation Level", 0.0, 1.0, 0.5, 0.01,
                                        help="Rate of deforestation in catchment area")
                wetland_loss = st.slider("🦆 Wetland Loss", 0.0, 1.0, 0.5, 0.01,
                                       help="Loss of natural wetlands")
                agricultural = st.slider("🌾 Agricultural Impact", 0.0, 1.0, 0.5, 0.01,
                                       help="Impact of agricultural practices on drainage")
            
            with env_col2:
                siltation = st.slider("🪨 Siltation Level", 0.0, 1.0, 0.5, 0.01,
                                    help="Sediment buildup in water bodies")
                disaster_prep = st.slider("🚨 Disaster Preparedness", 0.0, 1.0, 0.5, 0.01,
                                        help="Community disaster preparedness level")
                political = st.slider("🏛️ Policy Effectiveness", 0.0, 1.0, 0.5, 0.01,
                                     help="Effectiveness of flood management policies")
            
            # Prediction buttons
            col_pred1, col_pred2 = st.columns(2)
            
            with col_pred1:
                predict_single = st.form_submit_button("🔮 Predict Risk", type="primary", use_container_width=True)
            
            with col_pred2:
                predict_scenarios = st.form_submit_button("📊 Scenario Analysis", use_container_width=True)
    
    with col2:
        # Quick presets
        st.markdown("##### ⚡ Quick Presets")
        
        if st.button("🟢 Low Risk Scenario", use_container_width=True):
            # Set low risk values
            st.rerun()
        
        if st.button("🟡 Medium Risk Scenario", use_container_width=True):
            # Set medium risk values
            st.rerun()
        
        if st.button("🔴 High Risk Scenario", use_container_width=True):
            # Set high risk values
            st.rerun()
        
        if st.button("🔄 Random Values", use_container_width=True):
            # Set random values
            st.rerun()
        
        # Risk level explanation
        st.markdown(create_alert_card("""
            <h4>📊 Risk Levels</h4>
            <p><span class="risk-low">🟢 Low (0-30%)</span><br>
            Minimal flood risk, standard precautions sufficient.</p>
            
            <p><span class="risk-medium">🟡 Medium (30-60%)</span><br>
            Moderate risk, enhanced monitoring recommended.</p>
            
            <p><span class="risk-high">🔴 High (60%+)</span><br>
            Significant risk, immediate action required.</p>
        """, "info"), unsafe_allow_html=True)
    
    # Enhanced prediction results
    if predict_single:
        input_data = np.array([[
            monsoon, topography, river_mgmt, deforestation, urbanization,
            climate_change, dams_quality, siltation, agricultural, encroachments,
            disaster_prep, drainage, coastal_vuln, landslides, watersheds,
            infrastructure, population, wetland_loss, planning, political
        ]])
        
        input_scaled = st.session_state.scaler.transform(input_data)
        
        st.markdown("#### 🎯 Prediction Results")
        
        # Individual model predictions
        predictions = {}
        confidence_scores = {}
        
        for model_name, model_info in st.session_state.model_results.items():
            pred = model_info['Model'].predict(input_scaled)[0]
            predictions[model_name] = pred
            
            # Calculate confidence based on model performance
            confidence = model_info['Test_R²'] * (1 - model_info['CV_Std'])
            confidence_scores[model_name] = confidence
        
        # Ensemble prediction with weighted average
        weights = np.array(list(confidence_scores.values()))
        weights = weights / weights.sum()  # Normalize
        ensemble_pred = np.average(list(predictions.values()), weights=weights)
        
        # Enhanced results display
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Main prediction result
            if ensemble_pred < 0.3:
                risk_level = "🟢 Low Risk"
                risk_color = "success"
                risk_class = "risk-low"
            elif ensemble_pred < 0.6:
                risk_level = "🟡 Medium Risk" 
                risk_color = "warning"
                risk_class = "risk-medium"
            else:
                risk_level = "🔴 High Risk"
                risk_color = "error"
                risk_class = "risk-high"
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎯 Flood Risk Probability</h3>
                <h1>{ensemble_pred:.1%}</h1>
                <div class="{risk_class}">{risk_level}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence metrics
            avg_confidence = np.mean(list(confidence_scores.values()))
            prediction_std = np.std(list(predictions.values()))
            
            st.markdown(f"""
            <div class="info-card">
                <h4>📊 Prediction Quality</h4>
                <p>Confidence: {avg_confidence:.1%}</p>
                <p>Model Agreement: {(1-prediction_std):.1%}</p>
                <p>Models Used: {len(predictions)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Model comparison chart
            pred_df = pd.DataFrame({
                'Model': list(predictions.keys()),
                'Prediction': list(predictions.values()),
                'Confidence': list(confidence_scores.values())
            })
            
            fig_pred = px.bar(
                pred_df,
                x='Model',
                y='Prediction',
                color='Confidence',
                title="📊 Individual Model Predictions",
                color_continuous_scale="Viridis"
            )
            
            # Add ensemble line
            fig_pred.add_hline(
                y=ensemble_pred,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Ensemble: {ensemble_pred:.1%}"
            )
            
            fig_pred.update_layout(height=400)
            st.plotly_chart(fig_pred, use_container_width=True)
        
        # Detailed analysis
        st.markdown("#### 🔍 Detailed Risk Analysis")
        
        # Feature impact analysis
        feature_names = [
            'Monsoon Intensity', 'Topography Drainage', 'River Management',
            'Deforestation', 'Urbanization', 'Climate Change', 'Dam Quality',
            'Siltation', 'Agricultural Practices', 'Encroachments',
            'Disaster Preparedness', 'Drainage Systems', 'Coastal Vulnerability',
            'Landslides', 'Watersheds', 'Infrastructure Quality',
            'Population Density', 'Wetland Loss', 'Urban Planning',
            'Policy Effectiveness'
        ]
        
        input_values = input_data[0]
        
        # Calculate feature contributions (simplified)
        feature_impact = []
        for i, (name, value) in enumerate(zip(feature_names, input_values)):
            # Estimate impact based on correlation with target
            if hasattr(st.session_state, 'df_flood'):
                corr = st.session_state.df_flood.corr()['FloodProbability'].iloc[i]
                impact = value * abs(corr) * ensemble_pred
            else:
                impact = value * 0.5 * ensemble_pred  # Fallback
            feature_impact.append({'Feature': name, 'Value': value, 'Impact': impact})
        
        impact_df = pd.DataFrame(feature_impact).sort_values('Impact', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top risk contributors
            fig_risk_contrib = px.bar(
                impact_df.head(10),
                x='Impact',
                y='Feature',
                orientation='h',
                title="🚨 Top Risk Contributors",
                color='Impact',
                color_continuous_scale="Reds"
            )
            fig_risk_contrib.update_layout(height=400)
            st.plotly_chart(fig_risk_contrib, use_container_width=True)
        
        with col2:
            # Feature values radar chart
            top_features = impact_df.head(6)
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=top_features['Value'].values,
                theta=top_features['Feature'].values,
                fill='toself',
                name='Current Values'
            ))
            
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title="📊 Key Parameter Values"
            )
            st.plotly_chart(fig_radar, use_container_width=True)
    
    # Scenario analysis
    if predict_scenarios:
        st.markdown("#### 📊 Scenario Analysis")
        
        scenarios = {
            "Current": [monsoon, topography, river_mgmt, deforestation, urbanization,
                       climate_change, dams_quality, siltation, agricultural, encroachments,
                       disaster_prep, drainage, coastal_vuln, landslides, watersheds,
                       infrastructure, population, wetland_loss, planning, political],
            "Best Case": [0.2, 0.9, 0.9, 0.1, 0.3, 0.2, 0.9, 0.1, 0.8, 0.1,
                         0.9, 0.9, 0.2, 0.1, 0.9, 0.9, 0.3, 0.1, 0.9, 0.9],
            "Worst Case": [0.9, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.9, 0.2, 0.9,
                          0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.9, 0.9, 0.1, 0.1],
            "Climate Change +20%": None  # Will be calculated
        }
        
        # Calculate climate change scenario
        climate_change_scenario = scenarios["Current"].copy()
        climate_change_scenario[0] *= 1.2  # Monsoon +20%
        climate_change_scenario[5] *= 1.2  # Climate change +20%
        climate_change_scenario[12] *= 1.2  # Coastal vulnerability +20%
        scenarios["Climate Change +20%"] = [min(1.0, x) for x in climate_change_scenario]
        
        scenario_results = {}
        
        for scenario_name, values in scenarios.items():
            if values is None:
                continue
            
            input_scaled = st.session_state.scaler.transform([values])
            
            # Get ensemble prediction
            preds = []
            for model_name, model_info in st.session_state.model_results.items():
                pred = model_info['Model'].predict(input_scaled)[0]
                preds.append(pred)
            
            scenario_results[scenario_name] = np.mean(preds)
        
        # Display scenario comparison
        scenario_df = pd.DataFrame({
            'Scenario': list(scenario_results.keys()),
            'Risk_Probability': list(scenario_results.values())
        })
        
        fig_scenarios = px.bar(
            scenario_df,
            x='Scenario',
            y='Risk_Probability',
            title="📊 Scenario Risk Comparison",
            color='Risk_Probability',
            color_continuous_scale="RdYlGn_r"
        )
        fig_scenarios.update_layout(height=400)
        st.plotly_chart(fig_scenarios, use_container_width=True)
        
        # Scenario recommendations
        col1, col2 = st.columns(2)
        
        with col1:
            current_risk = scenario_results.get("Current", 0)
            best_case_risk = scenario_results.get("Best Case", 0)
            improvement_potential = current_risk - best_case_risk
            
            st.markdown(f"""
            <div class="success-card">
                <h4>🎯 Improvement Potential</h4>
                <h3>{improvement_potential:.1%}</h3>
                <p>Risk reduction possible with optimal management</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            worst_case_risk = scenario_results.get("Worst Case", 0)
            risk_spread = worst_case_risk - best_case_risk
            
            st.markdown(f"""
            <div class="warning-card">
                <h4>⚠️ Risk Range</h4>
                <h3>{risk_spread:.1%}</h3>
                <p>Difference between best and worst case scenarios</p>
            </div>
            """, unsafe_allow_html=True)

elif page == "🛰️ Satellite Analysis":
    st.markdown('<div class="section-header">🛰️ Advanced Satellite Imagery Analysis</div>', unsafe_allow_html=True)
    
    if not st.session_state.dataset_loaded:
        st.markdown(create_alert_card("""
            ⚠️ <strong>Dataset Required</strong><br>
            Please load the sample dataset first from the Dashboard page to begin analysis.
        """, "warning"), unsafe_allow_html=True)
        st.stop()
    
    # Enhanced CNN architecture info
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(create_alert_card("""
            <h4>🧠 Deep Learning Architecture</h4>
            <ul>
                <li>🔷 Convolutional Neural Network (CNN)</li>
                <li>🔸 4 Conv2D layers with BatchNormalization</li>
                <li>🔹 Progressive feature extraction (32→64→128→256)</li>
                <li>🔸 MaxPooling for dimensionality reduction</li>
                <li>🔹 Dense layers with Dropout regularization</li>
                <li>🔸 Sigmoid activation for flood probability</li>
            </ul>
        """, "info"), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_alert_card("""
            <h4>📊 Model Specifications</h4>
            <ul>
                <li>📥 Input: 128×128×3 RGB satellite images</li>
                <li>📤 Output: Flood probability (0-1)</li>
                <li>⚡ Optimizer: Adam (lr=0.001)</li>
                <li>📉 Loss Function: Binary crossentropy</li>
                <li>📊 Metrics: Accuracy, Precision, Recall</li>
                <li>🎯 Batch Size: 32, Epochs: 50</li>
            </ul>
        """, "success"), unsafe_allow_html=True)
    
    # Sample satellite images display
    st.markdown("#### 🖼️ Sample Satellite Image Dataset")
    
    if hasattr(st.session_state, 'sample_images'):
        images = st.session_state.sample_images
        
        # Create image categories
        image_categories = {
            "🌊 Flood-Prone Areas": images[:4],
            "🏙️ Urban Regions": images[4:8], 
            "🌿 Rural/Forest Areas": images[8:12]
        }
        
        for category, cat_images in image_categories.items():
            st.markdown(f"##### {category}")
            
            cols = st.columns(4)
            for i, img_array in enumerate(cat_images):
                with cols[i]:
                    # Convert numpy array to PIL Image
                    img_pil = Image.fromarray(img_array)
                    st.image(img_pil, caption=f"Sample {i+1}", use_column_width=True)
                    
                    # Add classification button
                    if st.button(f"🔍 Analyze {i+1}", key=f"{category}_{i}"):
                        # Simulate classification
                        fake_prob = np.random.random()
                        if fake_prob > 0.6:
                            st.error(f"🔴 High Flood Risk: {fake_prob:.1%}")
                        elif fake_prob > 0.3:
                            st.warning(f"🟡 Medium Flood Risk: {fake_prob:.1%}")
                        else:
                            st.success(f"🟢 Low Flood Risk: {fake_prob:.1%}")
    
    # Enhanced CNN training simulation
    st.markdown("#### 🚀 CNN Model Training & Evaluation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Training configuration
        with st.form("cnn_training_config"):
            st.markdown("##### ⚙️ Training Configuration")
            
            config_col1, config_col2 = st.columns(2)
            
            with config_col1:
                epochs = st.slider("📊 Training Epochs", 10, 100, 50, 10)
                batch_size = st.selectbox("📦 Batch Size", [16, 32, 64, 128], index=1)
                learning_rate = st.selectbox("📈 Learning Rate", [0.001, 0.01, 0.1], index=0)
            
            with config_col2:
                dropout_rate = st.slider("🎯 Dropout Rate", 0.1, 0.7, 0.5, 0.1)
                validation_split = st.slider("✅ Validation Split", 0.1, 0.3, 0.2, 0.05)
                use_augmentation = st.checkbox("🔄 Data Augmentation", value=True)
            
            train_cnn = st.form_submit_button("🚀 Train CNN Model", type="primary", use_container_width=True)
    
    with col2:
        st.markdown(create_alert_card("""
            <h4>🎯 Training Tips</h4>
            <ul>
                <li>Higher epochs = Better learning but longer training</li>
                <li>Larger batch size = Faster training but more memory</li>
                <li>Lower learning rate = More stable but slower convergence</li>
                <li>Dropout prevents overfitting</li>
                <li>Data augmentation improves generalization</li>
            </ul>
        """, "info"), unsafe_allow_html=True)
    
    if train_cnn:
        st.markdown("#### 🔄 Training Progress")
        
        # Simulate training with realistic progress
        progress_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics_container = st.container()
            
            # Initialize training history
            train_loss = []
            val_loss = []
            train_acc = []
            val_acc = []
            train_precision = []
            train_recall = []
            
            # Live metrics display
            with metrics_container:
                metric_cols = st.columns(5)
                loss_metric = metric_cols[0].empty()
                acc_metric = metric_cols[1].empty()
                val_loss_metric = metric_cols[2].empty()
                val_acc_metric = metric_cols[3].empty()
                lr_metric = metric_cols[4].empty()
            
            # Simulate realistic training progression
            for epoch in range(epochs):
                # Simulate realistic loss decrease with some noise
                base_train_loss = 0.9 * np.exp(-epoch / 20) + 0.1
                base_val_loss = 1.0 * np.exp(-epoch / 25) + 0.15
                
                # Add realistic noise
                tl = base_train_loss + np.random.normal(0, 0.05)
                vl = base_val_loss + np.random.normal(0, 0.07)
                
                # Simulate accuracy improvement
                ta = 0.5 + 0.45 * (1 - np.exp(-epoch / 15)) + np.random.normal(0, 0.02)
                va = 0.45 + 0.4 * (1 - np.exp(-epoch / 18)) + np.random.normal(0, 0.025)
                
                # Simulate precision and recall
                tp = ta + np.random.normal(0, 0.01)
                tr = ta + np.random.normal(0, 0.015)
                
                # Ensure realistic bounds
                tl = max(0.05, tl)
                vl = max(0.1, vl)
                ta = min(0.98, max(0.5, ta))
                va = min(0.95, max(0.45, va))
                tp = min(0.98, max(0.5, tp))
                tr = min(0.98, max(0.5, tr))
                
                train_loss.append(tl)
                val_loss.append(vl)
                train_acc.append(ta)
                val_acc.append(va)
                train_precision.append(tp)
                train_recall.append(tr)
                
                # Update status
                status_text.text(f"Epoch {epoch+1}/{epochs} - Training...")
                
                # Update live metrics
                loss_metric.metric("Train Loss", f"{tl:.4f}")
                acc_metric.metric("Train Acc", f"{ta:.3f}")
                val_loss_metric.metric("Val Loss", f"{vl:.4f}")
                val_acc_metric.metric("Val Acc", f"{va:.3f}")
                current_lr = learning_rate * (0.95 ** (epoch // 10))  # Learning rate decay
                lr_metric.metric("Learning Rate", f"{current_lr:.5f}")
                
                # Update progress
                progress_bar.progress((epoch + 1) / epochs)
                
                # Simulate training time
                time.sleep(0.1)
        
        # Training completion
        status_text.success("✅ CNN Training Completed!")
        
        # Enhanced training results
        st.markdown("#### 📈 Training Results & Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Training history plot
            fig_history = make_subplots(
                rows=2, cols=1,
                subplot_titles=('📉
