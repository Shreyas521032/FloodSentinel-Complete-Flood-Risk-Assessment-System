import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import time
import base64
from io import BytesIO
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
    if 'df_flood' not in st.session_state:
        st.session_state.df_flood = None
    if 'sample_images' not in st.session_state:
        st.session_state.sample_images = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    if 'y_train' not in st.session_state:
        st.session_state.y_train = None
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None

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

@st.cache_resource
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

@st.cache_data
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
        images.append(Image.fromarray(img))
    
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
                st.session_state.models_trained = False # Reset trained models on new data load
                st.session_state.model_results = {}
                st.session_state.X_test = None
                st.session_state.y_test = None
                st.session_state.X_train = None
                st.session_state.y_train = None
                st.session_state.scaler = None
                
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
    
    if not st.session_state.dataset_loaded or st.session_state.df_flood is None:
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
    correlations = df.corr(numeric_only=True)['FloodProbability'].abs().sort_values(ascending=False)[1:11]
    top_features = correlations.index.tolist() + ['FloodProbability']
    corr_matrix = df[top_features].corr(numeric_only=True)
    
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
            labels=['🟢 Low', '🟡 Medium', '🔴 High'],
            right=False # Include 0.0 in Low category
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
    
    correlations = df.corr(numeric_only=True)['FloodProbability'].abs().sort_values(ascending=False)[1:]
    
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
    
    if not st.session_state.dataset_loaded or st.session_state.df_flood is None:
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
            # Hyperparameter tuning is a complex feature, might be better to simplify or offer as advanced option
            # For now, keeping it as a checkbox but not implementing full GridSearchCV for all models
            enable_hyperparameter_tuning = st.checkbox(
                "🔧 Hyperparameter Tuning (Advanced)",
                help="Enable automated hyperparameter optimization (slower but potentially better results). Not fully implemented for all models."
            )
            
            parallel_training = st.checkbox(
                "⚡ Parallel Training",
                value=True,
                help="Use parallel processing for faster training (Note: Streamlit's nature might limit true parallelism for some operations)"
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
        
        # Stratify only if y is not continuous (e.g., binned for classification)
        # For regression, stratify is not typically used directly on continuous target.
        # Removing stratify for continuous target variable.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
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
                # Hyperparameter tuning (simplified example for one model type)
                if enable_hyperparameter_tuning and model_name == "🌳 Random Forest":
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [5, 10, 15, None]
                    }
                    grid_search = GridSearchCV(model, param_grid, cv=cv_folds, scoring='r2', n_jobs=-1)
                    grid_search.fit(X_train_scaled, y_train)
                    model = grid_search.best_estimator_
                    st.info(f"Best params for Random Forest: {grid_search.best_params_}")

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
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds, scoring='r2', n_jobs=-1 if parallel_training else 1)
                
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
            # Ensure there's at least one successful model training result
            if results:
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
            else:
                st.warning("No models were successfully trained to display results.")

elif page == "🔮 Risk Prediction":
    st.markdown('<div class="section-header">🔮 Intelligent Flood Risk Prediction</div>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained or not st.session_state.model_results:
        st.markdown(create_alert_card("""
            ⚠️ <strong>Models Required</strong><br>
            Please train models first from the Model Training page to make predictions.
        """, "warning"), unsafe_allow_html=True)
        st.stop()
    
    # Enhanced prediction interface
    st.markdown("#### 🎛️ Environmental Parameter Input")
    
    # Define feature names in the correct order as expected by the model
    prediction_feature_names = [
        'monsoon_intensity', 'topography_drainage', 'river_management',
        'deforestation', 'urbanization', 'climate_change', 'dams_quality',
        'siltation', 'agricultural_practices', 'encroachments',
        'disaster_preparedness', 'drainage_systems', 'coastal_vulnerability',
        'landslides', 'watersheds', 'infrastructure_quality',
        'population_density', 'wetland_loss', 'planning_adequacy',
        'political_factors'
    ]

    input_values_dict = {}

    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.form("enhanced_prediction_form"):
            # Organized parameter groups
            st.markdown("##### 🌦️ Climate & Weather Factors")
            climate_col1, climate_col2 = st.columns(2)
            
            with climate_col1:
                input_values_dict['monsoon_intensity'] = st.slider("🌧️ Monsoon Intensity", 0.0, 1.0, 0.5, 0.01,
                                   help="Intensity of monsoon season (0=low, 1=extreme)")
                input_values_dict['climate_change'] = st.slider("🌡️ Climate Change Impact", 0.0, 1.0, 0.5, 0.01,
                                         help="Local impact of climate change")
                input_values_dict['coastal_vulnerability'] = st.slider("🌊 Coastal Vulnerability", 0.0, 1.0, 0.5, 0.01,
                                        help="Vulnerability to coastal flooding")
            
            with climate_col2:
                input_values_dict['topography_drainage'] = st.slider("⛰️ Topography Drainage", 0.0, 1.0, 0.5, 0.01,
                                     help="Natural drainage capacity of terrain")
                input_values_dict['landslides'] = st.slider("⛰️ Landslide Risk", 0.0, 1.0, 0.5, 0.01,
                                     help="Risk of landslides affecting drainage")
                input_values_dict['watersheds'] = st.slider("💧 Watershed Condition", 0.0, 1.0, 0.5, 0.01,
                                     help="Health and capacity of watersheds")
            
            st.markdown("##### 🏗️ Infrastructure & Development")
            infra_col1, infra_col2 = st.columns(2)
            
            with infra_col1:
                input_values_dict['urbanization'] = st.slider("🏙️ Urbanization Level", 0.0, 1.0, 0.5, 0.01,
                                       help="Degree of urban development")
                input_values_dict['infrastructure_quality'] = st.slider("🏗️ Infrastructure Quality", 0.0, 1.0, 0.5, 0.01,
                                         help="Quality of flood-resistant infrastructure")
                input_values_dict['drainage_systems'] = st.slider("🚰 Drainage Systems", 0.0, 1.0, 0.5, 0.01,
                                    help="Effectiveness of urban drainage")
                input_values_dict['dams_quality'] = st.slider("🏗️ Dam Quality", 0.0, 1.0, 0.5, 0.01,
                                       help="Quality and maintenance of dams")
            
            with infra_col2:
                input_values_dict['river_management'] = st.slider("🏞️ River Management", 0.0, 1.0, 0.5, 0.01,
                                     help="Effectiveness of river management")
                input_values_dict['encroachments'] = st.slider("🏘️ Encroachments", 0.0, 1.0, 0.5, 0.01,
                                        help="Illegal constructions in flood plains")
                input_values_dict['population_density'] = st.slider("👥 Population Density", 0.0, 1.0, 0.5, 0.01,
                                     help="Population density in at-risk areas")
                input_values_dict['planning_adequacy'] = st.slider("📋 Urban Planning", 0.0, 1.0, 0.5, 0.01,
                                   help="Quality of urban planning")
            
            st.markdown("##### 🌿 Environmental Factors")
            env_col1, env_col2 = st.columns(2)
            
            with env_col1:
                input_values_dict['deforestation'] = st.slider("🌳 Deforestation Level", 0.0, 1.0, 0.5, 0.01,
                                        help="Rate of deforestation in catchment area")
                input_values_dict['wetland_loss'] = st.slider("🦆 Wetland Loss", 0.0, 1.0, 0.5, 0.01,
                                       help="Loss of natural wetlands")
                input_values_dict['agricultural_practices'] = st.slider("🌾 Agricultural Impact", 0.0, 1.0, 0.5, 0.01,
                                       help="Impact of agricultural practices on drainage")
            
            with env_col2:
                input_values_dict['siltation'] = st.slider("🪨 Siltation Level", 0.0, 1.0, 0.5, 0.01,
                                    help="Sediment buildup in water bodies")
                input_values_dict['disaster_preparedness'] = st.slider("🚨 Disaster Preparedness", 0.0, 1.0, 0.5, 0.01,
                                        help="Community disaster preparedness level")
                input_values_dict['political_factors'] = st.slider("🏛️ Policy Effectiveness", 0.0, 1.0, 0.5, 0.01,
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
            st.session_state.input_preset = "low"
            st.rerun()
        
        if st.button("🟡 Medium Risk Scenario", use_container_width=True):
            st.session_state.input_preset = "medium"
            st.rerun()
        
        if st.button("🔴 High Risk Scenario", use_container_width=True):
            st.session_state.input_preset = "high"
            st.rerun()
        
        if st.button("🔄 Random Values", use_container_width=True):
            st.session_state.input_preset = "random"
            st.rerun()
        
        # Apply preset values if set
        if 'input_preset' in st.session_state and st.session_state.input_preset:
            if st.session_state.input_preset == "low":
                for key in input_values_dict: input_values_dict[key] = 0.2
            elif st.session_state.input_preset == "medium":
                for key in input_values_dict: input_values_dict[key] = 0.5
            elif st.session_state.input_preset == "high":
                for key in input_values_dict: input_values_dict[key] = 0.8
            elif st.session_state.input_preset == "random":
                for key in input_values_dict: input_values_dict[key] = np.random.uniform(0.0, 1.0)
            st.session_state.input_preset = None # Clear preset after applying
            st.rerun() # Rerun to update slider values

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
        # Ensure input_data is in the correct order for the scaler and models
        input_data_ordered = [input_values_dict[feature] for feature in prediction_feature_names]
        input_data = np.array([input_data_ordered])
        
        input_scaled = st.session_state.scaler.transform(input_data)
        
        st.markdown("#### 🎯 Prediction Results")
        
        # Individual model predictions
        predictions = {}
        confidence_scores = {}
        
        for model_name, model_info in st.session_state.model_results.items():
            try:
                pred = model_info['Model'].predict(input_scaled)[0]
                predictions[model_name] = pred
                
                # Calculate confidence based on model performance
                # Ensure CV_Std is not NaN or inf, and R^2 is not negative
                cv_std = model_info['CV_Std'] if not np.isnan(model_info['CV_Std']) else 0
                test_r2 = model_info['Test_R²'] if model_info['Test_R²'] > 0 else 0
                confidence = test_r2 * (1 - cv_std)
                confidence_scores[model_name] = confidence
            except Exception as e:
                st.warning(f"Could not get prediction for {model_name}: {e}")
                continue
        
        if not predictions:
            st.error("No models were able to make a prediction. Please check model training results.")
            st.stop()

        # Ensemble prediction with weighted average
        # Filter out models that failed to predict
        valid_predictions = {k: v for k, v in predictions.items() if k in confidence_scores}
        valid_confidence_scores = {k: v for k, v in confidence_scores.items() if k in valid_predictions}

        if not valid_predictions:
            st.error("No valid predictions could be generated. Please ensure models are trained correctly.")
            st.stop()

        weights = np.array(list(valid_confidence_scores.values()))
        if weights.sum() == 0:
            # If all weights are zero (e.g., all R^2 were 0 or negative), use equal weights
            weights = np.ones(len(valid_predictions))
        weights = weights / weights.sum()  # Normalize
        ensemble_pred = np.average(list(valid_predictions.values()), weights=weights)
        
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
            avg_confidence = np.mean(list(valid_confidence_scores.values()))
            prediction_std = np.std(list(valid_predictions.values()))
            
            st.markdown(f"""
            <div class="info-card">
                <h4>📊 Prediction Quality</h4>
                <p>Confidence: {avg_confidence:.1%}</p>
                <p>Model Agreement: {(1-prediction_std):.1%}</p>
                <p>Models Used: {len(valid_predictions)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Model comparison chart
            pred_df = pd.DataFrame({
                'Model': list(valid_predictions.keys()),
                'Prediction': list(valid_predictions.values()),
                'Confidence': list(valid_confidence_scores.values())
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
        # Use the actual feature names from the DataFrame columns for consistency
        feature_names_df = st.session_state.df_flood.drop('FloodProbability', axis=1).columns.tolist()
        
        input_values_list = [input_values_dict[f] for f in feature_names_df] # Ensure order matches df
        
        # Calculate feature contributions (simplified)
        feature_impact = []
        if hasattr(st.session_state, 'df_flood') and st.session_state.df_flood is not None:
            # Calculate correlations only once and store if needed
            full_correlations = st.session_state.df_flood.corr(numeric_only=True)['FloodProbability']
        else:
            full_correlations = pd.Series(np.zeros(len(feature_names_df)), index=feature_names_df)

        for i, (name, value) in enumerate(zip(feature_names_df, input_values_list)):
            corr = full_correlations.get(name, 0) # Use .get to handle cases where feature might not be in correlation matrix
            impact = value * abs(corr) * ensemble_pred
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
            top_features_radar = impact_df.head(6) # Use a subset for readability
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=top_features_radar['Value'].values,
                theta=top_features_radar['Feature'].values,
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
        
        # Define scenarios using the prediction_feature_names for consistency
        scenarios = {
            "Current": [input_values_dict[f] for f in prediction_feature_names],
            "Best Case": [0.2, 0.9, 0.9, 0.1, 0.3, 0.2, 0.9, 0.1, 0.8, 0.1,
                         0.9, 0.9, 0.2, 0.1, 0.9, 0.9, 0.3, 0.1, 0.9, 0.9],
            "Worst Case": [0.9, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.9, 0.2, 0.9,
                          0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.9, 0.9, 0.1, 0.1],
            "Climate Change +20%": None  # Will be calculated
        }
        
        # Calculate climate change scenario based on current inputs
        climate_change_scenario_values = scenarios["Current"].copy()
        # Assuming indices for 'monsoon_intensity', 'climate_change', 'coastal_vulnerability'
        # These indices need to be carefully mapped to prediction_feature_names
        # For robustness, it's better to use feature names directly if possible, or ensure consistent ordering.
        # For now, relying on the original order from the provided code.
        try:
            idx_monsoon = prediction_feature_names.index('monsoon_intensity')
            idx_climate_change = prediction_feature_names.index('climate_change')
            idx_coastal_vuln = prediction_feature_names.index('coastal_vulnerability')

            climate_change_scenario_values[idx_monsoon] = min(1.0, climate_change_scenario_values[idx_monsoon] * 1.2)
            climate_change_scenario_values[idx_climate_change] = min(1.0, climate_change_scenario_values[idx_climate_change] * 1.2)
            climate_change_scenario_values[idx_coastal_vuln] = min(1.0, climate_change_scenario_values[idx_coastal_vuln] * 1.2)
            scenarios["Climate Change +20%"] = climate_change_scenario_values
        except ValueError as e:
            st.error(f"Error in scenario calculation: {e}. Feature not found in list.")
            scenarios.pop("Climate Change +20%", None) # Remove problematic scenario

        scenario_results = {}
        
        for scenario_name, values in scenarios.items():
            if values is None:
                continue
            
            try:
                input_scaled = st.session_state.scaler.transform([values])
                
                # Get ensemble prediction
                preds = []
                for model_name, model_info in st.session_state.model_results.items():
                    if 'Model' in model_info:
                        pred = model_info['Model'].predict(input_scaled)[0]
                        preds.append(pred)
                
                if preds:
                    scenario_results[scenario_name] = np.mean(preds)
                else:
                    st.warning(f"No valid model predictions for scenario: {scenario_name}")
            except Exception as e:
                st.error(f"Error predicting for scenario {scenario_name}: {e}")
                continue
        
        if not scenario_results:
            st.error("No scenario results could be generated.")
            st.stop()

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
                <h4>⚠️ Risk Spread</h4>
                <h3>{risk_spread:.1%}</h3>
                <p>Difference between worst and best case scenarios</p>
            </div>
            """, unsafe_allow_html=True)

elif page == "🛰️ Satellite Analysis":
    st.markdown('<div class="section-header">🛰️ Satellite Imagery Analysis</div>', unsafe_allow_html=True)
    
    if not st.session_state.dataset_loaded or st.session_state.sample_images is None:
        st.markdown(create_alert_card("""
            ⚠️ <strong>Sample Images Required</strong><br>
            Please load the sample dataset from the Dashboard page to view satellite images.
        """, "warning"), unsafe_allow_html=True)
        st.stop()
    
    st.markdown("#### 🖼️ Sample Satellite Images")
    st.write("These are simulated satellite images representing different environmental conditions.")
    
    # Display images in a gallery format
    num_images = len(st.session_state.sample_images)
    cols = st.columns(4) # Display 4 images per row
    
    for i, img in enumerate(st.session_state.sample_images):
        with cols[i % 4]:
            st.image(img, caption=f"Image {i+1}", use_column_width=True)

    st.markdown(create_alert_card("""
        <h4>💡 Image Interpretation</h4>
        <p>Images with more blue tones might indicate flooded areas or high water content. 
        Grayish tones could represent urban development, while greener tones suggest vegetation or rural areas.</p>
        <p><strong>Note:</strong> This is a simulated demonstration. Real satellite analysis involves complex image processing and machine learning techniques to detect flood extent, water levels, and land cover changes.</p>
    """, "info"), unsafe_allow_html=True)

elif page == "📈 Results & Export":
    st.markdown('<div class="section-header">📈 Model Performance & Export</div>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained or not st.session_state.model_results:
        st.markdown(create_alert_card("""
            ⚠️ <strong>Models Required</strong><br>
            Please train models first from the Model Training page to view results.
        """, "warning"), unsafe_allow_html=True)
        st.stop()
    
    results_df = pd.DataFrame(st.session_state.model_results).T
    results_df = results_df.drop(columns=['Model', 'Test_Predictions', 'Train_Predictions'], errors='ignore')
    
    st.markdown("#### 📊 Model Performance Summary")
    st.dataframe(results_df.sort_values(by='Test_R²', ascending=False), use_container_width=True)
    
    # Enhanced visualizations of model performance
    st.markdown("#### 📈 Performance Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_r2 = px.bar(
            results_df.reset_index(),
            x='index',
            y='Test_R²',
            title='R² Score Comparison',
            labels={'index': 'Model', 'Test_R²': 'R² Score'},
            color='Test_R²',
            color_continuous_scale=px.colors.sequential.Plasma
        )
        st.plotly_chart(fig_r2, use_container_width=True)
    
    with col2:
        fig_rmse = px.bar(
            results_df.reset_index(),
            x='index',
            y='Test_RMSE',
            title='RMSE Comparison',
            labels={'index': 'Model', 'Test_RMSE': 'RMSE'},
            color='Test_RMSE',
            color_continuous_scale=px.colors.sequential.Viridis_r # Reversed for lower RMSE = better
        )
        st.plotly_chart(fig_rmse, use_container_width=True)

    # Overfitting analysis
    st.markdown("#### 📉 Overfitting Analysis")
    fig_overfit = px.scatter(
        results_df.reset_index(),
        x='Train_R²',
        y='Test_R²',
        size='Overfitting_Score',
        color='Overfitting_Score',
        hover_name='index',
        title='Train vs Test R² (Overfitting)',
        labels={'index': 'Model', 'Train_R²': 'Train R²', 'Test_R²': 'Test R²', 'Overfitting_Score': 'Overfitting Score'},
        color_continuous_scale=px.colors.sequential.Inferno
    )
    fig_overfit.add_shape(
        type="line",
        x0=0, y0=0, x1=1, y1=1,
        line=dict(color="Red", dash="dash"),
        name="Ideal Fit"
    )
    st.plotly_chart(fig_overfit, use_container_width=True)

    # Export options
    st.markdown("#### 💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_export = results_df.to_csv(index=True).encode('utf-8')
        st.download_button(
            label="Download Results as CSV",
            data=csv_export,
            file_name="model_performance_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        excel_export = BytesIO()
        with pd.ExcelWriter(excel_export, engine='xlsxwriter') as writer:
            results_df.to_excel(writer, sheet_name='Model Results', index=True)
        excel_export.seek(0)
        st.download_button(
            label="Download Results as Excel",
            data=excel_export,
            file_name="model_performance_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    with col3:
        # Export trained models (simplified - usually involves pickling/joblib)
        # For demonstration, we'll just create a dummy file or indicate success
        st.info("Model export functionality (e.g., via Joblib) can be added here.")
        # Example of how you might offer a dummy download or message
        st.download_button(
            label="Download Trained Models (Dummy)",
            data="Dummy model export content.".encode('utf-8'),
            file_name="trained_models_dummy.txt",
            mime="text/plain",
            use_container_width=True
        )

# Footer
st.markdown("""
<div class="footer">
    <p>FloodSentinel Pro © 2025. All rights reserved.</p>
    <p>Powered by AI and Streamlit.</p>
</div>
""", unsafe_allow_html=True)


