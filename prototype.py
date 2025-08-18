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

# --- Configuration --- #
APP_TITLE = "FloodSentinel Pro - AI Flood Risk Assessment"
APP_ICON = "🌊"

# --- Helper Functions for UI Components --- #
def create_metric_card(title, value, icon="📊"):
    """Creates an enhanced metric card for display."""
    return f"""
    <div class="metric-card">
        <h3>{icon} {title}</h3>
        <h2>{value}</h2>
    </div>
    """

def create_alert_card(message, alert_type="info"):
    """Creates enhanced alert cards for various messages."""
    return f'<div class="{alert_type}-card">{message}</div>'

def get_image_as_base64(image_path):
    """Converts an image to base64 for embedding in HTML/CSS."""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return ""

# --- Data Generation and Model Definitions --- #
@st.cache_data
def generate_sample_flood_data(n_samples=1000):
    """Generates realistic sample flood data for demonstration."""
    np.random.seed(42)
    features = [
        'monsoon_intensity', 'topography_drainage', 'river_management',
        'deforestation', 'urbanization', 'climate_change', 'dams_quality',
        'siltation', 'agricultural_practices', 'encroachments',
        'disaster_preparedness', 'drainage_systems', 'coastal_vulnerability',
        'landslides', 'watersheds', 'infrastructure_quality',
        'population_density', 'wetland_loss', 'planning_adequacy',
        'political_factors'
    ]
    data = {}
    base_risk = np.random.beta(2, 3, n_samples)
    for i, feature in enumerate(features):
        correlation_strength = np.random.uniform(0.3, 0.8)
        noise = np.random.normal(0, 0.2, n_samples)
        feature_values = (base_risk * correlation_strength + 
                         np.random.uniform(0, 1, n_samples) * (1 - correlation_strength) + 
                         noise)
        feature_values = np.clip(feature_values, 0, 1)
        data[feature] = feature_values
    weights = np.random.uniform(0.5, 2.0, len(features))
    flood_prob = np.zeros(n_samples)
    for i, feature in enumerate(features):
        flood_prob += data[feature] * weights[i]
    flood_prob = flood_prob / flood_prob.max()
    flood_prob = np.clip(flood_prob ** 1.5, 0, 1)
    data['FloodProbability'] = flood_prob
    df = pd.DataFrame(data)
    return df

@st.cache_resource
def get_model_algorithms():
    """Returns a dictionary of machine learning models with optimized parameters."""
    return {
        "🌳 Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=10, min_samples_split=5, random_state=42
        ),
        "⚡ Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
        ),
        "🧠 Neural Network": MLPRegressor(
            hidden_layer_sizes=(100, 50, 25), max_iter=1000, learning_rate='adaptive', random_state=42
        ),
        "📈 Support Vector": SVR(kernel='rbf', C=1.0, gamma='scale'),
        "🔗 ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        "🎪 AdaBoost": AdaBoostRegressor(n_estimators=100, learning_rate=1.0, random_state=42),
        "🌿 Decision Tree": DecisionTreeRegressor(max_depth=10, min_samples_split=5, random_state=42),
        "👥 K-Neighbors": KNeighborsRegressor(n_neighbors=5, weights='distance'),
        "📊 Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "🎯 Linear Regression": LinearRegression()
    }

@st.cache_data
def generate_sample_satellite_images(num_images=12):
    """Generates sample satellite-like images for demonstration."""
    images = []
    for i in range(num_images):
        np.random.seed(i + 42)
        img = np.random.rand(128, 128, 3)
        if i < 4:
            img[:, :, 2] += 0.3
            img[:, :, 0] *= 0.7
        elif i < 8:
            img = (img * 0.6) + 0.2
        else:
            img[:, :, 1] += 0.2
            img[:, :, 0] *= 0.8
        noise = np.random.normal(0, 0.1, (128, 128, 3))
        img = np.clip(img + noise, 0, 1)
        img = (img * 255).astype(np.uint8)
        images.append(Image.fromarray(img))
    return images

# --- Session State Management --- #
def init_session_state():
    """Initializes Streamlit session state variables."""
    defaults = {
        'models_trained': False,
        'dataset_loaded': False,
        'model_results': {},
        'sample_data': None,
        'df_flood': None,
        'sample_images': None,
        'X_test': None,
        'y_test': None,
        'X_train': None,
        'y_train': None,
        'scaler': None,
        'trained_models': {}
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- Main Application Layout and Pages --- #
def apply_custom_css():
    """Applies custom CSS for enhanced UI design."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        .main { font-family: 'Inter', sans-serif; }
        .main-header { font-size: 3.5rem; font-weight: 700; text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 1rem; text-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
        .sub-header { text-align: center; font-size: 1.3rem; color: #6b7280; margin-bottom: 3rem; font-weight: 400; }
        .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 16px; color: white; text-align: center; margin: 0.5rem 0; box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3); transition: transform 0.3s ease, box-shadow 0.3s ease; border: 1px solid rgba(255, 255, 255, 0.1); }
        .metric-card:hover { transform: translateY(-5px); box-shadow: 0 20px 40px rgba(102, 126, 234, 0.4); }
        .metric-card h3 { margin: 0; font-size: 0.9rem; font-weight: 500; opacity: 0.9; }
        .metric-card h2 { margin: 0.5rem 0 0 0; font-size: 2.2rem; font-weight: 700; }
        .success-card { background: linear-gradient(135deg, #10b981 0%, #059669 100%); padding: 1.5rem; border-radius: 16px; color: white; margin: 1rem 0; box-shadow: 0 8px 25px rgba(16, 185, 129, 0.3); border-left: 4px solid #34d399; }
        .warning-card { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); padding: 1.5rem; border-radius: 16px; color: white; margin: 1rem 0; box-shadow: 0 8px 25px rgba(245, 158, 11, 0.3); border-left: 4px solid #fbbf24; }
        .info-card { background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); padding: 1.5rem; border-radius: 16px; color: white; margin: 1rem 0; box-shadow: 0 8px 25px rgba(59, 130, 246, 0.3); border-left: 4px solid #60a5fa; }
        .error-card { background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); padding: 1.5rem; border-radius: 16px; color: white; margin: 1rem 0; box-shadow: 0 8px 25px rgba(239, 68, 68, 0.3); border-left: 4px solid #f87171; }
        .sidebar .sidebar-content { background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%); }
        .stButton > button { border-radius: 25px; border: none; padding: 0.5rem 2rem; font-weight: 600; transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); }
        .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15); }
        .stForm { background: #f8fafc; padding: 2rem; border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); }
        .risk-low { background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 0.75rem 1.5rem; border-radius: 25px; font-weight: 600; text-align: center; box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3); }
        .risk-medium { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); color: white; padding: 0.75rem 1.5rem; border-radius: 25px; font-weight: 600; text-align: center; box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3); }
        .risk-high { background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); color: white; padding: 0.75rem 1.5rem; border-radius: 25px; font-weight: 600; text-align: center; box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3); }
        .image-gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 1rem; margin: 1rem 0; }
        .image-item { border-radius: 12px; overflow: hidden; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); transition: transform 0.3s ease; }
        .image-item:hover { transform: scale(1.05); }
        .progress-container { background: #f1f5f9; border-radius: 10px; padding: 1rem; margin: 1rem 0; }
        .footer { text-align: center; padding: 3rem 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 16px; margin-top: 3rem; box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3); }
        .dataframe { border: none !important; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05); }
        .section-header { font-size: 1.8rem; font-weight: 600; color: #1f2937; margin: 2rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 3px solid #667eea; }
    </style>
    """, unsafe_allow_html=True)

def page_dashboard():
    """Renders the Dashboard page."""
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
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Initialize Sample Dataset", type="primary", use_container_width=True):
            with st.spinner("🔄 Generating sample flood risk data..."):
                time.sleep(2)
                df_sample = generate_sample_flood_data(1000)
                sample_images = generate_sample_satellite_images(12)
                st.session_state.df_flood = df_sample
                st.session_state.sample_images = sample_images
                st.session_state.dataset_loaded = True
                st.session_state.models_trained = False
                st.session_state.model_results = {}
                st.session_state.X_test = None
                st.session_state.y_test = None
                st.session_state.X_train = None
                st.session_state.y_train = None
                st.session_state.scaler = None
            st.balloons()
            st.success("✅ Sample dataset loaded successfully!")
            st.rerun()
    if st.session_state.dataset_loaded:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(create_metric_card("Data Records", f"{len(st.session_state.df_flood):,}", "📋"), unsafe_allow_html=True)
        with col2:
            st.markdown(create_metric_card("Features", f"{len(st.session_state.df_flood.columns)-1}", "📊"), unsafe_allow_html=True)
        with col3:
            st.markdown(create_metric_card("Satellite Images", "12", "🛰️"), unsafe_allow_html=True)
        with col4:
            models_count = len(st.session_state.model_results) if st.session_state.models_trained else 0
            st.markdown(create_metric_card("Trained Models", f"{models_count}", "🤖"), unsafe_allow_html=True)

def page_data_analysis():
    """Renders the Data Analysis page."""
    st.markdown('<div class="section-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    if not st.session_state.dataset_loaded or st.session_state.df_flood is None:
        st.markdown(create_alert_card("""
            ⚠️ <strong>Dataset Required</strong><br>
            Please load the sample dataset first from the Dashboard page to begin analysis.
        """, "warning"), unsafe_allow_html=True)
        st.stop()
    df = st.session_state.df_flood
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(create_metric_card("Total Records", f"{len(df):,}", "📋"), unsafe_allow_html=True)
    with col2:
        st.markdown(create_metric_card("Features", f"{len(df.columns)-1}", "📊"), unsafe_allow_html=True)
    with col3:
        avg_risk = df['FloodProbability'].mean()
        st.markdown(create_metric_card("Avg Risk", f"{avg_risk:.2%}", "🎯"), unsafe_allow_html=True)
    with col4:
        high_risk_count = (df['FloodProbability'] > 0.7).sum()
        st.markdown(create_metric_card("High Risk Areas", f"{high_risk_count:,}", "🚨"), unsafe_allow_html=True)
    st.markdown("#### 🔥 Feature Correlation Matrix")
    correlations = df.corr(numeric_only=True)['FloodProbability'].abs().sort_values(ascending=False)[1:11]
    top_features = correlations.index.tolist() + ['FloodProbability']
    corr_matrix = df[top_features].corr(numeric_only=True)
    fig_heatmap = px.imshow(
        corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="RdYlBu_r",
        title="🔥 Top 10 Features Correlation with Flood Probability"
    )
    fig_heatmap.update_layout(height=600)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        fig_hist = px.histogram(
            df, x='FloodProbability', nbins=50, title="🎯 Flood Probability Distribution",
            color_discrete_sequence=['#667eea'], marginal="box"
        )
        fig_hist.update_layout(showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)
    with col2:
        df_risk = df.copy()
        df_risk['Risk_Category'] = pd.cut(
            df_risk['FloodProbability'], bins=[0, 0.3, 0.6, 1.0],
            labels=['🟢 Low', '🟡 Medium', '🔴 High'], right=False
        )
        risk_counts = df_risk['Risk_Category'].value_counts()
        fig_pie = px.pie(
            values=risk_counts.values, names=risk_counts.index,
            title="🎯 Risk Level Distribution", color_discrete_sequence=['#10b981', '#f59e0b', '#ef4444']
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown("#### 🔍 Feature Impact Analysis")
    correlations = df.corr(numeric_only=True)['FloodProbability'].abs().sort_values(ascending=False)[1:]
    fig_corr_bar = px.bar(
        x=correlations.values, y=correlations.index, orientation='h',
        title="🔍 Feature Correlation with Flood Probability",
        color=correlations.values, color_continuous_scale="Viridis",
        labels={'x': 'Correlation Strength', 'y': 'Features'}
    )
    fig_corr_bar.update_layout(height=700)
    st.plotly_chart(fig_corr_bar, use_container_width=True)
    st.markdown("#### 📈 Statistical Summary")
    summary_stats = df.describe()
    st.dataframe(summary_stats, use_container_width=True)

def page_model_training():
    """Renders the Model Training page."""
    st.markdown('<div class="section-header">⚙️ Advanced Model Training</div>', unsafe_allow_html=True)
    if not st.session_state.dataset_loaded or st.session_state.df_flood is None:
        st.markdown(create_alert_card("""
            ⚠️ <strong>Dataset Required</strong><br>
            Please load the sample dataset first from the Dashboard page to begin training.
        """, "warning"), unsafe_allow_html=True)
        st.stop()
    df = st.session_state.df_flood
    st.markdown("#### ⚙️ Training Configuration")
    with st.form("training_config"):
        col1, col2, col3 = st.columns(3)
        with col1:
            scaler_type = st.selectbox(
                "📊 Data Scaler:", ["StandardScaler", "MinMaxScaler", "RobustScaler"],
                help="Choose the scaling method for feature normalization"
            )
            test_size = st.slider(
                "🎯 Test Set Size:", 0.1, 0.4, 0.2, 0.05,
                help="Proportion of data to use for testing"
            )
        with col2:
            cv_folds = st.slider(
                "🔄 Cross-Validation Folds:", 3, 10, 5,
                help="Number of folds for cross-validation"
            )
            random_state = st.number_input(
                "🎲 Random State:", value=42, help="Seed for reproducibility"
            )
        with col3:
            enable_hyperparameter_tuning = st.checkbox(
                "🔧 Hyperparameter Tuning (Advanced)",
                help="Enable automated hyperparameter optimization (slower but potentially better results). Not fully implemented for all models."
            )
            parallel_training = st.checkbox(
                "⚡ Parallel Training", value=True,
                help="Use parallel processing for faster training (Note: Streamlit's nature might limit true parallelism for some operations)"
            )
        st.markdown("#### 🎯 Model Selection")
        models = get_model_algorithms()
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_models = st.multiselect(
                "Choose models to train:", list(models.keys()),
                default=list(models.keys())[:5],
                help="Select one or more models for training and comparison"
            )
        with col2:
            training_requested = st.form_submit_button("🚀 Start Training", type="primary", use_container_width=True)
    if training_requested:
        if not selected_models:
            st.markdown(create_alert_card("""
                ❌ <strong>No Models Selected</strong><br>
                Please select at least one model to train.
            """, "error"), unsafe_allow_html=True)
            st.stop()
        X = df.drop('FloodProbability', axis=1)
        y = df['FloodProbability']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        st.session_state.X_train = X_train
        st.session_state.y_train = y_train
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        scaler = None
        if scaler_type == "StandardScaler":
            scaler = StandardScaler()
        elif scaler_type == "MinMaxScaler":
            scaler = MinMaxScaler()
        elif scaler_type == "RobustScaler":
            scaler = RobustScaler()
        if scaler:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            st.session_state.scaler = scaler
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        st.session_state.model_results = {}
        st.session_state.trained_models = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i, model_name in enumerate(selected_models):
            status_text.text(f"🚀 Training {model_name}...")
            model = models[model_name]
            start_time = time.time()
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds, scoring='r2')
                st.session_state.model_results[model_name] = {
                    'MSE': mse, 'RMSE': rmse, 'R2': r2, 'MAE': mae,
                    'CV_R2_Mean': np.mean(cv_scores), 'CV_R2_Std': np.std(cv_scores),
                    'Time': time.time() - start_time
                }
                st.session_state.trained_models[model_name] = model
            except Exception as e:
                st.session_state.model_results[model_name] = {'Error': str(e)}
                st.error(f"Error training {model_name}: {e}")
            progress_bar.progress((i + 1) / len(selected_models))
        status_text.text("✅ Training complete!")
        st.session_state.models_trained = True
        st.success("All selected models have been trained!")
        st.rerun()
    if st.session_state.models_trained:
        st.markdown("#### 📈 Training Results Overview")
        results_df = pd.DataFrame(st.session_state.model_results).T
        st.dataframe(results_df.style.highlight_max(axis=0, subset=['R2', 'CV_R2_Mean']).highlight_min(axis=0, subset=['MSE', 'RMSE', 'MAE']), use_container_width=True)
        best_model_name = results_df['R2'].idxmax()
        st.markdown(create_alert_card(
            f"🏆 <strong>Best Performing Model:</strong> {best_model_name} with R2 Score: {results_df.loc[best_model_name]['R2']:.4f}",
            "success"
        ), unsafe_allow_html=True)

def page_risk_prediction():
    """Renders the Risk Prediction page."""
    st.markdown('<div class="section-header">🔮 Real-time Risk Prediction</div>', unsafe_allow_html=True)
    if not st.session_state.models_trained or not st.session_state.trained_models:
        st.markdown(create_alert_card("""
            ⚠️ <strong>Models Not Trained</strong><br>
            Please train models first from the Model Training page to make predictions.
        """, "warning"), unsafe_allow_html=True)
        st.stop()
    st.markdown("#### 📝 Enter New Data for Prediction")
    df_input = st.session_state.df_flood.drop('FloodProbability', axis=1)
    sample_input = df_input.iloc[0].to_dict() if not df_input.empty else {feature: 0.5 for feature in get_model_algorithms().values().__iter__().__next__().feature_names_in_}
    input_data = {}
    with st.form("prediction_form"):
        for i, feature in enumerate(sample_input.keys()):
            input_data[feature] = st.slider(f"**{feature.replace('_', ' ').title()}**", 0.0, 1.0, float(sample_input[feature]), 0.01)
        selected_prediction_model = st.selectbox(
            "Select Model for Prediction:",
            list(st.session_state.trained_models.keys()),
            help="Choose the trained model to use for predicting flood risk."
        )
        predict_button = st.form_submit_button("🔮 Predict Flood Risk", type="primary", use_container_width=True)
    if predict_button:
        if selected_prediction_model:
            model = st.session_state.trained_models[selected_prediction_model]
            input_df = pd.DataFrame([input_data])
            if st.session_state.scaler:
                input_scaled = st.session_state.scaler.transform(input_df)
            else:
                input_scaled = input_df
            prediction = model.predict(input_scaled)[0]
            st.markdown("#### 📊 Prediction Result")
            if prediction < 0.3:
                risk_level = "Low"
                risk_class = "risk-low"
            elif prediction < 0.6:
                risk_level = "Medium"
                risk_class = "risk-medium"
            else:
                risk_level = "High"
                risk_class = "risk-high"
            st.markdown(f"""
            <div style="text-align: center;">
                <h3>Predicted Flood Probability: <span style="font-size: 2.5rem; color: #667eea;">{prediction:.4f}</span></h3>
                <div class="{risk_class}">
                    Risk Level: {risk_level}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(create_alert_card(
                f"The predicted flood probability using the <strong>{selected_prediction_model}</strong> model is <strong>{prediction:.4f}</strong>. This indicates a <strong>{risk_level}</strong> risk level.",
                "info"
            ), unsafe_allow_html=True)
        else:
            st.markdown(create_alert_card("❌ Please select a model for prediction.", "error"), unsafe_allow_html=True)

def page_satellite_analysis():
    """Renders the Satellite Analysis page."""
    st.markdown('<div class="section-header">🛰️ Satellite Imagery Analysis</div>', unsafe_allow_html=True)
    if not st.session_state.dataset_loaded or st.session_state.sample_images is None:
        st.markdown(create_alert_card("""
            ⚠️ <strong>Sample Images Required</strong><br>
            Please load the sample dataset from the Dashboard page to view satellite images.
        """, "warning"), unsafe_allow_html=True)
        st.stop()
    st.markdown("#### 🖼️ Sample Satellite Images")
    st.markdown("<p>These are simulated satellite images demonstrating potential flood scenarios and terrain types.</p>", unsafe_allow_html=True)
    images_per_row = 4
    cols = st.columns(images_per_row)
    for i, img in enumerate(st.session_state.sample_images):
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        with cols[i % images_per_row]:
            st.markdown(f"""
            <div class="image-item">
                <img src="data:image/png;base64,{img_str}" style="width:100%; height:auto; display:block;">
                <p style="text-align:center; font-size:0.8em; margin-top:0.5rem;">Image {i+1}</p>
            </div>
            """, unsafe_allow_html=True)

def page_results_export():
    """Renders the Results & Export page."""
    st.markdown('<div class="section-header">📈 Results & Export</div>', unsafe_allow_html=True)
    if not st.session_state.models_trained or not st.session_state.model_results:
        st.markdown(create_alert_card("""
            ⚠️ <strong>No Results Available</strong><br>
            Please train models first from the Model Training page to view results and export data.
        """, "warning"), unsafe_allow_html=True)
        st.stop()
    st.markdown("#### 📊 Model Performance Summary")
    results_df = pd.DataFrame(st.session_state.model_results).T
    st.dataframe(results_df.style.highlight_max(axis=0, subset=['R2', 'CV_R2_Mean']).highlight_min(axis=0, subset=['MSE', 'RMSE', 'MAE']), use_container_width=True)
    st.markdown("#### ⬇️ Export Data")
    col1, col2 = st.columns(2)
    with col1:
        csv_data = st.session_state.df_flood.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Sample Data (CSV)",
            data=csv_data,
            file_name="flood_sample_data.csv",
            mime="text/csv",
            type="secondary",
            use_container_width=True
        )
    with col2:
        excel_data = BytesIO()
        results_df.to_excel(excel_data, index=True, sheet_name='Model Results')
        excel_data.seek(0)
        st.download_button(
            label="Download Model Results (Excel)",
            data=excel_data,
            file_name="model_training_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="secondary",
            use_container_width=True
        )

# --- Main Application Logic --- #
def main():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    apply_custom_css()
    init_session_state()

    st.markdown(f'<h1 class="main-header">{APP_ICON} FloodSentinel Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced AI-Powered Flood Risk Assessment System</p>', unsafe_allow_html=True)

    st.sidebar.markdown("### 🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["🏠 Dashboard", "📊 Data Analysis", "⚙️ Model Training", "🔮 Risk Prediction", "🛰️ Satellite Analysis", "📈 Results & Export"],
        help="Navigate through different sections of the application"
    )

    if page == "🏠 Dashboard":
        page_dashboard()
    elif page == "📊 Data Analysis":
        page_data_analysis()
    elif page == "⚙️ Model Training":
        page_model_training()
    elif page == "🔮 Risk Prediction":
        page_risk_prediction()
    elif page == "🛰️ Satellite Analysis":
        page_satellite_analysis()
    elif page == "📈 Results & Export":
        page_results_export()

    st.markdown("""
    <div class="footer">
        <p>&copy; 2025 FloodSentinel Pro. All rights reserved.</p>
        <p>Developed with ❤️ using Streamlit and Scikit-learn.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

