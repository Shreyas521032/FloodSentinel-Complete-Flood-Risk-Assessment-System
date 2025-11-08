import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
import os
import zipfile
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import json
import io
import pickle
import torch
import torch.nn as nn
from torchvision import models as torch_models
import timm

warnings.filterwarnings('ignore')

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="FloodSentinel - AI-Powered Flood Risk Assessment",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    
    .success-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        color: #333;
        margin: 1rem 0;
    }
    
    .footer {
        position: relative;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: transparent;
        color: #888;
        text-align: center;
        padding: 10px;
        font-size: 0.9em;
        transition: background-color 0.3s, color 0.3s;
    }
    .footer:hover {
        background-color: #f0f2f6;
        color: #2a5298;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE INITIALIZATION ====================
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'dataset_loaded' not in st.session_state:
    st.session_state.dataset_loaded = False
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}
if 'X_test_scaled' not in st.session_state:
    st.session_state.X_test_scaled = None
if 'pca_components' not in st.session_state:
    st.session_state.pca_components = None
if 'pca_model' not in st.session_state:
    st.session_state.pca_model = None
if 'df_flood' not in st.session_state:
    st.session_state.df_flood = None
if 'sat_files' not in st.session_state:
    st.session_state.sat_files = []
if 'ensemble_models' not in st.session_state:
    st.session_state.ensemble_models = {}
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

# ==================== DEEP LEARNING MODEL LOADING FUNCTIONS ====================

def load_pretrained_models(models_dir="pretrained_models"):
    """Load all pre-trained models from the directory"""
    loaded_models = {}
    
    if not os.path.exists(models_dir):
        st.warning(f"⚠️ Models directory '{models_dir}' not found. Place your model files in this directory.")
        return loaded_models
    
    try:
        # Load PyTorch models
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ResNet
        if os.path.exists(os.path.join(models_dir, "resnet_model_checkpoint.pth")):
            resnet = torch_models.resnet50(pretrained=False)
            resnet.fc = nn.Linear(resnet.fc.in_features, 2)
            resnet.load_state_dict(torch.load(os.path.join(models_dir, "resnet_model_checkpoint.pth"), map_location=device))
            resnet.to(device)
            resnet.eval()
            loaded_models['resnet'] = resnet
            st.success("✅ ResNet-50 loaded")
        
        # DenseNet
        if os.path.exists(os.path.join(models_dir, "densenet_model_checkpoint.pth")):
            densenet = torch_models.densenet121(pretrained=False)
            densenet.classifier = nn.Linear(densenet.classifier.in_features, 2)
            densenet.load_state_dict(torch.load(os.path.join(models_dir, "densenet_model_checkpoint.pth"), map_location=device))
            densenet.to(device)
            densenet.eval()
            loaded_models['densenet'] = densenet
            st.success("✅ DenseNet-121 loaded")
        
        # EfficientNet
        if os.path.exists(os.path.join(models_dir, "efficientnet_model_checkpoint.pth")):
            efficientnet = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
            efficientnet.load_state_dict(torch.load(os.path.join(models_dir, "efficientnet_model_checkpoint.pth"), map_location=device))
            efficientnet.to(device)
            efficientnet.eval()
            loaded_models['efficientnet'] = efficientnet
            st.success("✅ EfficientNet-B0 loaded")
        
        # ViT
        if os.path.exists(os.path.join(models_dir, "vit_model_checkpoint.pth")):
            vit = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
            vit.load_state_dict(torch.load(os.path.join(models_dir, "vit_model_checkpoint.pth"), map_location=device))
            vit.to(device)
            vit.eval()
            loaded_models['vit'] = vit
            st.success("✅ Vision Transformer loaded")
        
        # Load ensemble models
        if os.path.exists(os.path.join(models_dir, "meta_model.pkl")):
            with open(os.path.join(models_dir, "meta_model.pkl"), 'rb') as f:
                loaded_models['meta_model'] = pickle.load(f)
            st.success("✅ Meta-learner loaded")
        
        if os.path.exists(os.path.join(models_dir, "xgb_meta_model.pkl")):
            with open(os.path.join(models_dir, "xgb_meta_model.pkl"), 'rb') as f:
                loaded_models['xgb_meta'] = pickle.load(f)
            st.success("✅ XGBoost meta-learner loaded")
        
        # Load stacking models
        if os.path.exists(os.path.join(models_dir, "cnn_stacking_logistic.pkl")):
            with open(os.path.join(models_dir, "cnn_stacking_logistic.pkl"), 'rb') as f:
                loaded_models['stacking_logistic'] = pickle.load(f)
            st.success("✅ Stacking Logistic Regression loaded")
        
        if os.path.exists(os.path.join(models_dir, "cnn_stacking_xgb_model.pkl")):
            with open(os.path.join(models_dir, "cnn_stacking_xgb_model.pkl"), 'rb') as f:
                loaded_models['stacking_xgb'] = pickle.load(f)
            st.success("✅ Stacking XGBoost loaded")
        
        # Load metrics
        if os.path.exists(os.path.join(models_dir, "ensemble_metrics.pkl")):
            with open(os.path.join(models_dir, "ensemble_metrics.pkl"), 'rb') as f:
                loaded_models['metrics'] = pickle.load(f)
        
        # Load aggregator ensemble predictions
        if os.path.exists(os.path.join(models_dir, "cnn_aggregator_ensemble_predictions.pkl")):
            with open(os.path.join(models_dir, "cnn_aggregator_ensemble_predictions.pkl"), 'rb') as f:
                loaded_models['aggregator_predictions'] = pickle.load(f)
        
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
    
    return loaded_models

def preprocess_satellite_image(image, target_size=(224, 224)):
    """Preprocess satellite image for CNN prediction"""
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize(target_size)
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Apply CLAHE for contrast enhancement
        lab = cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
        
        return enhanced
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_with_ensemble(image, models_dict):
    """Make prediction using ensemble of models"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    predictions = {}
    
    # Preprocess image
    img_array = preprocess_satellite_image(image)
    if img_array is None:
        return None
    
    # Convert to tensor for PyTorch models
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    # Get predictions from each model
    with torch.no_grad():
        for model_name in ['resnet', 'densenet', 'efficientnet', 'vit']:
            if model_name in models_dict:
                try:
                    output = models_dict[model_name](img_tensor)
                    prob = torch.softmax(output, dim=1)[0, 1].cpu().numpy()
                    predictions[model_name] = float(prob)
                except Exception as e:
                    st.warning(f"⚠️ Error with {model_name}: {str(e)}")
    
    return predictions

def create_false_color_composite(image):
    """Create false color composite for better flood visualization"""
    try:
        img_array = np.array(image)
        r = img_array[:, :, 0]
        g = img_array[:, :, 1]
        b = img_array[:, :, 2]
        
        nir = 255 - b
        false_color = np.stack([nir, r, g], axis=-1)
        
        return Image.fromarray(false_color.astype(np.uint8))
    except Exception as e:
        return image

def extract_water_mask(image):
    """Extract potential water areas using color thresholding"""
    try:
        img_array = np.array(image)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        lower_water = np.array([0, 0, 0])
        upper_water = np.array([180, 255, 100])
        
        water_mask = cv2.inRange(hsv, lower_water, upper_water)
        
        kernel = np.ones((5, 5), np.uint8)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)
        
        water_mask_rgb = cv2.cvtColor(water_mask, cv2.COLOR_GRAY2RGB)
        water_mask_rgb[:, :, 0] = 0
        water_mask_rgb[:, :, 2] = 0
        
        return Image.fromarray(water_mask_rgb)
    except Exception as e:
        return image

# ==================== UTILITY FUNCTIONS ====================

def create_sample_data():
    """Create sample flood prediction data for demonstration with global coordinates"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'MonsoonIntensity': np.random.uniform(0.1, 1.0, n_samples),
        'TopographyDrainage': np.random.uniform(0.0, 1.0, n_samples),
        'RiverManagement': np.random.uniform(0.0, 1.0, n_samples),
        'Deforestation': np.random.uniform(0.0, 1.0, n_samples),
        'Urbanization': np.random.uniform(0.0, 1.0, n_samples),
        'ClimateChange': np.random.uniform(0.0, 1.0, n_samples),
        'DamsQuality': np.random.uniform(0.0, 1.0, n_samples),
        'Siltation': np.random.uniform(0.0, 1.0, n_samples),
        'AgriculturalPractices': np.random.uniform(0.0, 1.0, n_samples),
        'Encroachments': np.random.uniform(0.0, 1.0, n_samples),
        'IneffectiveDisasterPreparedness': np.random.uniform(0.0, 1.0, n_samples),
        'DrainageSystems': np.random.uniform(0.0, 1.0, n_samples),
        'CoastalVulnerability': np.random.uniform(0.0, 1.0, n_samples),
        'Landslides': np.random.uniform(0.0, 1.0, n_samples),
        'Watersheds': np.random.uniform(0.0, 1.0, n_samples),
        'DeterioratingInfrastructure': np.random.uniform(0.0, 1.0, n_samples),
        'PopulationScore': np.random.uniform(0.0, 1.0, n_samples),
        'WetlandLoss': np.random.uniform(0.0, 1.0, n_samples),
        'InadequatePlanning': np.random.uniform(0.0, 1.0, n_samples),
        'PoliticalFactors': np.random.uniform(0.0, 1.0, n_samples),
        'Latitude': np.random.uniform(-90, 90, n_samples),
        'Longitude': np.random.uniform(-180, 180, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    df['FloodProbability'] = (
        0.3 * df['MonsoonIntensity'] +
        0.15 * df['ClimateChange'] +
        0.1 * (1 - df['RiverManagement']) +
        0.1 * (1 - df['DamsQuality']) +
        0.1 * df['Deforestation'] +
        0.1 * df['Urbanization'] +
        0.05 * df['Siltation'] +
        0.1 * np.random.uniform(0, 0.2, n_samples)
    )
    
    df['FloodProbability'] = np.clip(df['FloodProbability'], 0, 1)
    
    return df

def get_model_algorithms():
    """Get default state-of-the-art algorithms"""
    return {
        "🌳 Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "🚀 XGBoost": xgb.XGBRegressor(random_state=42, n_estimators=100),
        "🧠 Neural Network": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
        "⚡ Gradient Boosting": GradientBoostingRegressor(random_state=42, n_estimators=100),
        "🎯 CatBoost": CatBoostRegressor(verbose=False, random_state=42, iterations=100),
    }

# ==================== MAIN APPLICATION ====================

st.markdown('<h1 class="main-header">🌊 FloodSentinel</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Flood Risk Assessment Using Multi-Temporal Satellite Imagery and Deep Neural Networks</p>', unsafe_allow_html=True)

# ==================== SIDEBAR NAVIGATION ====================

st.sidebar.markdown("### 🧭 Navigation")
page = st.sidebar.selectbox(
    "Choose a section:",
    ["🏠 Home", "📊 Data Analysis", "⚙️ Model Training", "🔮 Predictions", 
     "🛰️ Satellite Analysis", "🖼️ Image Flood Detection", "📈 Results Dashboard"]
)
st.sidebar.markdown("---") 
st.sidebar.link_button(
    "🌐 Ask the Sentinel Chatbot", 
    "https://flood-app-repo-chatbot-sck.streamlit.app/", 
    type="secondary", 
    help="Redirects to the complete Flood Risk Assessment System's AI Chatbot tab." 
)

# ==================== PAGE: HOME ====================

if page == "🏠 Home":
    st.markdown("### 🎯 Project Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>🌊 Problem Statement</h4>
            <p>Floods remain among the most destructive natural hazards globally, causing widespread loss of life, economic disruption, and environmental damage. Current flood prediction systems face critical limitations in data-scarce regions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-box">
            <h4>🎯 Our Solution</h4>
            <p>FloodSentinel combines machine learning for historical tabular data with deep neural networks for multi-temporal satellite imagery, embedding hydrological knowledge for enhanced interpretability.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
            <h4>🚀 Key Features</h4>
            <ul>
                <li>⚙️ State-of-the-art ML algorithms</li>
                <li>🛰️ Multi-temporal satellite imagery analysis</li>
                <li>📊 Real-time flood risk assessment</li>
                <li>🎯 Hydrological knowledge integration</li>
                <li>📈 Interactive visualizations</li>
                <li>🖼️ Ensemble deep learning models</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Dataset Loading")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Use Sample Data", type="primary", key="load_sample"):
            st.session_state.df_flood = create_sample_data()
            st.session_state.sat_files = []
            st.session_state.dataset_loaded = True
            st.success("✅ Sample dataset loaded successfully!")
            st.rerun()
    
    with col2:
        st.info("💡 Click 'Use Sample Data' to start exploring FloodSentinel")
    
    if st.session_state.dataset_loaded:
        st.success(f"✅ Dataset loaded with {len(st.session_state.df_flood)} records!")
        st.dataframe(st.session_state.df_flood.head(), use_container_width=True)

# ==================== PAGE: DATA ANALYSIS ====================

elif page == "📊 Data Analysis":
    st.markdown("### 📊 Exploratory Data Analysis")
    
    if not st.session_state.dataset_loaded:
        st.warning("⚠️ Please load datasets first from the Home page")
        st.stop()
    
    df = st.session_state.df_flood
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h3>📋 Records</h3>
            <h2>{len(df)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <h3>📊 Features</h3>
            <h2>{len(df.columns)-1}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <h3>🎯 Target</h3>
            <h2>FloodProbability</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <h3>🔢 PCA Components</h3>
            <h2>10</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("#### 📈 Feature Distribution Analysis")
    
    corr_matrix = df.corr(numeric_only=True).round(2)
    fig_corr = px.imshow(corr_matrix, 
                         text_auto=True, 
                         aspect="auto",
                         color_continuous_scale=px.colors.sequential.RdBu,
                         title="🔥 Feature Correlation Heatmap")
    fig_corr.update_layout(height=700, width=700)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if "FloodProbability" in df.columns:
            fig_hist = px.histogram(
                df, 
                x="FloodProbability",
                nbins=30,
                title="🎯 Flood Probability Distribution",
                color_discrete_sequence=["#4facfe"]
            )
            fig_hist.update_layout(bargap=0.1)
            st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        if "FloodProbability" in df.columns:
            fig_box = px.box(
                df, 
                y="FloodProbability",
                title="📊 Flood Probability Box Plot",
                color_discrete_sequence=["#fa709a"]
            )
            st.plotly_chart(fig_box, use_container_width=True)
    
    st.markdown("#### 🎯 Top Features Analysis")
    if "FloodProbability" in df.columns:
        correlations = df.corr(numeric_only=True)["FloodProbability"].abs().sort_values(ascending=False)[1:]
        
        fig_corr_bar = px.bar(
            x=correlations.values,
            y=correlations.index,
            orientation="h",
            title="🔍 Feature Correlation with Flood Probability",
            color=correlations.values,
            color_continuous_scale="Viridis"
        )
        fig_corr_bar.update_layout(height=600)
        st.plotly_chart(fig_corr_bar, use_container_width=True)

# ==================== PAGE: MODEL TRAINING ====================

elif page == "⚙️ Model Training":
    st.markdown("### ⚙️ State-of-the-Art Model Training")
    
    if not st.session_state.dataset_loaded:
        st.warning("⚠️ Please load datasets first from the Home page")
        st.stop()
    
    df = st.session_state.df_flood
    
    if "FloodProbability" not in df.columns:
        st.error("FloodProbability column not found in dataset")
        st.stop()
    
    st.markdown("#### ⚙️ Preprocessing Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        scaler_type = st.selectbox(
            "📊 Choose Scaler:",
            ["StandardScaler", "MinMaxScaler", "RobustScaler"]
        )
        
        test_size = st.slider("🎯 Test Size:", 0.1, 0.4, 0.2, 0.05)
    
    with col2:
        cv_folds = st.slider("🔄 Cross-Validation Folds:", 3, 10, 5)
        
        pca_components = st.slider("🔢 PCA Components:", 5, 20, 10)

    st.markdown("#### 🎯 Model Selection")
    
    models = get_model_algorithms()
    selected_models = st.multiselect(
        "Choose models to train:",
        list(models.keys()),
        default=list(models.keys())  # All default models selected
    )
    
    if st.button("🚀 Train Models", type="primary", key="train_models"):
        if not selected_models:
            st.error("❌ Please select at least one model")
            st.stop()
        
        X = df.drop("FloodProbability", axis=1)
        y = df["FloodProbability"]
        
        categorical_cols = X.select_dtypes(include='object').columns
        if len(categorical_cols) > 0:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        if scaler_type == "StandardScaler":
            scaler = StandardScaler()
        elif scaler_type == "MinMaxScaler":
            scaler = MinMaxScaler()
        else:
            scaler = RobustScaler()
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Apply PCA with specified components
        pca = PCA(n_components=pca_components)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        st.session_state.pca_components = pca.components_
        st.session_state.pca_model = pca
        st.session_state.pca_feature_names = X.columns
        
        st.info(f"✅ PCA applied: Reduced from {X.shape[1]} to {pca_components} features")
        st.info(f"📊 Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
        
        results = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, model_name in enumerate(selected_models):
            status_text.text(f"🔄 Training {model_name}...")
            
            model = models[model_name]
            
            start_time = time.time()
            try:
                # Train on PCA-transformed data
                model.fit(X_train_pca, y_train)
                y_pred = model.predict(X_test_pca)
                cv_scores = cross_val_score(model, X_train_pca, y_train, cv=cv_folds, scoring="r2")

                training_time = time.time() - start_time
                
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results[model_name] = {
                    "MSE": mse,
                    "RMSE": rmse,
                    "MAE": mae,
                    "R²": r2,
                    "CV_Mean": cv_scores.mean(),
                    "CV_Std": cv_scores.std(),
                    "Training_Time": training_time,
                    "Model": model,
                    "Predictions": y_pred
                }
            except Exception as e:
                st.error(f"Error training {model_name}: {str(e)}")
                
            progress_bar.progress((i + 1) / len(selected_models))
        
        if results:
            st.session_state.model_results = results
            st.session_state.models_trained = True
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.scaler = scaler
            st.session_state.X_test_scaled = X_test_scaled
            st.session_state.X_test_pca = X_test_pca
            
            status_text.text("✅ All models trained successfully!")
            st.success("🎉 Model training completed!")
            
            # Display PCA variance plot
            st.markdown("#### 📊 PCA Explained Variance")
            fig_pca = px.bar(
                x=list(range(1, len(pca.explained_variance_ratio_) + 1)),
                y=pca.explained_variance_ratio_,
                labels={'x': 'Principal Component', 'y': 'Explained Variance Ratio'},
                title='PCA Explained Variance by Component'
            )
            st.plotly_chart(fig_pca, use_container_width=True)
        else:
            st.error("❌ No models were successfully trained.")

# ==================== PAGE: PREDICTIONS ====================

elif page == "🔮 Predictions":
    st.markdown("### 🔮 Flood Risk Predictions")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first from the Model Training page")
        st.stop()
    
    df = st.session_state.df_flood
    
    st.markdown("#### 📝 Manual Prediction Input")
    
    with st.form("prediction_form"):
        st.markdown("##### 🌦️ Environmental Factors")
        
        col1, col2, col3 = st.columns(3)
        
        input_values = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        non_target_cols = [col for col in numeric_cols if col != 'FloodProbability']
        
        for i, col in enumerate(non_target_cols):
            col_idx = i % 3
            with [col1, col2, col3][col_idx]:
                min_val = float(df[col].min()) if not df[col].isna().all() else 0.0
                max_val = float(df[col].max()) if not df[col].isna().all() else 1.0
                mean_val = float(df[col].mean()) if not df[col].isna().all() else 0.5
                
                input_values[col] = st.slider(
                    f"📊 {col}", 
                    min_val, 
                    max_val, 
                    mean_val
                )

        submitted = st.form_submit_button("🔮 Get Prediction")
        
        if submitted:
            input_data = pd.DataFrame([input_values])
            
            training_cols = st.session_state.X_test.columns.tolist()
            input_data = input_data.reindex(columns=training_cols, fill_value=0)

            # Scale the input
            input_scaled = st.session_state.scaler.transform(input_data)
            
            # Apply PCA transformation
            input_pca = st.session_state.pca_model.transform(input_scaled)
            
            st.markdown("#### 🎯 Prediction Results")
            st.info(f"📊 Input transformed to {input_pca.shape[1]} PCA components")
            
            predictions = {}
            for model_name, model_info in st.session_state.model_results.items():
                try:
                    pred = model_info['Model'].predict(input_pca)[0]
                    predictions[model_name] = pred
                except Exception as e:
                    st.error(f"Error making prediction with {model_name}: {str(e)}")
            
            if predictions:
                col1, col2 = st.columns(2)
                
                with col1:
                    ensemble_pred = np.mean(list(predictions.values()))
                    
                    if ensemble_pred < 0.3:
                        risk_level = "🟢 Low Risk"
                    elif ensemble_pred < 0.6:
                        risk_level = "🟡 Medium Risk"
                    else:
                        risk_level = "🔴 High Risk"
                    
                    st.markdown(f"""
                    <div class="metric-container">
                        <h3>🎯 Ensemble Prediction</h3>
                        <h1>{ensemble_pred:.2%}</h1>
                        <h4>{risk_level}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    pred_df = pd.DataFrame({
                        'Model': list(predictions.keys()),
                        'Prediction': [f"{p:.2%}" for p in predictions.values()],
                        'Risk_Level': [
                            "🟢 Low" if p < 0.3 else "🟡 Medium" if p < 0.6 else "🔴 High"
                            for p in predictions.values()
                        ]
                    })
                    st.dataframe(pred_df, use_container_width=True)
                
                fig_pred = px.bar(
                    x=list(predictions.keys()),
                    y=list(predictions.values()),
                    title="📊 Model Predictions Comparison",
                    color=list(predictions.values()),
                    color_continuous_scale="RdYlGn_r"
                )
                fig_pred.update_layout(height=400)
                st.plotly_chart(fig_pred, use_container_width=True)

# ==================== PAGE: SATELLITE ANALYSIS ====================

elif page == "🛰️ Satellite Analysis":
    st.markdown("### 🛰️ Pre-trained Deep Learning Models")
    
    st.markdown("""
    This section uses pre-trained state-of-the-art deep learning models for flood detection:
    - **ResNet-50**: Deep residual network
    - **DenseNet-121**: Densely connected network
    - **EfficientNet-B0**: Efficient architecture
    - **Vision Transformer (ViT)**: Attention-based model
    - **Ensemble Models**: Meta-learners and stacking ensembles
    """)
    
    st.markdown("#### 📁 Load Pre-trained Models")
    
    models_dir = st.text_input("Models directory path:", value="pretrained_models")
    
    if st.button("🔄 Load Pre-trained Models", type="primary"):
        with st.spinner("Loading models..."):
            loaded_models = load_pretrained_models(models_dir)
            
            if loaded_models:
                st.session_state.ensemble_models = loaded_models
                st.session_state.models_loaded = True
                st.success(f"✅ Successfully loaded {len(loaded_models)} model components!")
            else:
                st.error("❌ No models were loaded. Please check the directory path.")
    
    if st.session_state.models_loaded:
        st.markdown("#### 📊 Loaded Models Summary")
        
        model_info = []
        for model_name in st.session_state.ensemble_models.keys():
            model_info.append({
                'Component': model_name,
                'Type': 'CNN' if model_name in ['resnet', 'densenet', 'efficientnet', 'vit'] else 'Ensemble',
                'Status': '✅ Loaded'
            })
        
        if model_info:
            st.dataframe(pd.DataFrame(model_info), use_container_width=True)
        
        # Display metrics if available
        if 'metrics' in st.session_state.ensemble_models:
            st.markdown("#### 📈 Model Performance Metrics")
            metrics = st.session_state.ensemble_models['metrics']
            
            if isinstance(metrics, dict):
                metrics_df = pd.DataFrame(metrics).T
                st.dataframe(metrics_df, use_container_width=True)
    
    else:
        st.info("👆 Click 'Load Pre-trained Models' to load your trained models")
        
        st.markdown("#### 📋 Required Model Files")
        st.markdown("""
        Place these files in your models directory:
        - `resnet_model_checkpoint.pth`
        - `densenet_model_checkpoint.pth`
        - `efficientnet_model_checkpoint.pth`
        - `vit_model_checkpoint.pth`
        - `meta_model.pkl`
        - `xgb_meta_model.pkl`
        - `cnn_stacking_logistic.pkl`
        - `cnn_stacking_xgb_model.pkl`
        - `ensemble_metrics.pkl` (optional)
        """)

# ==================== PAGE: IMAGE FLOOD DETECTION ====================

elif page == "🖼️ Image Flood Detection":
    st.markdown("### 🖼️ Upload Satellite Image for Flood Detection")
    
    if not st.session_state.models_loaded:
        st.warning("⚠️ Please load pre-trained models first from the 'Satellite Analysis' page")
        st.info("You can still use traditional computer vision analysis without loading models")
    
    st.markdown("""
    Upload a satellite image to detect potential flood areas. The system will:
    - Analyze the image using ensemble deep learning models
    - Generate false color composites to highlight water bodies
    - Extract potential water masks
    - Provide a comprehensive flood probability assessment
    """)
    
    uploaded_file = st.file_uploader(
        "📤 Upload Satellite Image (JPG, PNG, or TIFF)",
        type=['jpg', 'jpeg', 'png', 'tif', 'tiff'],
        help="Upload a satellite image for flood detection analysis"
    )
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            
            st.markdown("#### 🖼️ Image Analysis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.image(image, caption="Original Satellite Image", use_container_width=True)
            
            with col2:
                false_color = create_false_color_composite(image)
                st.image(false_color, caption="False Color Composite", use_container_width=True)
            
            with col3:
                water_mask = extract_water_mask(image)
                st.image(water_mask, caption="Water Mask", use_container_width=True)
            
            st.markdown("---")
            
            # Deep Learning Prediction
            if st.session_state.models_loaded:
                st.markdown("#### 🤖 Ensemble Deep Learning Analysis")
                
                with st.spinner("Running ensemble predictions..."):
                    predictions = predict_with_ensemble(image, st.session_state.ensemble_models)
                
                if predictions:
                    # Calculate ensemble prediction
                    ensemble_pred = np.mean(list(predictions.values()))
                    
                    if ensemble_pred < 0.3:
                        risk_level = "🟢 Low Flood Risk"
                        risk_color = "#4facfe"
                    elif ensemble_pred < 0.7:
                        risk_level = "🟡 Moderate Flood Risk"
                        risk_color = "#fee140"
                    else:
                        risk_level = "🔴 High Flood Risk"
                        risk_color = "#fa709a"
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-container" style="background: {risk_color};">
                            <h3>🎯 Ensemble Prediction</h3>
                            <h1>{ensemble_pred:.1%}</h1>
                            <h4>{risk_level}</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display individual model predictions
                        st.markdown("##### 📊 Individual Models")
                        for model_name, pred in predictions.items():
                            st.metric(f"{model_name.upper()}", f"{pred:.2%}")
                    
                    with col2:
                        # Gauge chart
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=ensemble_pred * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Flood Risk Assessment"},
                            delta={'reference': 50},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': risk_color},
                                'steps': [
                                    {'range': [0, 30], 'color': "lightgreen"},
                                    {'range': [30, 70], 'color': "lightyellow"},
                                    {'range': [70, 100], 'color': "lightcoral"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 70
                                }
                            }
                        ))
                        fig_gauge.update_layout(height=300)
                        st.plotly_chart(fig_gauge, use_container_width=True)
                        
                        # Bar chart of individual predictions
                        fig_bar = px.bar(
                            x=list(predictions.keys()),
                            y=list(predictions.values()),
                            title="Model-wise Flood Probability",
                            labels={'x': 'Model', 'y': 'Probability'},
                            color=list(predictions.values()),
                            color_continuous_scale="RdYlGn_r"
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
                    
                    st.markdown("#### 📊 Detailed Analysis")
                    
                    # Calculate water percentage
                    water_mask_gray = np.array(water_mask.convert('L'))
                    water_percentage = (np.sum(water_mask_gray > 0) / water_mask_gray.size) * 100
                    
                    analysis_details = {
                        "Image Dimensions": f"{image.size[0]} x {image.size[1]} pixels",
                        "Ensemble Flood Probability": f"{ensemble_pred:.2%}",
                        "Risk Classification": risk_level,
                        "Water Coverage (CV)": f"{water_percentage:.1f}%",
                        "Model Consensus": "High" if np.std(list(predictions.values())) < 0.15 else "Moderate",
                        "Recommendation": "Immediate action required" if ensemble_pred > 0.7 else 
                                        "Enhanced monitoring recommended" if ensemble_pred > 0.3 else 
                                        "Continue routine monitoring"
                    }
                    
                    st.json(analysis_details)
                    
                else:
                    st.error("❌ Error making predictions with ensemble models")
            
            else:
                st.markdown("#### 📊 Traditional Computer Vision Analysis")
                st.warning("⚠️ Deep learning models not loaded. Using traditional CV methods.")
                
                water_mask_gray = np.array(water_mask.convert('L'))
                water_percentage = (np.sum(water_mask_gray > 0) / water_mask_gray.size) * 100
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h3>💧 Water Coverage</h3>
                        <h1>{water_percentage:.1f}%</h1>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    risk = "High" if water_percentage > 30 else "Moderate" if water_percentage > 15 else "Low"
                    st.markdown(f"""
                    <div class="metric-container">
                        <h3>⚠️ Risk Level</h3>
                        <h1>{risk}</h1>
                    </div>
                    """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    else:
        st.info("👆 Please upload a satellite image to begin flood detection analysis")
        
        st.markdown("#### 💡 Supported Image Types")
        st.markdown("""
        This tool can analyze:
        - 🛰️ Sentinel-1 SAR imagery
        - 🛰️ Sentinel-2 multispectral imagery
        - 🌍 Landsat imagery
        - 📸 Aerial photography
        - 🚁 Drone imagery of flood-prone areas
        """)

# ==================== PAGE: RESULTS DASHBOARD ====================

elif page == "📈 Results Dashboard":
    st.markdown("### 📈 Results Dashboard")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first from the Model Training page")
        st.stop()
    
    results = st.session_state.model_results
    
    st.markdown("#### 🏆 Model Performance Overview")
    
    perf_data = []
    for model_name, metrics in results.items():
        perf_data.append({
            'Model': model_name,
            'R² Score': metrics['R²'],
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'CV Mean': metrics['CV_Mean'],
            'CV Std': metrics['CV_Std'],
            'Training Time (s)': metrics['Training_Time']
        })
    
    perf_df = pd.DataFrame(perf_data)
    perf_df = perf_df.sort_values('R² Score', ascending=False)
    
    st.markdown("##### 🥇 Top Performing Models")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if len(perf_df) > 0:
            best_model = perf_df.iloc[0]
            st.markdown(f"""<div class="metric-container"><h4>🥇 Best Model</h4><h3>{best_model['Model']}</h3><p>R² Score: {best_model['R² Score']:.4f}</p></div>""", unsafe_allow_html=True)
    
    with col2:
        if len(perf_df) > 0:
            fastest_model = perf_df.loc[perf_df['Training Time (s)'].idxmin()]
            st.markdown(f"""<div class="metric-container"><h4>⚡ Fastest Model</h4><h3>{fastest_model['Model']}</h3><p>Time: {fastest_model['Training Time (s)']:.2f}s</p></div>""", unsafe_allow_html=True)
    
    with col3:
        if len(perf_df) > 0:
            most_stable = perf_df.loc[perf_df['CV Std'].idxmin()]
            st.markdown(f"""<div class="metric-container"><h4>🎯 Most Stable</h4><h3>{most_stable['Model']}</h3><p>CV Std: {most_stable['CV Std']:.4f}</p></div>""", unsafe_allow_html=True)
    
    st.markdown("##### 📊 Detailed Performance Metrics")
    st.dataframe(perf_df, use_container_width=True)
    
    st.markdown("#### 📊 Performance Visualizations")
    
    if len(perf_df) > 0:
        fig_r2 = px.bar(
            perf_df.sort_values('R² Score'),
            x='R² Score',
            y='Model',
            orientation='h',
            title='🎯 R² Score Comparison',
            color='R² Score',
            color_continuous_scale='Viridis'
        )
        fig_r2.update_layout(height=500)
        st.plotly_chart(fig_r2, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_rmse = px.bar(
                perf_df.sort_values('RMSE'),
                x='RMSE',
                y='Model',
                orientation='h',
                title='📉 RMSE Comparison (Lower is Better)',
                color='RMSE',
                color_continuous_scale='Reds'
            )
            fig_rmse.update_layout(height=400)
            st.plotly_chart(fig_rmse, use_container_width=True)
        
        with col2:
            fig_time = px.bar(
                perf_df.sort_values('Training Time (s)'),
                x='Training Time (s)',
                y='Model',
                orientation='h',
                title='⏱️ Training Time Comparison',
                color='Training Time (s)',
                color_continuous_scale='Blues'
            )
            fig_time.update_layout(height=400)
            st.plotly_chart(fig_time, use_container_width=True)
        
        # Export results
        st.markdown("#### 💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Download JSON Results", type="secondary"):
                results_json = {}
                for model_name, metrics in results.items():
                    results_json[model_name] = {
                        'MSE': float(metrics['MSE']),
                        'RMSE': float(metrics['RMSE']),
                        'MAE': float(metrics['MAE']),
                        'R²': float(metrics['R²']),
                        'CV_Mean': float(metrics['CV_Mean']),
                        'CV_Std': float(metrics['CV_Std']),
                        'Training_Time': float(metrics['Training_Time'])
                    }
                
                st.download_button(
                    label="Download",
                    data=json.dumps(results_json, indent=4),
                    file_name="floodsentinel_results.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📥 Download CSV Results", type="secondary"):
                csv_buffer = io.StringIO()
                perf_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Download",
                    data=csv_buffer.getvalue(),
                    file_name="floodsentinel_results.csv",
                    mime="text/csv"
                )

# ==================== SIDEBAR INFORMATION ====================

st.sidebar.markdown("---")

if st.session_state.dataset_loaded:
    st.sidebar.success("✅ Datasets Loaded")
else:
    st.sidebar.error("❌ Datasets Not Loaded")

if st.session_state.models_trained:
    st.sidebar.success("✅ ML Models Trained")
    st.sidebar.info(f"🎯 {len(st.session_state.model_results)} models ready")
else:
    st.sidebar.error("❌ ML Models Not Trained")

if st.session_state.models_loaded:
    st.sidebar.success("✅ DL Models Loaded")
    st.sidebar.info(f"🤖 {len(st.session_state.ensemble_models)} components")
else:
    st.sidebar.error("❌ DL Models Not Loaded")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Quick Stats")
if st.session_state.dataset_loaded:
    st.sidebar.metric("📋 Total Records", len(st.session_state.df_flood))

if st.session_state.pca_model:
    st.sidebar.metric("🔢 PCA Components", st.session_state.pca_model.n_components_)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info("""
🌊 **FloodSentinel** combines:
- ⚙️ 5 state-of-the-art ML algorithms
- 🛰️ Ensemble deep learning models
- 📊 Real-time risk assessment
- 🎯 PCA dimensionality reduction
- 📈 Comprehensive visualizations
- 🖼️ Multi-model flood detection
""")

# ==================== FOOTER ====================

st.markdown("---")
st.markdown("""
    <div class="footer">
        <p>Crafted with ❤️ by Shreyas, Chinmay and Kaivalya.<br>
        Project: FloodSentinel - AI-Powered Flood Risk Assessment</p>
    </div>
""", unsafe_allow_html=True)
