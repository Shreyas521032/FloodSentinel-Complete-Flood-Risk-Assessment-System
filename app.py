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
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
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
import kagglehub
import warnings
import os
import zipfile
import cv2
from PIL import Image
# import tensorflow as tf # Not needed for PyTorch DL
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import json

# --- Deep Learning Imports (from flood_dl_module) ---
import glob
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm
import rasterio
# ---------------------------------------------------

warnings.filterwarnings('ignore')

# --- Deep Learning Configuration (from flood_dl_module) ---
MODEL_NAME = 'resnet18'
IMAGE_SIZE = 224
NUM_CLASSES = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------------------------------------------

# --- Deep Learning Utility Functions (from flood_dl_module) ---

def load_label_map(data_dir):
    """
    Loads the flood label mapping from S1list.json and S2list.json.
    
    NOTE: This function assumes the JSON files are present in the data_dir/SEN12FLOOD/
    and follows the structure observed in the notebook.
    """
    s1_json_path = os.path.join(data_dir, "SEN12FLOOD", "S1list.json")
    s2_json_path = os.path.join(data_dir, "SEN12FLOOD", "S2list.json")
    
    flood_labels = {}
    
    if os.path.exists(s1_json_path):
        with open(s1_json_path, "r") as f:
            s1_data = json.load(f)
        for folder, details in s1_data.items():
            # Assuming FLOODING status is present in one of the nested entries
            flood_status = any(
                entry.get("FLOODING", False)
                for key, entry in details.items()
                if isinstance(entry, dict) and "FLOODING" in entry
            )
            flood_labels[folder] = int(flood_status)

    if os.path.exists(s2_json_path):
        with open(s2_json_path, "r") as f:
            s2_data = json.load(f)
        for folder, details in s2_data.items():
            # Update mapping using Sentinel-2 data
            flood_status = any(
                entry.get("FLOODING", False)
                for key, entry in details.items()
                if isinstance(entry, dict) and "FLOODING" in entry
            )
            flood_labels[folder] = int(flood_status)
            
    return flood_labels

def preprocess_tif_image(image_path, transform):
    """
    Reads a TIFF image (potentially multi-band) using rasterio,
    converts it to a displayable 3-channel image (RGB or normalized single-band),
    and applies the given PyTorch transform.
    """
    try:
        with rasterio.open(image_path) as src:
            if src.count >= 3:
                # Read first 3 bands (assuming they are suitable for RGB or similar visualization)
                arr = np.dstack([src.read(i) for i in range(1, 4)])
                
                # Normalize to 0-255 for visualization
                rgb = np.zeros_like(arr, dtype=np.uint8)
                for i in range(3):
                    band = arr[:, :, i]
                    lo, hi = np.nanmin(band), np.nanmax(band)
                    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                        rgb[:, :, i] = ((band - lo) / (hi - lo) * 255).astype(np.uint8)
                    else:
                        rgb[:, :, i] = 0 # Black if data is invalid
                
            else:
                # Fallback: single-band -> normalize -> replicate to 3 channels
                arr = src.read(1).astype(float)
                lo, hi = np.nanmin(arr), np.nanmax(arr)
                if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
                    arr = np.zeros_like(arr, dtype=np.uint8)
                else:
                    arr = ((arr - lo)/(hi - lo) * 255).astype(np.uint8)
                rgb = np.stack([arr]*3, axis=-1)

        # Convert to PIL and apply transform
        pil = Image.fromarray(rgb)
        return transform(pil)

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        # Return a blank tensor
        return torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE))

# --- PyTorch Model Definition ---

def create_model(model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=True):
    """
    Creates a pre-trained model from the timm library and adapts it for binary classification.
    """
    try:
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        return model
    except Exception as e:
        print(f"Error creating model {model_name}: {e}")
        # Fallback to a simple sequential model if timm fails
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * IMAGE_SIZE * IMAGE_SIZE, 1),
        )

# --- Prediction Function ---

def predict_flood(model, image_path, data_dir=None):
    """
    Predicts the flood status (0 or 1) for a given satellite image path.
    """
    model.eval()
    
    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        # Standard ImageNet normalization, a common practice for pre-trained models
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Preprocess the image
    image_tensor = preprocess_tif_image(image_path, transform)
    
    # Add batch dimension and move to device
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(image_tensor)
        
        # Apply sigmoid for binary classification probability
        probability = torch.sigmoid(output).item()
        
        # Determine the final label (0 or 1)
        prediction_label = 1 if probability >= 0.5 else 0
        
    return prediction_label, probability

# --- Placeholder for Model Loading (since we don't have a trained model file) ---

def load_trained_model(model_path=None):
    """
    Loads a trained model from a file. If no path is provided, returns an untrained model.
    """
    model = create_model()
    model.to(DEVICE)
    
    if model_path and os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load model state from {model_path}. Using untrained model. Error: {e}")
    else:
        print("Warning: No trained model file provided or found. Using an untrained model for demonstration.")
        
    return model

# --- Visualization Function (Simple) ---

def visualize_flood_detection(image_path, prediction_label, probability):
    """
    Generates a simple visualization of the flood detection result.
    
    Returns:
        A matplotlib figure object.
    """
    try:
        # Read the image using PIL for simple display
        if image_path.lower().endswith('.tif'):
            # For TIFF, use rasterio to get a displayable image
            with rasterio.open(image_path) as src:
                if src.count >= 3:
                    # Read first 3 bands
                    arr = np.dstack([src.read(i) for i in range(1, 4)])
                    
                    # Normalize to 0-255 for visualization
                    rgb = np.zeros_like(arr, dtype=np.uint8)
                    for i in range(3):
                        band = arr[:, :, i]
                        lo, hi = np.nanmin(band), np.nanmax(band)
                        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                            rgb[:, :, i] = ((band - lo) / (hi - lo) * 255).astype(np.uint8)
                        else:
                            rgb[:, :, i] = 0
                    img_to_display = Image.fromarray(rgb)
                else:
                    # Single band
                    arr = src.read(1).astype(float)
                    lo, hi = np.nanmin(arr), np.nanmax(arr)
                    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                        arr = ((arr - lo)/(hi - lo) * 255).astype(np.uint8)
                    else:
                        arr = np.zeros_like(arr, dtype=np.uint8)
                    img_to_display = Image.fromarray(arr, mode='L').convert('RGB')
        else:
            img_to_display = Image.open(image_path).convert('RGB')
            
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(img_to_display)
        
        status = "FLOOD DETECTED" if prediction_label == 1 else "NO FLOOD DETECTED"
        color = 'red' if prediction_label == 1 else 'green'
        
        title = f"{status}\nProbability: {probability:.4f}"
        
        ax.set_title(title, color=color, fontsize=16, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"Error in visualization: {e}")
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.text(0.5, 0.5, f"Error loading image for visualization: {e}", ha='center', va='center')
        ax.axis('off')
        return fig

# ---------------------------------------------------
# --- Streamlit App Logic (from prototype.py) ---
# ---------------------------------------------------

st.set_page_config(
    page_title="FloodSentinel - AI-Powered Flood Risk Assessment",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Initialize session state
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
if 'df_flood' not in st.session_state:
    st.session_state.df_flood = None
if 'sat_files' not in st.session_state:
    st.session_state.sat_files = []
if 'dl_model' not in st.session_state:
    st.session_state.dl_model = None

st.markdown('<h1 class="main-header">🌊 FloodSentinel</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Flood Risk Assessment Using Multi-Temporal Satellite Imagery and Deep Neural Networks</p>', unsafe_allow_html=True)

st.sidebar.markdown("### 🧭 Navigation")
page = st.sidebar.selectbox(
    "Choose a section:",
    ["🏠 Home", "📊 Data Analysis", "⚙️ Model Training", "🔮 Predictions", "🛰️ Satellite Analysis", "📈 Results Dashboard", "📸 Image Detection"]
)
st.sidebar.markdown("---") 

st.sidebar.link_button(
    "💾 Load Deep Learning Model",
    "javascript:void(0)",
    type="primary",
    help="Loads the pre-trained PyTorch deep learning model for image detection."
)
if st.sidebar.button("Load Deep Learning Model", key="load_dl_model_btn"):
    with st.spinner("Loading Deep Learning Model..."): 
        st.session_state.dl_model = load_trained_model()
        st.success("Deep Learning Model Loaded (ResNet18 on " + str(DEVICE) + ")")

st.sidebar.link_button(
    "🌐 Ask the Sentinel Chatbot", 
    "https://flood-app-repo-chatbot-sck.streamlit.app/", 
    type="secondary", 
    help="Redirects to the complete Flood Risk Assessment System's AI Chatbot tab." 
)


@st.cache_resource
def load_datasets_actual():
    """Load datasets from Kaggle, including unzipping image data."""
    try:
        with st.spinner("🔄 Downloading datasets from Kaggle..."):
            path_tabular = kagglehub.dataset_download("naiyakhalid/flood-prediction-dataset")
            st.success(f"✅ Tabular data downloaded to: {path_tabular}")
            path_sat = kagglehub.dataset_download("rhythmroy/sen12flood-flood-detection-dataset")
            st.success(f"✅ Satellite imagery data downloaded to: {path_sat}")
            
            flood_files = [os.path.join(root, file) for root, dirs, files in os.walk(path_tabular) for file in files if file.endswith('.csv')]
            if flood_files:
                df_flood = pd.read_csv(flood_files[0])
                st.success(f"✅ Loaded flood prediction dataset with {len(df_flood)} records")
            else:
                st.error("❌ No CSV files found in flood prediction dataset")
                return None, []
                
            sat_files = []
            st.info("Unzipping satellite image data. This may take a while...")
            zip_paths = [os.path.join(root, file) for root, dirs, files in os.walk(path_sat) for file in files if file.endswith('.zip')]

            if not zip_paths:
                st.warning("⚠️ No zip files found. Assuming data is already unzipped.")
                for root, dirs, files in os.walk(path_sat):
                    for file in files:
                        if file.lower().endswith(('.tif', '.png', '.jpg', '.jpeg')):
                            sat_files.append(os.path.join(root, file))
            else:
                for zip_path in zip_paths:
                    try:
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            extract_dir = os.path.join(path_sat, os.path.basename(zip_path).replace('.zip', ''))
                            zip_ref.extractall(extract_dir)
                            st.success(f"✅ Unzipped: {os.path.basename(zip_path)}")
                            for root, _, files in os.walk(extract_dir):
                                for file in files:
                                    if file.lower().endswith(('.tif', '.png', '.jpg', '.jpeg')):
                                        sat_files.append(os.path.join(root, file))
                    except zipfile.BadZipFile:
                        st.warning(f"⚠️ Corrupted zip file: {zip_path}")
                    except Exception as e:
                        st.error(f"❌ Error unzipping {zip_path}: {str(e)}")

            if sat_files:
                st.success(f"✅ Found and processed {len(sat_files)} satellite images!")
            else:
                st.warning("⚠️ No satellite images were found after unzipping.")
            
            return df_flood, sat_files
            
    except Exception as e:
        st.error(f"❌ Error loading datasets: {str(e)}")
        return None, []

def get_model_algorithms():
    """Get all state-of-the-art algorithms"""
    return {
        "🌳 Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "🚀 XGBoost": xgb.XGBRegressor(random_state=42),
        "💡 LightGBM": lgb.LGBMRegressor(random_state=42),
        "🎯 CatBoost": CatBoostRegressor(verbose=False, random_state=42),
        "⚡ Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "🧠 Neural Network": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
        "📈 Support Vector": SVR(kernel='rbf'),
        "🔗 ElasticNet": ElasticNet(random_state=42),
        "🎪 AdaBoost": AdaBoostRegressor(random_state=42),
        "🌿 Decision Tree": DecisionTreeRegressor(random_state=42),
        "👥 K-Neighbors": KNeighborsRegressor(n_neighbors=5),
        "📊 Ridge Regression": Ridge(random_state=42),
    }

# --- Removed Keras-based Deep Learning Functions (Replaced by flood_dl_module) ---
# def preprocess_image(img_path, target_size=(128, 128)):
# def create_cnn_model(input_shape=(128, 128, 3)):

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
        'WetlandLoss': np.random.uniform(0.0, 1.0, n_samples),
        'InadequatePlanning': np.random.uniform(0.0, 1.0, n_samples),
        'Latitude': np.random.uniform(-90, 90, n_samples),
        'Longitude': np.random.uniform(-180, 180, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Simple formula to create a target variable (FloodProbability)
    df['FloodProbability'] = (
        0.2 * df['MonsoonIntensity'] + 
        0.15 * df['TopographyDrainage'] + 
        0.1 * df['Deforestation'] + 
        0.1 * df['Urbanization'] + 
        0.1 * df['ClimateChange'] + 
        0.05 * df['Siltation'] + 
        np.random.normal(0, 0.1, n_samples)
    )
    
    # Normalize and clip the probability
    min_prob, max_prob = df['FloodProbability'].min(), df['FloodProbability'].max()
    df['FloodProbability'] = (df['FloodProbability'] - min_prob) / (max_prob - min_prob)
    df['FloodProbability'] = np.clip(df['FloodProbability'], 0, 1)
    
    return df

if page == "📸 Image Detection":
    st.title("📸 Satellite Image Flood Detection")
    st.markdown("---")

    if st.session_state.dl_model is None:
        st.warning("Please load the Deep Learning Model from the sidebar first.")
    else:
        st.info(f"Deep Learning Model ({st.session_state.dl_model.__class__.__name__}) is loaded and ready on {DEVICE}.")
        
        st.subheader("Upload Image for Detection")
        uploaded_file = st.file_uploader(
            "Choose a satellite image file (TIFF, PNG, JPG, JPEG)", 
            type=['tif', 'tiff', 'png', 'jpg', 'jpeg']
        )

        if uploaded_file is not None:
            # Save the uploaded file temporarily
            temp_file_path = os.path.join("/tmp", uploaded_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
            st.write(f"File: {uploaded_file.name} (Size: {len(uploaded_file.getbuffer()) / 1024:.2f} KB)")

            if st.button("Run Flood Detection", type="primary"):
                with st.spinner("Analyzing image with Deep Learning Model..."): 
                    try:
                        # The predict_flood function handles TIFF and other formats
                        prediction_label, probability = predict_flood(st.session_state.dl_model, temp_file_path)
                        
                        st.subheader("Detection Results")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if prediction_label == 1:
                                st.error(f"🚨 FLOOD DETECTED! (Probability: {probability:.4f})")
                            else:
                                st.success(f"✅ NO FLOOD DETECTED (Probability: {probability:.4f})")
                            
                            st.metric(label="Flood Probability", value=f"{probability*100:.2f}%", delta_color="off")
                            st.metric(label="Model Used", value=st.session_state.dl_model.__class__.__name__)
                            st.metric(label="Image Size", value=f"{IMAGE_SIZE}x{IMAGE_SIZE} (Input to Model)")
                            
                        with col2:
                            # Generate and display the visualization
                            fig = visualize_flood_detection(temp_file_path, prediction_label, probability)
                            st.pyplot(fig)
                            
                    except Exception as e:
                        st.error(f"An error occurred during prediction: {e}")
                    finally:
                        # Clean up the temporary file
                        os.remove(temp_file_path)

    st.subheader("Satellite Data Analysis (Sample)")
    if st.session_state.dataset_loaded and st.session_state.sat_files:
        st.info(f"Found {len(st.session_state.sat_files)} satellite files in the dataset.")
        sample_files = st.session_state.sat_files[:5]
        st.markdown("##### Sample File Paths:")
        for f in sample_files:
            st.code(f, language='text')
    elif st.session_state.dataset_loaded:
        st.warning("Dataset loaded, but no satellite image files were found.")
    else:
        st.info("Load the dataset from the 'Home' page to see sample satellite file paths.")

if page == "🏠 Home":
    st.markdown("### 🎯 Project Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <h4>About FloodSentinel</h4>
        <p>FloodSentinel is an advanced AI-powered system designed for comprehensive flood risk assessment. It combines traditional machine learning models for predicting flood probability based on environmental factors with cutting-edge deep learning models for real-time flood detection from satellite imagery.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 💾 Data Loading")
        
        if st.button("📥 Load Datasets (Tabular & Satellite)", type="primary"):
            df_flood, sat_files = load_datasets_actual()
            if df_flood is not None:
                st.session_state.df_flood = df_flood
                st.session_state.sat_files = sat_files
                st.session_state.dataset_loaded = True
                st.success("Datasets loaded and ready for analysis!")
            else:
                st.session_state.dataset_loaded = False
                st.error("Failed to load datasets.")
        
        if st.session_state.dataset_loaded:
            st.success("✅ Datasets are loaded in session state.")
            st.dataframe(st.session_state.df_flood.head(), use_container_width=True)
        else:
            st.info("Click the button above to load the datasets from Kaggle.")
            
    with col2:
        st.markdown("""
        <div class="success-box">
        <h4>Key Features</h4>
        <ul>
            <li><strong>Tabular ML:</strong> Predict flood probability based on 10+ environmental factors.</li>
            <li><strong>Deep Learning:</strong> Detect flood presence from uploaded satellite images (TIFF, PNG, JPG).</li>
            <li><strong>Visualization:</strong> Interactive plots for data analysis and model results.</li>
            <li><strong>Modular Design:</strong> Separation of ML/DL logic for clean code.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 📊 Sample Data Creation")
        if st.button("✨ Create Sample Data (Fallback)", type="secondary"):
            df_flood = create_sample_data()
            st.session_state.df_flood = df_flood
            st.session_state.dataset_loaded = True
            st.session_state.sat_files = []
            st.success("Sample data created and loaded for demonstration.")
            st.dataframe(df_flood.head(), use_container_width=True)
            
    st.markdown("---")
    st.markdown('<p class="footer">Developed with Streamlit and Python for AI-Powered Flood Risk Assessment.</p>', unsafe_allow_html=True)

elif page == "📊 Data Analysis":
    st.markdown("### 📊 Exploratory Data Analysis (EDA)")
    
    if not st.session_state.dataset_loaded:
        st.warning("⚠️ Please load datasets first from the Home page")
        st.stop()
        
    df = st.session_state.df_flood
    
    st.markdown("#### 📈 Data Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Features", len(df.columns))
    col3.metric("Target Variable", "FloodProbability")
    
    st.markdown("#### 🗺️ Geographic Distribution")
    
    fig_map = px.scatter_geo(
        df,
        lat='Latitude',
        lon='Longitude',
        color='FloodProbability',
        hover_name=df.index,
        projection="natural earth",
        title="Geographic Distribution of Flood Risk Samples"
    )
    st.plotly_chart(fig_map, use_container_width=True)
    
    st.markdown("#### 📉 Feature Correlation")
    
    corr_matrix = df.corr()
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Feature Correlation Heatmap"
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown("#### 📊 Target Variable Distribution")
    
    fig_hist = px.histogram(
        df,
        x='FloodProbability',
        nbins=20,
        title="Distribution of Flood Probability"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

elif page == "⚙️ Model Training":
    st.markdown("### ⚙️ Machine Learning Model Training")
    
    if not st.session_state.dataset_loaded:
        st.warning("⚠️ Please load datasets first from the Home page")
        st.stop()
        
    df = st.session_state.df_flood
    
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
        
        random_state = st.number_input("🎲 Random State:", value=42)

    st.markdown("#### 🎯 Model Selection")
    
    models = get_model_algorithms()
    selected_models = st.multiselect(
        "Choose models to train:",
        list(models.keys()),
        default=list(models.keys())[:6]
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
            X, y, test_size=test_size, random_state=int(random_state)
        )
        
        if scaler_type == "StandardScaler":
            scaler = StandardScaler()
        elif scaler_type == "MinMaxScaler":
            scaler = MinMaxScaler()
        else:
            scaler = RobustScaler()
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        pca = PCA(n_components=min(10, X_train_scaled.shape[1]))
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        st.session_state.pca_components = pca.components_
        st.session_state.pca_feature_names = X.columns
        
        results = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, model_name in enumerate(selected_models):
            status_text.text(f"🔄 Training {model_name}...")
            
            model = models[model_name]
            
            start_time = time.time()
            try:
                if model_name in ["🌳 Random Forest", "🚀 XGBoost", "💡 LightGBM", "🎯 CatBoost", "⚡ Gradient Boosting", "🌿 Decision Tree", "🧠 Neural Network"]:
                    model.fit(X_train_pca, y_train)
                    y_pred = model.predict(X_test_pca)
                    cv_scores = cross_val_score(model, X_train_pca, y_train, cv=cv_folds, scoring="r2")
                else:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds, scoring="r2")

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
            
            status_text.text("✅ All models trained successfully!")
            st.success("🎉 Model training completed!")
        else:
            st.error("❌ No models were successfully trained.")

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

            input_scaled = st.session_state.scaler.transform(input_data)
            
            st.markdown("#### 🎯 Prediction Results")
            
            predictions = {}
            for model_name, model_info in st.session_state.model_results.items():
                try:
                    if model_name in ["🌳 Random Forest", "🚀 XGBoost", "💡 LightGBM", "🎯 CatBoost", "⚡ Gradient Boosting", "🌿 Decision Tree", "🧠 Neural Network"]:
                        pca_input = np.dot(input_scaled, st.session_state.pca_components.T)
                        pred = model_info['Model'].predict(pca_input)[0]
                    else:
                        pred = model_info['Model'].predict(input_scaled)[0]
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

elif page == "🛰️ Satellite Analysis":
    st.markdown("### 🛰️ Satellite Imagery Analysis: A Step-by-Step Guide")
    
    if not st.session_state.dataset_loaded:
        st.warning("⚠️ Please load datasets first from the Home page")
        st.stop()
    
    st.markdown("#### 1. Data Preprocessing & Visualization")
    st.markdown("""
    Satellite images contain data in multiple **spectral bands** beyond what the human eye can see (Red, Green, Blue). To make this raw data useful, we combine different bands to create informative images. The images below demonstrate this process:
    - **True Color:** A human-readable photo created by combining the Red, Green, and Blue bands.
    - **False Color:** A composite using different bands (like Near-Infrared) to highlight specific features like water.
    """)
    
    st.markdown("#### 2. Deep Learning Model Integration")
    st.markdown("""
    The deep learning component uses a pre-trained **ResNet18** model, adapted for binary classification (Flood/No Flood).
    - **Input:** Satellite image (TIFF, PNG, JPG) resized to 224x224 pixels.
    - **Processing:** The `preprocess_tif_image` function handles multi-band TIFF files, normalizing and converting them to a 3-channel image suitable for the model.
    - **Output:** A probability score (0 to 1) indicating the likelihood of flood presence.
    
    Go to the **📸 Image Detection** tab to test this feature with your own image.
    """)
    
    st.markdown("#### 3. Data Source")
    st.markdown("""
    The deep learning model is designed to work with data similar to the **SEN12FLOOD** dataset, which contains Sentinel-1 (SAR) and Sentinel-2 (Optical) imagery with flood labels. The current implementation uses a generic image processing pipeline to handle various image types, including the TIFF files found in this dataset.
    """)

elif page == "📈 Results Dashboard":
    st.markdown("### 📈 Model Results Dashboard")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first from the Model Training page")
        st.stop()
        
    results = st.session_state.model_results
    
    # Convert results to a DataFrame for easy display and plotting
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df = results_df.drop(columns=['Model', 'Predictions'])
    
    st.markdown("#### 🎯 Performance Metrics Comparison")
    st.dataframe(results_df.sort_values(by='R²', ascending=False), use_container_width=True)
    
    # Plotting R² Score
    fig_r2 = px.bar(
        results_df.sort_values(by='R²', ascending=False),
        y='R²',
        title='R² Score Comparison',
        color='R²',
        color_continuous_scale=px.colors.sequential.Viridis
    )
    st.plotly_chart(fig_r2, use_container_width=True)
    
    # Plotting RMSE
    fig_rmse = px.bar(
        results_df.sort_values(by='RMSE', ascending=True),
        y='RMSE',
        title='Root Mean Squared Error (RMSE) Comparison',
        color='RMSE',
        color_continuous_scale=px.colors.sequential.Reds_r
    )
    st.plotly_chart(fig_rmse, use_container_width=True)
    
    # Plotting Training Time
    fig_time = px.bar(
        results_df.sort_values(by='Training_Time', ascending=True),
        y='Training_Time',
        title='Training Time Comparison (Seconds)',
        color='Training_Time',
        color_continuous_scale=px.colors.sequential.Plasma
    )
    st.plotly_chart(fig_time, use_container_width=True)
    
    st.markdown("#### 📊 Cross-Validation Results")
    
    fig_cv = px.bar(
        results_df.sort_values(by='CV_Mean', ascending=False),
        y='CV_Mean',
        error_y='CV_Std',
        title='Cross-Validation Mean R² with Standard Deviation',
        color='CV_Mean',
        color_continuous_scale=px.colors.sequential.Greens
    )
    st.plotly_chart(fig_cv, use_container_width=True)
