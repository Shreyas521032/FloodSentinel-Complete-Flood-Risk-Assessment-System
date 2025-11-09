import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import kagglehub
import warnings
import os
import zipfile
import cv2
from PIL import Image
import tensorflow as tf
import time
import json
import io
import pickle
import gzip
import torch
import torch.nn as nn
from torchvision import models as torch_models
import timm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import glob
from tqdm import tqdm

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
if 'cnn_models_trained' not in st.session_state:
    st.session_state.cnn_models_trained = False
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'train_images' not in st.session_state:
    st.session_state.train_images = []
if 'test_images' not in st.session_state:
    st.session_state.test_images = []

# ==================== CUSTOM DATASET CLASS ====================

class FloodDataset(Dataset):
    """Custom Dataset for Flood Detection"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            # Return a blank image if loading fails
            blank = torch.zeros(3, 224, 224)
            return blank, label

# ==================== IMAGE PREPROCESSING & TRAINING FUNCTIONS ====================

def load_and_prepare_satellite_data(sat_path, sample_size=1000, test_split=0.2):
    """Load and prepare satellite imagery data for training"""
    
    st.info(f"🔍 Scanning for flood images in: {sat_path}")
    
    # Find all image files
    flood_images = []
    no_flood_images = []
    
    # Search for images in subdirectories
    for root, dirs, files in os.walk(sat_path):
        for file in files:
            if file.lower().endswith(('.tif', '.png', '.jpg', '.jpeg')):
                full_path = os.path.join(root, file)
                
                # Determine label based on directory structure or filename
                if 'flood' in root.lower() or 'flood' in file.lower():
                    if 'non' not in root.lower() and 'no' not in root.lower():
                        flood_images.append(full_path)
                    else:
                        no_flood_images.append(full_path)
                elif 'non' in root.lower() or 'no' in root.lower():
                    no_flood_images.append(full_path)
                else:
                    # If unclear, analyze the image path more carefully
                    path_lower = full_path.lower()
                    if any(x in path_lower for x in ['flood', 'inundated', 'water']):
                        flood_images.append(full_path)
                    else:
                        no_flood_images.append(full_path)
    
    st.info(f"📊 Found {len(flood_images)} flood images and {len(no_flood_images)} non-flood images")
    
    # Sample images
    sample_per_class = sample_size // 2
    
    if len(flood_images) > sample_per_class:
        flood_sample = np.random.choice(flood_images, sample_per_class, replace=False).tolist()
    else:
        flood_sample = flood_images
        st.warning(f"⚠️ Only {len(flood_images)} flood images available")
    
    if len(no_flood_images) > sample_per_class:
        no_flood_sample = np.random.choice(no_flood_images, sample_per_class, replace=False).tolist()
    else:
        no_flood_sample = no_flood_images
        st.warning(f"⚠️ Only {len(no_flood_images)} non-flood images available")
    
    # Combine samples
    all_images = flood_sample + no_flood_sample
    all_labels = [1] * len(flood_sample) + [0] * len(no_flood_sample)
    
    # Shuffle
    combined = list(zip(all_images, all_labels))
    np.random.shuffle(combined)
    all_images, all_labels = zip(*combined)
    
    # Split into train/test
    split_idx = int(len(all_images) * (1 - test_split))
    
    train_images = list(all_images[:split_idx])
    train_labels = list(all_labels[:split_idx])
    
    test_images = list(all_images[split_idx:])
    test_labels = list(all_labels[split_idx:])
    
    st.success(f"✅ Prepared {len(train_images)} training and {len(test_images)} test images")
    
    return train_images, train_labels, test_images, test_labels

def get_transforms():
    """Get image transforms for training and testing"""
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform

def train_cnn_model(model, train_loader, test_loader, model_name, epochs=5, device='cuda'):
    """Train a CNN model"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    model = model.to(device)
    
    best_acc = 0.0
    train_losses = []
    train_accs = []
    val_accs = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * correct / total
        val_accs.append(val_acc)
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Update progress
        progress_bar.progress((epoch + 1) / epochs)
        status_text.text(f"{model_name} - Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
    
    return model, {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_acc': best_acc
    }

def extract_cnn_features(models_dict, data_loader, device='cuda'):
    """Extract features from all CNN models"""
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            
            batch_features = []
            
            for model_name in ['resnet', 'densenet', 'vit', 'efficientnet']:
                if model_name in models_dict:
                    model = models_dict[model_name]
                    model.eval()
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of flood class
                    batch_features.append(probs.cpu().numpy())
            
            if len(batch_features) == 4:  # All 4 models available
                batch_features = np.column_stack(batch_features)
                all_features.append(batch_features)
                all_labels.extend(labels.numpy())
    
    if all_features:
        return np.vstack(all_features), np.array(all_labels)
    else:
        return None, None

# ==================== PREDICTION FUNCTIONS ====================

def preprocess_for_flood_detection(image, target_size=(224, 224)):
    """Enhanced preprocessing specifically for flood detection"""
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
        return None

def detect_water_features(image):
    """Enhanced water detection with multiple color spaces"""
    img_array = np.array(image)
    
    # Convert to different color spaces
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # Water detection in HSV (dark areas with low saturation)
    lower_water_hsv = np.array([0, 0, 0])
    upper_water_hsv = np.array([180, 50, 100])
    water_mask_hsv = cv2.inRange(hsv, lower_water_hsv, upper_water_hsv)
    
    # Blue channel analysis (water is often blue/dark blue)
    b_channel = img_array[:, :, 2]
    _, water_mask_blue = cv2.threshold(b_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Combine masks
    water_mask = cv2.bitwise_or(water_mask_hsv, water_mask_blue)
    
    # Clean up with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Calculate percentage
    water_percentage = (np.sum(water_mask > 0) / water_mask.size) * 100
    
    return water_mask, water_percentage

def detect_fire_features(image):
    """Detect fire/orange/red features that might be confused with flood"""
    img_array = np.array(image)
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # Fire typically has red/orange/yellow hues
    lower_fire1 = np.array([0, 100, 100])
    upper_fire1 = np.array([10, 255, 255])
    
    lower_fire2 = np.array([160, 100, 100])
    upper_fire2 = np.array([180, 255, 255])
    
    lower_fire3 = np.array([10, 100, 100])
    upper_fire3 = np.array([30, 255, 255])
    
    fire_mask1 = cv2.inRange(hsv, lower_fire1, upper_fire1)
    fire_mask2 = cv2.inRange(hsv, lower_fire2, upper_fire2)
    fire_mask3 = cv2.inRange(hsv, lower_fire3, upper_fire3)
    
    fire_mask = cv2.bitwise_or(fire_mask1, fire_mask2)
    fire_mask = cv2.bitwise_or(fire_mask, fire_mask3)
    
    fire_percentage = (np.sum(fire_mask > 0) / fire_mask.size) * 100
    
    return fire_mask, fire_percentage

def detect_vegetation(image):
    """Detect green vegetation"""
    img_array = np.array(image)
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # Green vegetation detection
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    veg_mask = cv2.inRange(hsv, lower_green, upper_green)
    veg_percentage = (np.sum(veg_mask > 0) / veg_mask.size) * 100
    
    return veg_mask, veg_percentage

def analyze_image_context(image):
    """Comprehensive image analysis to determine if it's actually flood-related"""
    water_mask, water_pct = detect_water_features(image)
    fire_mask, fire_pct = detect_fire_features(image)
    veg_mask, veg_pct = detect_vegetation(image)
    
    # Calculate image statistics
    img_array = np.array(image)
    mean_brightness = np.mean(img_array)
    
    # Decision logic
    is_likely_flood = False
    confidence = 0.0
    reason = ""
    
    if fire_pct > 15:
        is_likely_flood = False
        confidence = 0.95
        reason = f"High fire/heat signature detected ({fire_pct:.1f}%). Image appears to show fire, not flood."
    elif veg_pct > 60 and water_pct < 10:
        is_likely_flood = False
        confidence = 0.85
        reason = f"Dense vegetation detected ({veg_pct:.1f}%), minimal water. Not a flood scenario."
    elif water_pct > 20 and fire_pct < 5:
        is_likely_flood = True
        confidence = 0.80
        reason = f"Significant water coverage detected ({water_pct:.1f}%). Likely flood scenario."
    elif water_pct > 10 and mean_brightness < 100:
        is_likely_flood = True
        confidence = 0.65
        reason = f"Dark areas with water detected. Possible flood scenario."
    else:
        is_likely_flood = False
        confidence = 0.70
        reason = f"Insufficient flood indicators. Water: {water_pct:.1f}%, Fire: {fire_pct:.1f}%, Vegetation: {veg_pct:.1f}%"
    
    return {
        'is_likely_flood': is_likely_flood,
        'confidence': confidence,
        'reason': reason,
        'water_percentage': water_pct,
        'fire_percentage': fire_pct,
        'vegetation_percentage': veg_pct,
        'mean_brightness': mean_brightness
    }

def predict_with_ensemble(image, models_dict):
    """Make prediction using ensemble of models with context analysis"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # First, analyze image context
    context = analyze_image_context(image)
    
    # If clearly not a flood scenario, return early
    if not context['is_likely_flood'] and context['confidence'] > 0.85:
        return {
            'predictions': {},
            'ensemble_pred': 0.0,
            'context': context,
            'rejected': True
        }
    
    predictions = {}
    cnn_features = []
    
    # Preprocess image
    _, test_transform = get_transforms()
    img_tensor = test_transform(image).unsqueeze(0).to(device)
    
    # Get predictions from each CNN model
    with torch.no_grad():
        for model_name in ['resnet', 'densenet', 'vit', 'efficientnet']:
            if model_name in models_dict:
                try:
                    model = models_dict[model_name]
                    model.eval()
                    output = model(img_tensor)
                    prob = torch.softmax(output, dim=1)[0, 1].cpu().numpy()
                    predictions[model_name] = float(prob)
                    cnn_features.append(prob)
                except Exception as e:
                    st.warning(f"⚠️ Error with {model_name}: {str(e)}")
    
    # Try to use ensemble models if CNN features are available
    if len(cnn_features) == 4:  # All 4 CNNs loaded
        cnn_feature_vector = np.array(cnn_features).reshape(1, -1)
        
        # Try meta models
        for meta_name in ['meta_model', 'xgb_meta_model']:
            if meta_name in models_dict:
                try:
                    meta_pred = models_dict[meta_name].predict_proba(cnn_feature_vector)[0, 1]
                    predictions[f"{meta_name}_ensemble"] = float(meta_pred)
                except Exception as e:
                    pass
        
        # Try stacking models
        for stack_name in ['cnn_stacking_logistic', 'cnn_stacking_ensemble_xgb_model']:
            if stack_name in models_dict:
                try:
                    if hasattr(models_dict[stack_name], 'predict_proba'):
                        stack_pred = models_dict[stack_name].predict_proba(cnn_feature_vector)[0, 1]
                    else:
                        stack_pred = models_dict[stack_name].predict(cnn_feature_vector)[0]
                    predictions[f"{stack_name}_ensemble"] = float(stack_pred)
                except Exception as e:
                    pass
    
    # Calculate ensemble prediction
    if predictions:
        ensemble_pred = np.mean(list(predictions.values()))
        
        # Apply context-based adjustment
        if context['fire_percentage'] > 10:
            ensemble_pred *= 0.3
        elif context['vegetation_percentage'] > 50 and context['water_percentage'] < 15:
            ensemble_pred *= 0.5
        elif context['water_percentage'] > 25:
            ensemble_pred = min(ensemble_pred * 1.2, 1.0)
    else:
        ensemble_pred = 0.0
    
    return {
        'predictions': predictions,
        'ensemble_pred': ensemble_pred,
        'context': context,
        'rejected': False
    }

def create_false_color_composite(image):
    """Create false color composite"""
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
    """Extract water mask"""
    try:
        water_mask, _ = detect_water_features(image)
        water_mask_rgb = cv2.cvtColor(water_mask, cv2.COLOR_GRAY2RGB)
        water_mask_rgb[:, :, 0] = 0
        water_mask_rgb[:, :, 2] = 0
        
        return Image.fromarray(water_mask_rgb)
    except Exception as e:
        return image

# ==================== DATASET LOADING ====================

@st.cache_resource
def load_datasets_from_kaggle():
    """Load datasets from Kaggle"""
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
                return None, None
                
            return df_flood, path_sat
            
    except Exception as e:
        st.error(f"❌ Error loading datasets: {str(e)}")
        return None, None

def create_sample_data():
    """Create sample flood prediction data"""
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

def load_pretrained_tabular_models(models_dir="Saved_Model"):
    """Load all pre-trained tabular models with robust error handling"""
    loaded_models = {}
    
    if not os.path.exists(models_dir):
        st.warning(f"⚠️ Models directory '{models_dir}' not found.")
        return loaded_models
    
    model_files = {
        'linear_regression.pkl': '📈 Linear Regression',
        'ridge.pkl': '📊 Ridge',
        'lasso.pkl': '🔗 Lasso',
        'k_neighbors_regressor.pkl': '👥 K-Neighbors',
        'decision_tree_regressor.pkl': '🌿 Decision Tree',
        'xgboost_regressor.pkl': '🚀 XGBoost',
        'lightgbm_regressor.pkl': '💡 LightGBM',
        'catboost_regressor.pkl': '🎯 CatBoost',
        'support_vector_regressor.pkl': '📈 SVR',
    }
    
    for filename, display_name in model_files.items():
        filepath = os.path.join(models_dir, filename)
        
        if not os.path.exists(filepath):
            continue
        
        try:
            with open(filepath, 'rb') as f:
                try:
                    loaded_models[display_name] = pickle.load(f)
                    st.success(f"✅ {display_name} loaded")
                except:
                    try:
                        import joblib
                        loaded_models[display_name] = joblib.load(filepath)
                        st.success(f"✅ {display_name} loaded")
                    except:
                        st.warning(f"⚠️ Could not load {display_name}")
        except Exception as e:
            st.error(f"❌ Error loading {display_name}: {str(e)}")
    
    return loaded_models

def load_ensemble_models(models_dir="pretrained_models"):
    """Load pre-trained ensemble models"""
    loaded_models = {}
    
    if not os.path.exists(models_dir):
        st.warning(f"⚠️ Ensemble models directory '{models_dir}' not found.")
        return loaded_models
    
    ensemble_files = {
        'meta_model.pkl': 'Meta Model',
        'xgb_meta_model.pkl': 'XGBoost Meta Model',
        'cnn_stacking_logistic.pkl': 'CNN Stacking (Logistic)',
        'cnn_stacking_ensemble_xgb_model.pkl': 'CNN Stacking (XGBoost)',
    }
    
    for filename, display_name in ensemble_files.items():
        filepath = os.path.join(models_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    loaded_models[filename.replace('.pkl', '')] = pickle.load(f)
                st.success(f"✅ {display_name} loaded")
            except Exception as e:
                try:
                    import joblib
                    loaded_models[filename.replace('.pkl', '')] = joblib.load(filepath)
                    st.success(f"✅ {display_name} loaded (joblib)")
                except:
                    st.warning(f"⚠️ Could not load {display_name}")
    
    return loaded_models

# ==================== MAIN APPLICATION ====================

st.markdown('<h1 class="main-header">🌊 FloodSentinel</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Flood Risk Assessment Using Multi-Temporal Satellite Imagery and Deep Neural Networks</p>', unsafe_allow_html=True)

# ==================== SIDEBAR NAVIGATION ====================

st.sidebar.markdown("### 🧭 Navigation")
page = st.sidebar.selectbox(
    "Choose a section:",
    ["🏠 Home", "📊 Data Analysis", "⚙️ Model Training", "🤖 Train CNN Models",
     "🔮 Predictions", "🖼️ Image Flood Detection", "📈 Results Dashboard"]
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
            <p>Floods remain among the most destructive natural hazards globally, causing widespread loss of life, economic disruption, and environmental damage.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-box">
            <h4>🎯 Our Solution</h4>
            <p>FloodSentinel combines machine learning for historical tabular data with deep neural networks for satellite imagery analysis.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
            <h4>🚀 Key Features</h4>
            <ul>
                <li>⚙️ 9 Pre-trained ML algorithms</li>
                <li>🛰️ Train CNN models on satellite data</li>
                <li>📊 Context-aware flood detection</li>
                <li>🎯 Fire/vegetation filtering</li>
                <li>📈 Interactive visualizations</li>
                <li>🤖 Ensemble predictions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Dataset Loading")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Load from Kaggle", type="primary", key="load_kaggle"):
            df_flood, sat_path = load_datasets_from_kaggle()
            if df_flood is not None and sat_path is not None:
                st.session_state.df_flood = df_flood
                st.session_state.sat_path = sat_path
                st.session_state.dataset_loaded = True
                st.rerun()
    
    with col2:
        if st.button("📊 Use Sample Data", type="secondary", key="load_sample"):
            st.session_state.df_flood = create_sample_data()
            st.session_state.sat_path = None
            st.session_state.dataset_loaded = True
            st.success("✅ Sample dataset loaded successfully!")
            st.info("⚠️ CNN training not available with sample data")
            st.rerun()
    
    if st.session_state.dataset_loaded:
        st.success(f"✅ Dataset loaded with {len(st.session_state.df_flood)} records!")
        st.dataframe(st.session_state.df_flood.head(), use_container_width=True)
        
        if hasattr(st.session_state, 'sat_path') and st.session_state.sat_path:
            st.info(f"📂 Satellite images available at: {st.session_state.sat_path}")

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
        img_status = "✅ Available" if hasattr(st.session_state, 'sat_path') and st.session_state.sat_path else "❌ N/A"
        st.markdown(f"""
        <div class="metric-container">
            <h3>🛰️ Images</h3>
            <h2>{img_status}</h2>
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

# ==================== PAGE: MODEL TRAINING (TABULAR) ====================

elif page == "⚙️ Model Training":
    st.markdown("### ⚙️ Load Pre-trained Tabular Models")
    
    if not st.session_state.dataset_loaded:
        st.warning("⚠️ Please load datasets first from the Home page")
        st.stop()
    
    df = st.session_state.df_flood
    
    if "FloodProbability" not in df.columns:
        st.error("FloodProbability column not found in dataset")
        st.stop()
    
    st.markdown("#### 📁 Load Pre-trained Tabular Models")
    
    models_dir = st.text_input("Tabular models directory:", value="Saved_Model")
    
    if st.button("🔄 Load Pre-trained Models", type="primary", key="load_tabular"):
        with st.spinner("Loading pre-trained models..."):
            loaded_models = load_pretrained_tabular_models(models_dir)
            
            if loaded_models:
                st.session_state.model_results = {}
                
                # Prepare data for evaluation
                X = df.drop("FloodProbability", axis=1)
                y = df["FloodProbability"]
                
                categorical_cols = X.select_dtypes(include='object').columns
                if len(categorical_cols) > 0:
                    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Scale data
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Apply PCA
                pca = PCA(n_components=10)
                X_train_pca = pca.fit_transform(X_train_scaled)
                X_test_pca = pca.transform(X_test_scaled)
                
                st.session_state.pca_model = pca
                st.session_state.scaler = scaler
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.X_test_pca = X_test_pca
                
                # Evaluate each model
                progress_bar = st.progress(0)
                for i, (model_name, model) in enumerate(loaded_models.items()):
                    try:
                        y_pred = model.predict(X_test_pca)
                        
                        # Regression metrics
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        
                        # Classification metrics (threshold at 0.5)
                        y_pred_class = (y_pred >= 0.5).astype(int)
                        y_test_class = (y_test >= 0.5).astype(int)
                        
                        accuracy = accuracy_score(y_test_class, y_pred_class)
                        precision = precision_score(y_test_class, y_pred_class, zero_division=0)
                        recall = recall_score(y_test_class, y_pred_class, zero_division=0)
                        f1 = f1_score(y_test_class, y_pred_class, zero_division=0)
                        
                        st.session_state.model_results[model_name] = {
                            "MSE": mse,
                            "RMSE": rmse,
                            "MAE": mae,
                            "R²": r2,
                            "Accuracy": accuracy,
                            "Precision": precision,
                            "Recall": recall,
                            "F1_Score": f1,
                            "CV_Mean": r2,
                            "CV_Std": 0.0,
                            "Training_Time": 0.0,
                            "Model": model,
                            "Predictions": y_pred
                        }
                    except Exception as e:
                        st.warning(f"⚠️ Could not evaluate {model_name}: {str(e)}")
                    
                    progress_bar.progress((i + 1) / len(loaded_models))
                
                st.session_state.models_trained = True
                st.success(f"✅ Loaded and evaluated {len(st.session_state.model_results)} models!")
                st.info(f"📊 PCA applied: Explained variance = {pca.explained_variance_ratio_.sum():.2%}")
            else:
                st.error("❌ No models were loaded. Check the directory path.")
    
    if st.session_state.models_trained:
        st.markdown("#### 📊 Loaded Models Summary")
        
        model_names = list(st.session_state.model_results.keys())
        st.success(f"✅ {len(model_names)} tabular models ready")
        
        for name in model_names:
            acc = st.session_state.model_results[name]['Accuracy']
            if acc > 0.85:
                st.markdown(f"🟢 {name}: Accuracy = {acc:.2%}")
            elif acc > 0.70:
                st.markdown(f"🟡 {name}: Accuracy = {acc:.2%}")
            else:
                st.markdown(f"🟠 {name}: Accuracy = {acc:.2%}")

# ==================== PAGE: TRAIN CNN MODELS ====================

elif page == "🤖 Train CNN Models":
    st.markdown("### 🤖 Train Deep Learning Models on Satellite Imagery")
    
    if not st.session_state.dataset_loaded:
        st.warning("⚠️ Please load datasets first from the Home page")
        st.stop()
    
    if not hasattr(st.session_state, 'sat_path') or st.session_state.sat_path is None:
        st.error("❌ Satellite imagery path not available")
        st.info("Please load datasets from Kaggle on the Home page")
        st.stop()
    
    st.markdown("""
    This section trains 4 CNN architectures on a sample of the Sen12Flood dataset:
    - **ResNet-50**: Deep residual network
    - **DenseNet-121**: Densely connected network
    - **Vision Transformer (ViT)**: Attention-based model
    - **EfficientNet-B0**: Efficient convolutional network
    
    After training, pre-trained ensemble models are loaded to combine CNN predictions.
    """)
    
    # Training configuration
    st.markdown("#### ⚙️ Training Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sample_size = st.number_input("Sample Size (images)", min_value=100, max_value=5000, value=1000, step=100)
    
    with col2:
        epochs = st.number_input("Training Epochs", min_value=1, max_value=20, value=5, step=1)
    
    with col3:
        batch_size = st.number_input("Batch Size", min_value=8, max_value=64, value=16, step=8)
    
    test_split = st.slider("Test Split Ratio", min_value=0.1, max_value=0.3, value=0.2, step=0.05)
    
    if st.button("🚀 Start Training", type="primary", key="start_training"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.info(f"🖥️ Using device: {device}")
        
        # Step 1: Load and prepare data
        st.markdown("#### 📂 Step 1: Loading Satellite Data")
        try:
            train_images, train_labels, test_images, test_labels = load_and_prepare_satellite_data(
                st.session_state.sat_path, 
                sample_size=sample_size, 
                test_split=test_split
            )
            
            st.session_state.train_images = train_images
            st.session_state.train_labels = train_labels
            st.session_state.test_images = test_images
            st.session_state.test_labels = test_labels
            
        except Exception as e:
            st.error(f"❌ Error loading satellite data: {str(e)}")
            st.stop()
        
        # Step 2: Create data loaders
        st.markdown("#### 🔄 Step 2: Creating Data Loaders")
        try:
            train_transform, test_transform = get_transforms()
            
            train_dataset = FloodDataset(train_images, train_labels, transform=train_transform)
            test_dataset = FloodDataset(test_images, test_labels, transform=test_transform)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            
            st.success(f"✅ Created data loaders: {len(train_dataset)} train, {len(test_dataset)} test")
            
        except Exception as e:
            st.error(f"❌ Error creating data loaders: {str(e)}")
            st.stop()
        
        # Step 3: Train models
        st.markdown("#### 🤖 Step 3: Training CNN Models")
        
        trained_models = {}
        training_histories = {}
        
        # Train ResNet-50
        st.markdown("##### 🔄 Training ResNet-50...")
        try:
            resnet = torch_models.resnet50(pretrained=True)
            resnet.fc = nn.Linear(resnet.fc.in_features, 2)
            resnet_trained, resnet_history = train_cnn_model(resnet, train_loader, test_loader, "ResNet-50", epochs=epochs, device=device)
            trained_models['resnet'] = resnet_trained
            training_histories['ResNet-50'] = resnet_history
        except Exception as e:
            st.error(f"❌ Error training ResNet: {str(e)}")
        
        # Train DenseNet-121
        st.markdown("##### 🔄 Training DenseNet-121...")
        try:
            densenet = torch_models.densenet121(pretrained=True)
            densenet.classifier = nn.Linear(densenet.classifier.in_features, 2)
            densenet_trained, densenet_history = train_cnn_model(densenet, train_loader, test_loader, "DenseNet-121", epochs=epochs, device=device)
            trained_models['densenet'] = densenet_trained
            training_histories['DenseNet-121'] = densenet_history
        except Exception as e:
            st.error(f"❌ Error training DenseNet: {str(e)}")
        
        # Train Vision Transformer
        st.markdown("##### 🔄 Training Vision Transformer...")
        try:
            vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
            vit_trained, vit_history = train_cnn_model(vit, train_loader, test_loader, "ViT", epochs=epochs, device=device)
            trained_models['vit'] = vit_trained
            training_histories['ViT'] = vit_history
        except Exception as e:
            st.error(f"❌ Error training ViT: {str(e)}")
        
        # Train EfficientNet-B0
        st.markdown("##### 🔄 Training EfficientNet-B0...")
        try:
            efficientnet = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)
            efficientnet_trained, efficientnet_history = train_cnn_model(efficientnet, train_loader, test_loader, "EfficientNet-B0", epochs=epochs, device=device)
            trained_models['efficientnet'] = efficientnet_trained
            training_histories['EfficientNet-B0'] = efficientnet_history
        except Exception as e:
            st.error(f"❌ Error training EfficientNet: {str(e)}")
        
        # Save trained models
        st.session_state.ensemble_models.update(trained_models)
        st.session_state.cnn_models_trained = True
        
        st.success(f"✅ Successfully trained {len(trained_models)} CNN models!")
        
        # Display training results
        st.markdown("#### 📊 Training Results")
        
        results_data = []
        for model_name, history in training_histories.items():
            results_data.append({
                'Model': model_name,
                'Best Val Accuracy': f"{history['best_acc']:.2f}%",
                'Final Train Accuracy': f"{history['train_accs'][-1]:.2f}%",
                'Final Val Accuracy': f"{history['val_accs'][-1]:.2f}%"
            })
        
        st.dataframe(pd.DataFrame(results_data), use_container_width=True)
        
        # Plot training curves
        fig = go.Figure()
        
        for model_name, history in training_histories.items():
            fig.add_trace(go.Scatter(
                x=list(range(1, len(history['val_accs']) + 1)),
                y=history['val_accs'],
                mode='lines+markers',
                name=model_name
            ))
        
        fig.update_layout(
            title="📈 Validation Accuracy During Training",
            xaxis_title="Epoch",
            yaxis_title="Accuracy (%)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Step 4: Load ensemble models
        st.markdown("#### 🔗 Step 4: Loading Ensemble Models")
        
        ensemble_dir = st.text_input("Ensemble models directory:", value="pretrained_models", key="ensemble_dir")
        
        ensemble_loaded = load_ensemble_models(ensemble_dir)
        st.session_state.ensemble_models.update(ensemble_loaded)
        
        if ensemble_loaded:
            st.success(f"✅ Loaded {len(ensemble_loaded)} ensemble models!")
            st.info("🎉 Training complete! You can now use Image Flood Detection.")
        else:
            st.warning("⚠️ No ensemble models loaded. CNN predictions only.")
    
    # Display training status
    if st.session_state.cnn_models_trained:
        st.markdown("---")
        st.markdown("#### ✅ Training Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            cnn_count = len([k for k in st.session_state.ensemble_models.keys() if k in ['resnet', 'densenet', 'vit', 'efficientnet']])
            st.success(f"🤖 {cnn_count}/4 CNN models trained")
        
        with col2:
            ensemble_count = len([k for k in st.session_state.ensemble_models.keys() if k not in ['resnet', 'densenet', 'vit', 'efficientnet']])
            st.info(f"🔗 {ensemble_count} ensemble models loaded")

# ==================== PAGE: PREDICTIONS ====================

elif page == "🔮 Predictions":
    st.markdown("### 🔮 Flood Risk Predictions")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please load tabular models first from the Model Training page")
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

            # Scale and PCA transform
            input_scaled = st.session_state.scaler.transform(input_data)
            input_pca = st.session_state.pca_model.transform(input_scaled)
            
            st.markdown("#### 🎯 Prediction Results")
            
            predictions = {}
            for model_name, model_info in st.session_state.model_results.items():
                try:
                    pred = model_info['Model'].predict(input_pca)[0]
                    predictions[model_name] = pred
                except Exception as e:
                    st.error(f"Error with {model_name}: {str(e)}")
            
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

# ==================== PAGE: IMAGE FLOOD DETECTION ====================

elif page == "🖼️ Image Flood Detection":
    st.markdown("### 🖼️ Advanced Flood Detection")
    
    if not st.session_state.cnn_models_trained:
        st.warning("⚠️ CNN models not trained yet")
        st.info("✅ **You can still use context-aware analysis!**")
        st.markdown("""
        The app includes intelligent context analysis that works without deep learning models:
        - 🔥 **Fire detection** - Identifies fire/heat signatures
        - 🌿 **Vegetation analysis** - Detects green vegetation
        - 💧 **Water detection** - Identifies water bodies
        - 📊 **Smart classification** - Rule-based flood assessment
        
        To use full deep learning predictions, go to "🤖 Train CNN Models" first.
        """)
    else:
        cnn_count = len([k for k in st.session_state.ensemble_models.keys() if k in ['resnet', 'densenet', 'vit', 'efficientnet']])
        ensemble_count = len([k for k in st.session_state.ensemble_models.keys() if k not in ['resnet', 'densenet', 'vit', 'efficientnet']])
        st.success(f"✅ {cnn_count} CNN models and {ensemble_count} ensemble models ready!")
    
    st.markdown("""
    Upload a satellite or aerial image for flood detection. The system includes:
    - 🔥 Fire detection to avoid false positives
    - 🌿 Vegetation analysis
    - 💧 Water body detection
    - 🤖 Deep learning ensemble predictions (if models trained)
    """)
    
    uploaded_file = st.file_uploader(
        "📤 Upload Image (JPG, PNG, or TIFF)",
        type=['jpg', 'jpeg', 'png', 'tif', 'tiff'],
        help="Upload a satellite or aerial image"
    )
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            
            st.markdown("#### 🖼️ Image Analysis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.image(image, caption="Original Image", use_container_width=True)
            
            with col2:
                false_color = create_false_color_composite(image)
                st.image(false_color, caption="False Color", use_container_width=True)
            
            with col3:
                water_mask = extract_water_mask(image)
                st.image(water_mask, caption="Water Mask", use_container_width=True)
            
            st.markdown("---")
            
            # Context Analysis
            st.markdown("#### 🔍 Context Analysis")
            context = analyze_image_context(image)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("💧 Water Coverage", f"{context['water_percentage']:.1f}%")
            with col2:
                st.metric("🔥 Fire Signature", f"{context['fire_percentage']:.1f}%")
            with col3:
                st.metric("🌿 Vegetation", f"{context['vegetation_percentage']:.1f}%")
            
            # Deep Learning Analysis
            if st.session_state.cnn_models_trained:
                st.markdown("#### 🤖 Deep Learning Analysis")
                
                with st.spinner("Running ensemble predictions..."):
                    result = predict_with_ensemble(image, st.session_state.ensemble_models)
                
                if result and not result['rejected']:
                    predictions = result['predictions']
                    ensemble_pred = result['ensemble_pred']
                    
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
                            <h3>🎯 Flood Probability</h3>
                            <h1>{ensemble_pred:.1%}</h1>
                            <h4>{risk_level}</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("##### 📊 All Model Predictions")
                        
                        # Separate CNN and Ensemble predictions
                        cnn_preds = {k: v for k, v in predictions.items() if k in ['resnet', 'densenet', 'vit', 'efficientnet']}
                        ensemble_preds = {k: v for k, v in predictions.items() if k not in ['resnet', 'densenet', 'vit', 'efficientnet']}
                        
                        if cnn_preds:
                            st.markdown("**🤖 CNN Models:**")
                            for model_name, pred in cnn_preds.items():
                                st.metric(model_name.upper(), f"{pred:.2%}")
                        
                        if ensemble_preds:
                            st.markdown("**🔗 Ensemble Models:**")
                            for model_name, pred in ensemble_preds.items():
                                display_name = model_name.replace('_ensemble', '').replace('_', ' ').title()
                                st.metric(display_name, f"{pred:.2%}")
                    
                    with col2:
                        # Gauge chart
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=ensemble_pred * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Flood Risk"},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': risk_color},
                                'steps': [
                                    {'range': [0, 30], 'color': "lightgreen"},
                                    {'range': [30, 70], 'color': "lightyellow"},
                                    {'range': [70, 100], 'color': "lightcoral"}
                                ],
                            }
                        ))
                        st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    # Model comparison visualization
                    if len(predictions) > 1:
                        st.markdown("#### 📊 Model Comparison")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            model_names = list(predictions.keys())
                            model_preds = [predictions[m] * 100 for m in model_names]
                            
                            colors = []
                            for name in model_names:
                                if name in ['resnet', 'densenet', 'vit', 'efficientnet']:
                                    colors.append('CNN')
                                else:
                                    colors.append('Ensemble')
                            
                            fig_compare = px.bar(
                                x=model_names,
                                y=model_preds,
                                color=colors,
                                title="🔍 All Model Predictions",
                                labels={'x': 'Model', 'y': 'Flood Probability (%)'},
                                color_discrete_map={'CNN': '#667eea', 'Ensemble': '#fa709a'}
                            )
                            fig_compare.add_hline(y=ensemble_pred * 100, line_dash="dash", 
                                                line_color="red", annotation_text="Final Ensemble",
                                                line_width=2)
                            fig_compare.update_layout(xaxis_tickangle=-45)
                            st.plotly_chart(fig_compare, use_container_width=True)
                        
                        with col2:
                            if len(predictions) >= 3:
                                fig_radar = go.Figure()
                                
                                fig_radar.add_trace(go.Scatterpolar(
                                    r=model_preds,
                                    theta=[name[:15] for name in model_names],
                                    fill='toself',
                                    name='Predictions',
                                    line_color='rgb(102, 126, 234)'
                                ))
                                
                                fig_radar.update_layout(
                                    polar=dict(
                                        radialaxis=dict(visible=True, range=[0, 100])
                                    ),
                                    title="🎯 Model Consensus View",
                                    showlegend=False
                                )
                                st.plotly_chart(fig_radar, use_container_width=True)
                    
                    st.info(f"💡 Analysis: {context['reason']}")
                    
                elif result and result['rejected']:
                    st.error("❌ Image Rejected - Not a Flood Scenario")
                    st.warning(f"🔍 Reason: {context['reason']}")
                    st.info("This image appears to show fire, vegetation, or other non-flood scenarios.")
                
                else:
                    st.error("❌ Error in ensemble prediction")
            
            else:
                st.markdown("#### 📊 Context-Based Assessment")
                
                if context['is_likely_flood']:
                    st.success(f"✅ Likely Flood Scenario (Confidence: {context['confidence']:.0%})")
                else:
                    st.error(f"❌ Not a Flood Scenario (Confidence: {context['confidence']:.0%})")
                
                st.info(f"💡 {context['reason']}")
                
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    else:
        st.info("👆 Upload an image to begin analysis")

# ==================== PAGE: RESULTS DASHBOARD ====================

elif page == "📈 Results Dashboard":
    st.markdown("### 📈 Results Dashboard")
    
    # Tabular models results
    if st.session_state.models_trained:
        st.markdown("#### 🏆 Tabular Model Performance")
        
        results = st.session_state.model_results
        
        perf_data = []
        for model_name, metrics in results.items():
            perf_data.append({
                'Model': model_name,
                'Accuracy': metrics['Accuracy'],
                'Precision': metrics['Precision'],
                'Recall': metrics['Recall'],
                'F1 Score': metrics['F1_Score'],
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
            })
        
        perf_df = pd.DataFrame(perf_data)
        perf_df = perf_df.sort_values('Accuracy', ascending=False)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if len(perf_df) > 0:
                best = perf_df.iloc[0]
                st.markdown(f"""<div class="metric-container"><h4>🥇 Best Model</h4><h3>{best['Model']}</h3><p>Accuracy: {best['Accuracy']:.2%}</p></div>""", unsafe_allow_html=True)
        
        with col2:
            avg_acc = perf_df['Accuracy'].mean()
            st.markdown(f"""<div class="metric-container"><h4>📊 Average Accuracy</h4><h2>{avg_acc:.2%}</h2></div>""", unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""<div class="metric-container"><h4>🎯 Models</h4><h2>{len(perf_df)}</h2></div>""", unsafe_allow_html=True)
        
        st.dataframe(perf_df.style.format({
            'Accuracy': '{:.2%}',
            'Precision': '{:.2%}',
            'Recall': '{:.2%}',
            'F1 Score': '{:.4f}',
            'RMSE': '{:.4f}',
            'MAE': '{:.4f}'
        }), use_container_width=True)
        
        # Comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig_acc = px.bar(
                perf_df.sort_values('Accuracy'),
                x='Accuracy',
                y='Model',
                orientation='h',
                title='🎯 Model Accuracy Comparison',
                color='Accuracy',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_acc, use_container_width=True)
        
        with col2:
            fig_f1 = px.bar(
                perf_df.sort_values('F1 Score'),
                x='F1 Score',
                y='Model',
                orientation='h',
                title='📊 F1 Score Comparison',
                color='F1 Score',
                color_continuous_scale='Plasma'
            )
            st.plotly_chart(fig_f1, use_container_width=True)
    else:
        st.warning("⚠️ No tabular models loaded yet")
    
    # CNN models results
    if st.session_state.cnn_models_trained:
        st.markdown("---")
        st.markdown("#### 🤖 CNN Model Performance")
        
        cnn_models = {k: v for k, v in st.session_state.ensemble_models.items() if k in ['resnet', 'densenet', 'vit', 'efficientnet']}
        
        if cnn_models and hasattr(st.session_state, 'test_images'):
            st.success(f"✅ {len(cnn_models)} CNN models trained")
            
            # Test on a few random images
            st.markdown("##### 🔍 Sample Predictions")
            
            if len(st.session_state.test_images) > 0:
                num_samples = min(5, len(st.session_state.test_images))
                sample_indices = np.random.choice(len(st.session_state.test_images), num_samples, replace=False)
                
                for idx in sample_indices:
                    img_path = st.session_state.test_images[idx]
                    true_label = st.session_state.test_labels[idx]
                    
                    try:
                        img = Image.open(img_path).convert('RGB')
                        
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.image(img, caption=f"True Label: {'Flood' if true_label == 1 else 'No Flood'}", use_container_width=True)
                        
                        with col2:
                            result = predict_with_ensemble(img, st.session_state.ensemble_models)
                            
                            if result and not result['rejected']:
                                pred = result['ensemble_pred']
                                
                                st.markdown(f"""
                                **Prediction:** {pred:.2%} {'🔴 Flood' if pred > 0.5 else '🟢 No Flood'}
                                
                                **Individual Models:**
                                """)
                                
                                for model_name, model_pred in result['predictions'].items():
                                    st.text(f"  • {model_name}: {model_pred:.2%}")
                            else:
                                st.warning("Prediction rejected by context analysis")
                    except Exception as e:
                        st.error(f"Error processing sample: {str(e)}")
                    
                    st.markdown("---")
        else:
            st.info("Train CNN models first to see performance metrics")
    else:
        st.info("💡 Train CNN models on the '🤖 Train CNN Models' page to see deep learning results")

# ==================== SIDEBAR STATUS ====================

st.sidebar.markdown("---")

if st.session_state.dataset_loaded:
    st.sidebar.success("✅ Dataset Loaded")
else:
    st.sidebar.error("❌ Dataset Not Loaded")

if st.session_state.models_trained:
    st.sidebar.success("✅ Tabular Models Loaded")
    st.sidebar.info(f"🎯 {len(st.session_state.model_results)} models")
else:
    st.sidebar.error("❌ Tabular Models Not Loaded")

if st.session_state.cnn_models_trained:
    cnn_count = len([k for k in st.session_state.ensemble_models.keys() if k in ['resnet', 'densenet', 'vit', 'efficientnet']])
    st.sidebar.success(f"✅ {cnn_count} CNN Models Trained")
else:
    st.sidebar.error("❌ CNN Models Not Trained")

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info("""
🌊 **FloodSentinel**
- 9 Pre-trained ML models
- Train 4 CNN architectures
- Context-aware detection
- Fire/vegetation filtering
- Ensemble predictions
""")

# ==================== FOOTER ====================

st.markdown("---")
st.markdown("""
    <div class="footer">
        <p>Crafted with ❤️ by Shreyas, Chinmay and Kaivalya.<br>
        Project: FloodSentinel</p>
    </div>
""", unsafe_allow_html=True)
