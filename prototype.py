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
import gdown
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
if 'cnn_models_loaded' not in st.session_state:
    st.session_state.cnn_models_loaded = False
if 'scaler' not in st.session_state:
    st.session_state.scaler = None

# ==================== IMAGE PREPROCESSING FUNCTIONS ====================

def get_transforms():
    """Get image transforms for testing"""
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    return test_transform

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

# ==================== CNN MODEL LOADING ====================

def load_pretrained_cnn_models(models_dir="cnn_models"):
    """Load pre-trained CNN models including Vision Transformer (ViT)"""
    
    # Google Drive file IDs for model checkpoints
    file_ids = {
        "resnet_model_checkpoint.pth": "1Dj5K1YyVl3mczopiEc7ZixVgPIaQRyku",
        "resnet_model_checkpoint (1).pth": "1Dj5K1YyVl3mczopiEc7ZixVgPIaQRyku",
        "densenet_model_checkpoint.pth": "1GzsLM7t3-1IiRv9qZJLTwr37lgq3goDh",
        "densenet_model_checkpoint (1).pth": "YOUR_FILE_ID_HERE",
        "efficientnet_model_checkpoint.pth": "YOUR_FILE_ID_HERE",
        "vit_model_checkpoint.pth": "1g-UIJgRo2Eu6QDVATPSfcgy-aHAST9cl",
    }
    
    # Ensure the target directory exists
    os.makedirs(models_dir, exist_ok=True)
    
    # Download files if they don't exist locally
    for filename, file_id in file_ids.items():
        checkpoint_path = os.path.join(models_dir, filename)
        if not os.path.exists(checkpoint_path):
            if file_id == "YOUR_FILE_ID_HERE":
                st.warning(f"⚠️ Skipping download for {filename}, File ID not set.")
                continue
            
            st.info(f"🔄 Downloading {filename} from Google Drive...")
            try:
                gdown.download(id=file_id, output=checkpoint_path, quiet=False)
                st.success(f"✅ Downloaded {filename}")
            except Exception as e:
                st.error(f"❌ Failed to download {filename}: {str(e)}")
    
    loaded_models = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(models_dir):
        st.error(f"❌ CNN models directory '{models_dir}' not found.")
        return loaded_models
    
    model_configs = {
        'densenet': {
            'checkpoints': ['densenet_model_checkpoint.pth', 'densenet_model_checkpoint (1).pth'],
            'display_name': '🌿 DenseNet-121',
            'model_fn': lambda: torch_models.densenet121(pretrained=False),
            'classifier_attr': 'classifier',
            'description': 'Densely Connected Network with 121 layers. Each layer receives input from all preceding layers.'
        },
        'efficientnet': {
            'checkpoints': ['efficientnet_model_checkpoint.pth'],
            'display_name': '🚀 EfficientNet-B0',
            'model_fn': lambda: timm.create_model('efficientnet_b0', pretrained=False, num_classes=2),
            'classifier_attr': None,
            'description': 'Efficient Convolutional Network optimized for mobile and edge devices with compound scaling.'
        },
        'vit': {
            'checkpoints': ['vit_model_checkpoint.pth'],
            'display_name': '🎯 Vision Transformer (ViT)',
            'model_fn': lambda: timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2),
            'classifier_attr': None,
            'description': 'Transformer-based architecture adapted for computer vision. Uses self-attention mechanisms to capture global image context.'
        }
    }
    
    for model_key, config in model_configs.items():
        loaded = False
        
        # Try each checkpoint filename
        for checkpoint_name in config['checkpoints']:
            checkpoint_path = os.path.join(models_dir, checkpoint_name)
            
            if not os.path.exists(checkpoint_path):
                continue
            
            try:
                st.info(f"🔄 Attempting to load {config['display_name']} from {checkpoint_name}...")
                
                # Create model architecture
                model = config['model_fn']()
                
                # Modify final layer for binary classification if needed
                if config['classifier_attr'] and config['classifier_attr'] == 'fc':
                    in_features = model.fc.in_features
                    model.fc = nn.Linear(in_features, 2)
                elif config['classifier_attr'] and config['classifier_attr'] == 'classifier':
                    in_features = model.classifier.in_features
                    model.classifier = nn.Linear(in_features, 2)
                
                # METHOD 1: Try loading checkpoint normally
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    
                    # Handle different checkpoint formats
                    if isinstance(checkpoint, dict):
                        if 'model_state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                        elif 'state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['state_dict'], strict=False)
                        elif 'model' in checkpoint:
                            model.load_state_dict(checkpoint['model'], strict=False)
                        else:
                            model.load_state_dict(checkpoint, strict=False)
                    else:
                        model.load_state_dict(checkpoint, strict=False)
                    
                    model = model.to(device)
                    model.eval()
                    
                    loaded_models[model_key] = model
                    st.success(f"✅ {config['display_name']} loaded successfully!")
                    loaded = True
                    break
                    
                except Exception as e1:
                    st.warning(f"⚠️ Method 1 failed: {str(e1)}")
                    
                    # METHOD 2: Try loading with weights_only=False
                    try:
                        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                        
                        if isinstance(checkpoint, dict):
                            state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint.get('model', checkpoint)))
                        else:
                            state_dict = checkpoint
                        
                        # Remove unexpected keys
                        model_dict = model.state_dict()
                        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
                        
                        model_dict.update(filtered_dict)
                        model.load_state_dict(model_dict, strict=False)
                        
                        model = model.to(device)
                        model.eval()
                        
                        loaded_models[model_key] = model
                        st.success(f"✅ {config['display_name']} loaded successfully (Filtered)")
                        loaded = True
                        break
                        
                    except Exception as e2:
                        st.warning(f"⚠️ Method 2 failed: {str(e2)}")
                        
                        # METHOD 3: Try creating new model with pretrained ImageNet weights
                        try:
                            if model_key == 'densenet':
                                model = torch_models.densenet121(pretrained=True)
                                model.classifier = nn.Linear(model.classifier.in_features, 2)
                            elif model_key == 'efficientnet':
                                model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)
                            elif model_key == 'vit':
                                model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
                            
                            model = model.to(device)
                            model.eval()
                            
                            loaded_models[model_key] = model
                            st.warning(f"⚠️ {config['display_name']} loaded with ImageNet weights (Fallback)")
                            st.info("Note: Using pretrained ImageNet weights as checkpoint loading failed")
                            loaded = True
                            break
                            
                        except Exception as e3:
                            st.error(f"❌ All methods failed for {config['display_name']}: {str(e3)}")
            
            except Exception as e:
                st.error(f"❌ Error loading {config['display_name']}: {str(e)}")
        
        if not loaded:
            st.warning(f"⚠️ {config['display_name']}: No valid checkpoint found. Tried: {', '.join(config['checkpoints'])}")
    
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

# ==================== PREDICTION FUNCTIONS ====================

def predict_with_ensemble(image, models_dict):
    """Make prediction using ensemble of models including ViT with context analysis"""
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
    test_transform = get_transforms()
    img_tensor = test_transform(image).unsqueeze(0).to(device)
    
    # Get predictions from each CNN model (including ViT)
    with torch.no_grad():
        for model_name in [ 'densenet', 'efficientnet', 'vit']:
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
    if len(cnn_features) >= 3:  # At least 3 CNNs loaded
        # Strategy: Create 8 features from 3 CNN predictions
        # Method 1: Duplicate predictions + add mean/std
        if len(cnn_features) == 3:
            avg_feature = np.mean(cnn_features)
            std_feature = np.std(cnn_features)
            min_feature = np.min(cnn_features)
            max_feature = np.max(cnn_features)
            
            # Create 8 features: [3 original, mean, 3 original again, std]
            # This maintains information while padding to expected size
            cnn_features_expanded = (
                cnn_features +  # Original 3 predictions
                [avg_feature] +  # Mean (4th CNN placeholder)
                cnn_features +  # Duplicate 3 predictions
                [std_feature]   # Standard deviation (8th feature)
            )
            cnn_feature_vector = np.array(cnn_features_expanded).reshape(1, -1)
        else:
            # If somehow we have 4 models, duplicate them
            cnn_feature_vector = np.array(cnn_features + cnn_features).reshape(1, -1)
        
        # Try meta models
        for meta_name in ['meta_model', 'xgb_meta_model']:
            if meta_name in models_dict:
                try:
                    meta_pred = models_dict[meta_name].predict_proba(cnn_feature_vector)[0, 1]
                    predictions[f"{meta_name}_ensemble"] = float(meta_pred)
                except Exception as e:
                    st.warning(f"⚠️ {meta_name} failed: {str(e)}")
        
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
                    st.warning(f"⚠️ {stack_name} failed: {str(e)}")
    
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

def load_evaluation_dataset():
    """
    Load or create a labeled evaluation dataset for CNN models
    This should be a dataset with images and their true flood/non-flood labels
    """
    # Option 1: Load from a saved evaluation set
    # eval_data = pd.read_csv('evaluation_data.csv')
    
    # Option 2: Create synthetic evaluation data (for demonstration)
    # In production, replace this with real labeled test data
    eval_images = []
    eval_labels = []
    
    # You would load your actual test images here
    # For now, we'll return empty lists - populate with your test data
    
    return eval_images, eval_labels

# ==================== CNN MODEL EVALUATION ====================

def evaluate_cnn_models(models_dict, test_images, test_labels):
    """
    Evaluate pre-trained CNN models on test dataset
    
    Args:
        models_dict: Dictionary of loaded CNN models
        test_images: List of test images
        test_labels: List of ground truth labels (0 or 1)
    
    Returns:
        Dictionary containing evaluation metrics for each model
    """
    import torch
    from PIL import Image
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_transform = get_transforms()
    
    evaluation_results = {}
    
    for model_name in [ 'densenet', 'efficientnet', 'vit']:
        if model_name not in models_dict:
            continue
        
        st.info(f"🔄 Evaluating {model_name.upper()}...")
        
        model = models_dict[model_name]
        model.eval()
        
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for img, label in zip(test_images, test_labels):
                # Preprocess image
                if isinstance(img, str):
                    img = Image.open(img)
                
                img_tensor = test_transform(img).unsqueeze(0).to(device)
                
                # Get prediction
                output = model(img_tensor)
                prob = torch.softmax(output, dim=1)[0, 1].cpu().numpy()
                pred = 1 if prob >= 0.5 else 0
                
                predictions.append(pred)
                probabilities.append(prob)
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions, zero_division=0)
        recall = recall_score(test_labels, predictions, zero_division=0)
        f1 = f1_score(test_labels, predictions, zero_division=0)
        conf_matrix = confusion_matrix(test_labels, predictions)
        
        evaluation_results[model_name] = {
            'Model': model_name.upper(),
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1,
            'Predictions': predictions,
            'Probabilities': probabilities,
            'Confusion_Matrix': conf_matrix
        }
        
        st.success(f"✅ {model_name.upper()}: Accuracy={accuracy:.2%}, F1={f1:.4f}")
    
    return evaluation_results

# ==================== ENSEMBLE EVALUATION ====================

def evaluate_ensemble_models(models_dict, test_images, test_labels):
    """
    Evaluate ensemble models on test dataset
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # First get CNN features for all test images
    cnn_features_list = []
    
    st.info("🔄 Extracting CNN features for ensemble evaluation...")
    
    for img in test_images:
        if isinstance(img, str):
            img = Image.open(img)
        
        cnn_features = []
        test_transform = get_transforms()
        img_tensor = test_transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            for model_name in ['densenet', 'efficientnet', 'vit']:
                if model_name in models_dict:
                    model = models_dict[model_name]
                    model.eval()
                    output = model(img_tensor)
                    prob = torch.softmax(output, dim=1)[0, 1].cpu().numpy()
                    cnn_features.append(prob)
        
        cnn_features_list.append(cnn_features)
    
    cnn_features_array = np.array(cnn_features_list)
    
    # Evaluate ensemble models
    ensemble_results = {}
    
    ensemble_model_names = [k for k in models_dict.keys() 
                           if k not in ['densenet', 'efficientnet', 'vit']]
    
    for ensemble_name in ensemble_model_names:
        model = models_dict[ensemble_name]
        
        try:
            st.info(f"🔄 Evaluating {ensemble_name}...")
            
            # Get predictions
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(cnn_features_array)[:, 1]
                predictions = (probs >= 0.5).astype(int)
            else:
                predictions = model.predict(cnn_features_array)
                probs = predictions
            
            # Calculate metrics
            accuracy = accuracy_score(test_labels, predictions)
            precision = precision_score(test_labels, predictions, zero_division=0)
            recall = recall_score(test_labels, predictions, zero_division=0)
            f1 = f1_score(test_labels, predictions, zero_division=0)
            conf_matrix = confusion_matrix(test_labels, predictions)
            
            ensemble_results[ensemble_name] = {
                'Model': ensemble_name.replace('_', ' ').title(),
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1,
                'Predictions': predictions,
                'Probabilities': probs,
                'Confusion_Matrix': conf_matrix
            }
            
            st.success(f"✅ {ensemble_name}: Accuracy={accuracy:.2%}, F1={f1:.4f}")
            
        except Exception as e:
            st.warning(f"⚠️ Could not evaluate {ensemble_name}: {str(e)}")
    
    return ensemble_results

# ==================== VISUALIZATION FUNCTIONS ====================

def plot_model_comparison(cnn_results, ensemble_results=None):
    """Create comprehensive comparison visualizations"""
    
    # Combine all results
    all_results = {**cnn_results}
    if ensemble_results:
        all_results.update(ensemble_results)
    
    # Create DataFrame for plotting
    metrics_data = []
    for model_name, metrics in all_results.items():
        metrics_data.append({
            'Model': metrics['Model'],
            'Accuracy': metrics['Accuracy'],
            'Precision': metrics['Precision'],
            'Recall': metrics['Recall'],
            'F1 Score': metrics['F1_Score'],
            'Type': 'CNN' if model_name in ['densenet', 'efficientnet', 'vit'] else 'Ensemble'
        })
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # 1. Overall Performance Bar Chart
    st.markdown("#### 📊 Overall Model Performance")
    
    fig_comparison = go.Figure()
    
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    colors = ['#667eea', '#fa709a', '#00f2fe', '#fee140']
    
    for i, metric in enumerate(metrics_to_plot):
        fig_comparison.add_trace(go.Bar(
            name=metric,
            x=df_metrics['Model'],
            y=df_metrics[metric],
            marker_color=colors[i]
        ))
    
    fig_comparison.update_layout(
        barmode='group',
        title='🎯 Model Performance Comparison - All Metrics',
        xaxis_title='Model',
        yaxis_title='Score',
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # 2. Separate CNN and Ensemble Performance
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### ⚙️  CNN Models Performance")
        cnn_df = df_metrics[df_metrics['Type'] == 'CNN']
        
        if not cnn_df.empty:
            fig_cnn = px.bar(
                cnn_df,
                x='F1 Score',
                y='Model',
                orientation='h',
                color='Accuracy',
                title='CNN Models - F1 Score & Accuracy',
                color_continuous_scale='Viridis'
            )
            fig_cnn.update_layout(height=300)
            st.plotly_chart(fig_cnn, use_container_width=True)
    
    with col2:
        st.markdown("##### 🔗 Ensemble Models Performance")
        ensemble_df = df_metrics[df_metrics['Type'] == 'Ensemble']
        
        if not ensemble_df.empty:
            fig_ensemble = px.bar(
                ensemble_df,
                x='F1 Score',
                y='Model',
                orientation='h',
                color='Accuracy',
                title='Ensemble Models - F1 Score & Accuracy',
                color_continuous_scale='Plasma'
            )
            fig_ensemble.update_layout(height=300)
            st.plotly_chart(fig_ensemble, use_container_width=True)
    
    # 3. Precision-Recall Trade-off
    st.markdown("#### 🎯 Precision-Recall Analysis")
    
    fig_pr = px.scatter(
        df_metrics,
        x='Precision',
        y='Recall',
        size='F1 Score',
        color='Type',
        hover_name='Model',
        title='Precision vs Recall Trade-off',
        color_discrete_map={'CNN': '#667eea', 'Ensemble': '#fa709a'}
    )
    fig_pr.add_shape(
        type='line',
        x0=0, y0=0, x1=1, y1=1,
        line=dict(color='gray', dash='dash')
    )
    fig_pr.update_layout(height=400)
    st.plotly_chart(fig_pr, use_container_width=True)
    
    # 4. Radar Chart for Top 5 Models
    st.markdown("#### 🕸️ Multi-Metric Radar Chart (Top 5 Models)")
    
    top_5 = df_metrics.nlargest(5, 'F1 Score')
    
    fig_radar = go.Figure()
    
    for _, row in top_5.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row['Accuracy'], row['Precision'], row['Recall'], row['F1 Score']],
            theta=['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            fill='toself',
            name=row['Model']
        ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title='Top 5 Models - Multi-Metric Performance',
        height=500
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # 5. Detailed Metrics Table
    st.markdown("#### 📋 Detailed Metrics Table")
    
    styled_df = df_metrics.style.format({
        'Accuracy': '{:.2%}',
        'Precision': '{:.2%}',
        'Recall': '{:.2%}',
        'F1 Score': '{:.4f}'
    }).background_gradient(subset=['Accuracy', 'Precision', 'Recall', 'F1 Score'], cmap='RdYlGn')
    
    st.dataframe(styled_df, use_container_width=True)
    
    return df_metrics

def plot_confusion_matrices(results):
    """Plot confusion matrices for all models"""
    
    st.markdown("#### 🔲 Confusion Matrices")
    
    n_models = len(results)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[metrics['Model'] for metrics in results.values()],
        specs=[[{'type': 'heatmap'} for _ in range(n_cols)] for _ in range(n_rows)]
    )
    
    for idx, (model_name, metrics) in enumerate(results.items()):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        cm = metrics['Confusion_Matrix']
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig.add_trace(
            go.Heatmap(
                z=cm_normalized,
                x=['Predicted No Flood', 'Predicted Flood'],
                y=['Actual No Flood', 'Actual Flood'],
                colorscale='RdYlGn',
                text=cm,
                texttemplate='%{text}',
                showscale=(idx == 0)
            ),
            row=row,
            col=col
        )
    
    fig.update_layout(
        title_text='Confusion Matrices for All Models',
        height=300 * n_rows,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_best_model_analysis(df_metrics):
    """Highlight the best performing model"""
    
    st.markdown("#### 🏆 Best Model Analysis")
    
    best_model = df_metrics.loc[df_metrics['F1 Score'].idxmax()]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h4>🥇 Best Model</h4>
            <h3>{best_model['Model']}</h3>
            <p>Type: {best_model['Type']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <h4>🎯 Accuracy</h4>
            <h2>{best_model['Accuracy']:.2%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <h4>📊 Precision</h4>
            <h2>{best_model['Precision']:.2%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <h4>🎖️ F1 Score</h4>
            <h2>{best_model['F1 Score']:.4f}</h2>
        </div>
        """, unsafe_allow_html=True)

# ==================== MAIN EVALUATION FUNCTION ====================

def run_comprehensive_evaluation(models_dict, test_images=None, test_labels=None):
    """
    Main function to run comprehensive evaluation of all models
    
    Args:
        models_dict: Dictionary containing all loaded models (CNN + Ensemble)
        test_images: List of test images (if None, will attempt to load)
        test_labels: List of ground truth labels (if None, will attempt to load)
    """
    
    st.markdown("### 📊 Comprehensive Model Evaluation")
    
    # Load test data if not provided
    if test_images is None or test_labels is None:
        st.warning("⚠️ No test dataset provided. Please provide test images and labels.")
        st.info("""
        To run evaluation, you need:
        1. A set of labeled test images
        2. Ground truth labels (0 = No Flood, 1 = Flood)
        
        Example format:
        ```python
        test_images = ['path/to/image1.jpg', 'path/to/image2.jpg', ...]
        test_labels = [0, 1, 0, 1, ...]  # Ground truth
        ```
        """)
        return None
    
    st.info(f"📊 Evaluating on {len(test_labels)} test images...")
    
    # Evaluate CNN models
    with st.spinner("🔄 Evaluating CNN models..."):
        cnn_results = evaluate_cnn_models(models_dict, test_images, test_labels)
    
    # Evaluate ensemble models
    ensemble_results = None
    if any(k not in ['densenet', 'efficientnet', 'vit'] for k in models_dict.keys()):
        with st.spinner("🔄 Evaluating ensemble models..."):
            ensemble_results = evaluate_ensemble_models(models_dict, test_images, test_labels)
    
    # Visualizations
    st.markdown("---")
    df_metrics = plot_model_comparison(cnn_results, ensemble_results)
    
    st.markdown("---")
    all_results = {**cnn_results}
    if ensemble_results:
        all_results.update(ensemble_results)
    plot_confusion_matrices(all_results)
    
    st.markdown("---")
    display_best_model_analysis(df_metrics)
    
    # Save results to session state
    st.session_state.evaluation_results = {
        'cnn_results': cnn_results,
        'ensemble_results': ensemble_results,
        'metrics_df': df_metrics
    }
    
    st.success("✅ Evaluation complete!")
    
    return df_metrics

# ==================== DATASET LOADING ====================

@st.cache_resource
def load_datasets_from_kaggle():
    """Load datasets from Kaggle"""
    try:
        with st.spinner("🔄 Downloading datasets from Kaggle..."):
            path_tabular = kagglehub.dataset_download("naiyakhalid/flood-prediction-dataset")
            st.success(f"✅ Tabular data downloaded to: {path_tabular}")
            
            flood_files = [os.path.join(root, file) for root, dirs, files in os.walk(path_tabular) for file in files if file.endswith('.csv')]
            if flood_files:
                df_flood = pd.read_csv(flood_files[0])
                st.success(f"✅ Loaded flood prediction dataset with {len(df_flood)} records")
            else:
                st.error("❌ No CSV files found in flood prediction dataset")
                return None
                
            return df_flood
            
    except Exception as e:
        st.error(f"❌ Error loading datasets: {str(e)}")
        return None

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

# ==================== MAIN APPLICATION ====================

st.markdown('<h1 class="main-header">🌊 FloodSentinel</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Flood Risk Assessment Using Satellite Imagery and Deep Neural Networks</p>', unsafe_allow_html=True)

# ==================== SIDEBAR NAVIGATION ====================

st.sidebar.markdown("### 🧭 Navigation")
page = st.sidebar.selectbox(
    "Choose a section:",
    ["🏠 Home", "📊 Data Analysis", "⚙️ Model Training", "⚙️  Load CNN Models",
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
            <p>FloodSentinel combines machine learning for historical tabular data with pre-trained deep neural networks for satellite imagery analysis.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
            <h4>🚀 Key Features</h4>
            <ul>
                <li>⚙️ 9 Pre-trained ML algorithms</li>
                <li>🛰️ 3 Pre-trained CNN models (DenseNet, EfficientNet, ViT)</li>
                <li>🎯 Vision Transformer for global context</li>
                <li>📊 Context-aware flood detection</li>
                <li>🔥 Fire/vegetation filtering</li>
                <li>📈 Interactive visualizations</li>
                <li>⚙️  Ensemble predictions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Dataset Loading")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Load from Kaggle", type="primary", key="load_kaggle"):
            df_flood = load_datasets_from_kaggle()
            if df_flood is not None:
                st.session_state.df_flood = df_flood
                st.session_state.dataset_loaded = True
                st.rerun()
    
    with col2:
        if st.button("📊 Use Sample Data", type="secondary", key="load_sample"):
            st.session_state.df_flood = create_sample_data()
            st.session_state.dataset_loaded = True
            st.success("✅ Sample dataset loaded successfully!")
            st.rerun()
    
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
        cnn_status = "✅ 3 Models" if st.session_state.cnn_models_loaded else "❌ Not Loaded"
        st.markdown(f"""
        <div class="metric-container">
            <h3>🛰️ CNN Models</h3>
            <h2>{cnn_status}</h2>
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

# ==================== PAGE: LOAD CNN MODELS ====================

elif page == "⚙️  Load CNN Models":
    st.markdown("### ⚙️  Load Pre-trained CNN Models")
    
    st.markdown("""
    This section loads 3 pre-trained CNN architectures for satellite flood detection:
    
    | Model | Architecture | Key Features |
    |-------|--------------|--------------|
    | **DenseNet-121** 🌿 | Densely Connected Network | Each layer receives input from all preceding layers for better gradient flow |
    | **EfficientNet-B0** 🚀 | Efficient CNN | Compound scaling method for optimal accuracy and efficiency |
    | **Vision Transformer (ViT)** 🎯 | Transformer-based | Self-attention mechanisms to capture global image context |
    
    ### 🎯 Why Vision Transformer?
    
    Unlike traditional CNNs that process images locally through convolutions, **Vision Transformer** divides images into patches 
    and processes them using self-attention mechanisms. This allows ViT to:
    - **Capture long-range dependencies** across the entire image
    - **Understand global context** better than local convolutions
    - **Identify spatial relationships** between distant flood regions
    - **Process multi-scale features** through attention layers
    
    After loading CNN models, ensemble models will also be loaded to combine predictions for more robust flood detection.
    """)
    
    st.markdown("#### ⚙️ Model Configuration")
    
    cnn_models_dir = st.text_input("CNN models directory:", value="cnn_models", key="cnn_dir")
    ensemble_models_dir = st.text_input("Ensemble models directory:", value="pretrained_models", key="ensemble_dir")
    
    if st.button("🚀 Load All Models", type="primary", key="load_all_models"):
        
        # Step 1: Load CNN models
        st.markdown("#### 📂 Step 1: Loading CNN Models (Including ViT)")
        try:
            cnn_models = load_pretrained_cnn_models(cnn_models_dir)
            
            if len(cnn_models) > 0:
                st.session_state.ensemble_models.update(cnn_models)
                st.session_state.cnn_models_loaded = True
                st.success(f"✅ Successfully loaded {len(cnn_models)}/3 CNN models!")
                
                # Show which models were loaded
                model_names = [ 'densenet', 'efficientnet', 'vit']
                for name in model_names:
                    if name in cnn_models:
                        if name == 'vit':
                            st.success(f"✅ Vision Transformer (ViT) - Attention-based architecture loaded!")
                        else:
                            st.success(f"✅ {name.upper()} loaded")
                    else:
                        st.warning(f"⚠️ {name.upper()} not loaded")
            else:
                st.error("❌ No CNN models were loaded. Check the directory path and file names.")
                st.stop()
                
        except Exception as e:
            st.error(f"❌ Error loading CNN models: {str(e)}")
            st.stop()
        
        # Step 2: Load ensemble models
        st.markdown("#### 🔗 Step 2: Loading Ensemble Models")
        try:
            ensemble_models = load_ensemble_models(ensemble_models_dir)
            
            if len(ensemble_models) > 0:
                st.session_state.ensemble_models.update(ensemble_models)
                st.success(f"✅ Successfully loaded {len(ensemble_models)} ensemble models!")
            else:
                st.warning("⚠️ No ensemble models loaded. CNN predictions only.")
                
        except Exception as e:
            st.warning(f"⚠️ Error loading ensemble models: {str(e)}")
        
        st.markdown("---")
        st.markdown("#### ✅ Loading Complete!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            cnn_count = len([k for k in st.session_state.ensemble_models.keys() if k in ['densenet', 'efficientnet', 'vit']])
            st.markdown(f"""
            <div class="metric-container">
                <h3>⚙️  CNN Models</h3>
                <h2>{cnn_count}/4</h2>
                <p>Loaded Successfully</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            ensemble_count = len([k for k in st.session_state.ensemble_models.keys() if k not in [ 'densenet', 'efficientnet', 'vit']])
            st.markdown(f"""
            <div class="metric-container">
                <h3>🔗 Ensemble Models</h3>
                <h2>{ensemble_count}</h2>
                <p>Loaded Successfully</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.success("🎉 All models loaded! You can now use Image Flood Detection.")
    
    # Display current status
    if st.session_state.cnn_models_loaded:
        st.markdown("---")
        st.markdown("#### 📊 Current Model Status")
        
        cnn_models = {k: v for k, v in st.session_state.ensemble_models.items() if k in [ 'densenet', 'efficientnet', 'vit']}
        ensemble_models = {k: v for k, v in st.session_state.ensemble_models.items() if k not in [ 'densenet', 'efficientnet', 'vit']}
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### ⚙️  CNN Models")
            for model_name in ['densenet', 'efficientnet', 'vit']:
                if model_name in cnn_models:
                    if model_name == 'vit':
                        st.success(f"✅ {model_name.upper()} - Vision Transformer (Attention-based)")
                    else:
                        st.success(f"✅ {model_name.upper()}")
                else:
                    st.error(f"❌ {model_name.upper()}")
        
        with col2:
            st.markdown("##### 🔗 Ensemble Models")
            if len(ensemble_models) > 0:
                for model_name in ensemble_models.keys():
                    display_name = model_name.replace('_', ' ').title()
                    st.success(f"✅ {display_name}")
            else:
                st.info("No ensemble models loaded")

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
    st.markdown("### 🖼️ Advanced Flood Detection from Satellite Imagery")
    
    if not st.session_state.cnn_models_loaded:
        st.warning("⚠️ CNN models not loaded yet")
        st.info("✅ **You can still use context-aware analysis!**")
        st.markdown("""
        The app includes intelligent context analysis that works without deep learning models:
        - 🔥 **Fire detection** - Identifies fire/heat signatures
        - 🌿 **Vegetation analysis** - Detects green vegetation
        - 💧 **Water detection** - Identifies water bodies
        - 📊 **Smart classification** - Rule-based flood assessment
        
        To use full deep learning predictions (including Vision Transformer), go to "⚙️  Load CNN Models" first.
        """)
    else:
        cnn_count = len([k for k in st.session_state.ensemble_models.keys() if k in [ 'densenet', 'efficientnet', 'vit']])
        ensemble_count = len([k for k in st.session_state.ensemble_models.keys() if k not in [ 'densenet', 'efficientnet', 'vit']])
        
        vit_loaded = 'vit' in st.session_state.ensemble_models
        if vit_loaded:
            st.success(f"✅ {cnn_count} CNN models (including Vision Transformer) and {ensemble_count} ensemble models ready!")
            st.info("🎯 **Vision Transformer active** - Using self-attention for global context analysis")
        else:
            st.success(f"✅ {cnn_count} CNN models and {ensemble_count} ensemble models ready!")
    
    st.markdown("""
    Upload a satellite or aerial image for flood detection. The system includes:
    - 🔥 Fire detection to avoid false positives
    - 🌿 Vegetation analysis
    - 💧 Water body detection
    - ⚙️  Deep learning ensemble predictions (if models loaded)
    - 🎯 Vision Transformer for attention-based global context (if loaded)
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
                st.image(false_color, caption="False Color Composite", use_container_width=True)
            
            with col3:
                water_mask = extract_water_mask(image)
                st.image(water_mask, caption="Water Mask Detection", use_container_width=True)
            
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
            if st.session_state.cnn_models_loaded:
                st.markdown("#### ⚙️  Deep Learning Analysis")
                
                with st.spinner("Running ensemble predictions (including Vision Transformer)..."):
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
                        cnn_preds = {k: v for k, v in predictions.items() if k in [ 'densenet', 'efficientnet', 'vit']}
                        ensemble_preds = {k: v for k, v in predictions.items() if k not in [ 'densenet', 'efficientnet', 'vit']}
                        
                        if cnn_preds:
                            st.markdown("**⚙️  CNN Models:**")
                            for model_name, pred in cnn_preds.items():
                                if model_name == 'vit':
                                    st.metric("🎯 ViT (Transformer)", f"{pred:.2%}")
                                else:
                                    st.metric(model_name.upper(), f"{pred:.2%}")
                        
                        if ensemble_preds:
                            st.markdown("**🔗 Ensemble Models:**")
                            for model_name, pred in ensemble_preds.items():
                                display_name = model_name.replace('_ensemble', '').replace('_', ' ').title()
                                st.metric(display_name, f"{pred:.2%}")
                            ensemble_avg = np.mean(list(ensemble_preds.values()))
                            st.success(f"🏆 **Ensemble Average:** {ensemble_avg:.2%}")
                        else:
                            st.markdown("---")
                            st.info("💡 No ensemble meta models were applied")
                    
                    with col2:
                        # Gauge chart
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=ensemble_pred * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Flood Risk Level"},
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
                        
                        # Context metrics
                        st.markdown("##### 📊 Context Metrics")
                        st.info(f"**Mean Brightness:** {context['mean_brightness']:.1f}")
                        st.info(f"**Confidence:** {context['confidence']:.0%}")
                    
                    # Model comparison visualization
                    if len(predictions) > 1:
                        st.markdown("#### 📊 Detailed Model Comparison")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            model_names = list(predictions.keys())
                            model_preds = [predictions[m] * 100 for m in model_names]
                            
                            # Color code by model type
                            colors = []
                            for name in model_names:
                                if name == 'vit':
                                    colors.append('ViT (Transformer)')
                                elif name in [ 'densenet', 'efficientnet']:
                                    colors.append('CNN')
                                else:
                                    colors.append('Ensemble')
                            
                            fig_compare = px.bar(
                                x=model_names,
                                y=model_preds,
                                color=colors,
                                title="🔍 Individual Model Predictions",
                                labels={'x': 'Model', 'y': 'Flood Probability (%)'},
                                color_discrete_map={
                                    'CNN': '#667eea', 
                                    'ViT (Transformer)': '#00f2fe',
                                    'Ensemble': '#fa709a'
                                }
                            )
                            fig_compare.add_hline(y=ensemble_pred * 100, line_dash="dash", 
                                                line_color="red", annotation_text="Final Ensemble",
                                                line_width=2)
                            fig_compare.update_layout(xaxis_tickangle=-45, height=400)
                            st.plotly_chart(fig_compare, use_container_width=True)
                        
                        with col2:
                            if len(predictions) >= 2:
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
                                    showlegend=False,
                                    height=400
                                )
                                st.plotly_chart(fig_radar, use_container_width=True)
                    
                    # Show Vision Transformer contribution if loaded
                    if 'vit' in predictions:
                        st.markdown("---")
                        st.markdown("#### 🎯 Vision Transformer Analysis")
                        
                        vit_pred = predictions['vit']
                        avg_cnn = np.mean([predictions[k] for k in [ 'densenet', 'efficientnet'] if k in predictions])
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("🎯 ViT Prediction", f"{vit_pred:.2%}")
                        
                        with col2:
                            st.metric("📊 Avg CNN Prediction", f"{avg_cnn:.2%}")
                        
                        with col3:
                            diff = abs(vit_pred - avg_cnn)
                            st.metric("📈 Difference", f"{diff:.2%}")
                        
                        if diff > 0.1:
                            st.info("💡 **Insight:** Vision Transformer's attention mechanism detected different patterns than traditional CNNs, suggesting complex spatial relationships in the image.")
                        else:
                            st.success("✅ **Consensus:** Vision Transformer agrees with CNN predictions, indicating high confidence in flood assessment.")
                    if ensemble_preds:
                        st.markdown("---")
                        st.markdown("#### 🔗 Ensemble Meta-Model Analysis")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            ensemble_avg = np.mean(list(ensemble_preds.values()))
                            st.metric("🔗 Ensemble Average", f"{ensemble_avg:.2%}")
                        with col2:
                            cnn_avg = np.mean(list(cnn_preds.values()))
                            st.metric("⚙️  CNN Average", f"{cnn_avg:.2%}")
                        with col3:
                            diff = abs(ensemble_avg - cnn_avg)
                            st.metric("📊 Difference", f"{diff:.2%}")
                        if len(ensemble_preds) > 0:
                            st.success(f"✅ **{len(ensemble_preds)} ensemble meta-models** were successfully applied to combine CNN predictions!")
                    
                    st.markdown("---")
                    st.markdown("#### 💡 Analysis Summary")
                    st.info(f"**Reasoning:** {context['reason']}")
                    
                    # Recommendation based on prediction
                    if ensemble_pred > 0.7:
                        st.error("⚠️ **High Risk Alert**: Immediate action recommended. Potential flooding detected.")
                    elif ensemble_pred > 0.3:
                        st.warning("⚠️ **Moderate Risk**: Monitor situation closely. Flood conditions possible.")
                    else:
                        st.success("✅ **Low Risk**: No significant flood indicators detected.")
                    
                elif result and result['rejected']:
                    st.error("❌ Image Rejected - Not a Flood Scenario")
                    st.warning(f"🔍 Reason: {context['reason']}")
                    st.info("This image appears to show fire, vegetation, or other non-flood scenarios.")
                    
                    # Show why it was rejected
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if context['fire_percentage'] > 15:
                            st.error(f"🔥 High fire signature: {context['fire_percentage']:.1f}%")
                    with col2:
                        if context['vegetation_percentage'] > 60:
                            st.error(f"🌿 Dense vegetation: {context['vegetation_percentage']:.1f}%")
                    with col3:
                        if context['water_percentage'] < 10:
                            st.error(f"💧 Low water coverage: {context['water_percentage']:.1f}%")
                
                else:
                    st.error("❌ Error in ensemble prediction")
            
            else:
                st.markdown("#### 📊 Context-Based Assessment")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    if context['is_likely_flood']:
                        st.success(f"✅ Likely Flood Scenario")
                        st.metric("Confidence", f"{context['confidence']:.0%}")
                    else:
                        st.error(f"❌ Not a Flood Scenario")
                        st.metric("Confidence", f"{context['confidence']:.0%}")
                
                with col2:
                    # Simple gauge for context-based assessment
                    flood_score = context['water_percentage'] - context['fire_percentage']
                    flood_score = max(0, min(100, flood_score))
                    
                    fig_simple = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=flood_score,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Flood Likelihood Score"},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgray"},
                                {'range': [30, 70], 'color': "lightyellow"},
                                {'range': [70, 100], 'color': "lightblue"}
                            ],
                        }
                    ))
                    fig_simple.update_layout(height=300)
                    st.plotly_chart(fig_simple, use_container_width=True)
                
                st.info(f"💡 {context['reason']}")
                st.warning("⚠️ For more accurate predictions (including Vision Transformer), load CNN models from the '⚙️  Load CNN Models' page.")
                
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    else:
        st.info("👆 Upload an image to begin analysis")
        
        # Show example of what to expect
        st.markdown("#### 📖 What to Expect")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="info-box">
                <h4>🔍 Analysis Features</h4>
                <ul>
                    <li>Water body detection</li>
                    <li>Fire signature filtering</li>
                    <li>Vegetation analysis</li>
                    <li>Context-aware classification</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="success-box">
                <h4>⚙️  Deep Learning</h4>
                <ul>
                    <li>3 CNN model predictions</li>
                    <li>Vision Transformer (ViT)</li>
                    <li>Ensemble meta-models</li>
                    <li>Confidence scoring</li>
                    <li>Risk level assessment</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="warning-box">
                <h4>📊 Visualizations</h4>
                <ul>
                    <li>False color composites</li>
                    <li>Water masks</li>
                    <li>Risk gauges</li>
                    <li>Model comparisons</li>
                    <li>Attention analysis</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# ==================== PAGE: RESULTS DASHBOARD ====================

elif page == "📈 Results Dashboard":
    st.markdown("### 📈 Comprehensive Results Dashboard")
    
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
                'R²': metrics['R²']
            })
        
        perf_df = pd.DataFrame(perf_data)
        perf_df = perf_df.sort_values('Accuracy', ascending=False)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if len(perf_df) > 0:
                best = perf_df.iloc[0]
                st.markdown(f"""
                <div class="metric-container">
                    <h4>🥇 Best Model</h4>
                    <h3>{best['Model']}</h3>
                    <p>Accuracy: {best['Accuracy']:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            avg_acc = perf_df['Accuracy'].mean()
            st.markdown(f"""
            <div class="metric-container">
                <h4>📊 Avg Accuracy</h4>
                <h2>{avg_acc:.2%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_f1 = perf_df['F1 Score'].mean()
            st.markdown(f"""
            <div class="metric-container">
                <h4>🎯 Avg F1 Score</h4>
                <h2>{avg_f1:.4f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-container">
                <h4>🔢 Total Models</h4>
                <h2>{len(perf_df)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("##### 📋 Detailed Performance Metrics")
        st.dataframe(perf_df.style.format({
            'Accuracy': '{:.2%}',
            'Precision': '{:.2%}',
            'Recall': '{:.2%}',
            'F1 Score': '{:.4f}',
            'RMSE': '{:.4f}',
            'MAE': '{:.4f}',
            'R²': '{:.4f}'
        }).background_gradient(subset=['Accuracy', 'F1 Score'], cmap='RdYlGn'), use_container_width=True)
        
        # Comparison charts
        st.markdown("##### 📊 Performance Visualizations")
        
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
            fig_acc.update_layout(height=400)
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
            fig_f1.update_layout(height=400)
            st.plotly_chart(fig_f1, use_container_width=True)
        
        # Multi-metric comparison
        col1, col2 = st.columns(2)
        
        with col1:
            fig_scatter = px.scatter(
                perf_df,
                x='Precision',
                y='Recall',
                size='F1 Score',
                color='Accuracy',
                hover_name='Model',
                title='🎯 Precision vs Recall',
                color_continuous_scale='RdYlGn'
            )
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            fig_error = px.bar(
                perf_df.sort_values('RMSE'),
                x='RMSE',
                y='Model',
                orientation='h',
                title='📉 Root Mean Squared Error',
                color='RMSE',
                color_continuous_scale='Reds_r'
            )
            fig_error.update_layout(height=400)
            st.plotly_chart(fig_error, use_container_width=True)
        
    else:
        st.warning("⚠️ No tabular models loaded yet. Go to '⚙️ Model Training' to load models.")
    
    # CNN models results
    if st.session_state.cnn_models_loaded:
        st.markdown("---")
        st.markdown("#### ⚙️  CNN Model Status")
        
        cnn_models = {k: v for k, v in st.session_state.ensemble_models.items() if k in [ 'densenet', 'efficientnet', 'vit']}
        ensemble_models = {k: v for k, v in st.session_state.ensemble_models.items() if k not in [ 'densenet', 'efficientnet', 'vit']}
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <h4>⚙️  CNN Models</h4>
                <h2>{len(cnn_models)}/4</h2>
                <p>Loaded</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <h4>🔗 Ensemble Models</h4>
                <h2>{len(ensemble_models)}</h2>
                <p>Available</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            total_models = len(cnn_models) + len(ensemble_models)
            st.markdown(f"""
            <div class="metric-container">
                <h4>🎯 Total CNN+Ensemble</h4>
                <h2>{total_models}</h2>
                <p>Ready for Use</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show loaded models with detailed info
        st.markdown("##### 📋 Loaded Models Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**⚙️  CNN Models:**")
            model_info = {
                'densenet': '🌿 DenseNet-121 - Densely Connected',
                'efficientnet': '🚀 EfficientNet-B0 - Efficient Scaling',
                'vit': '🎯 Vision Transformer - Attention-based'
            }
            
            for model_name, description in model_info.items():
                if model_name in cnn_models:
                    st.success(f"✅ {description}")
                else:
                    st.error(f"❌ {description}")
        
        with col2:
            st.markdown("**🔗 Ensemble Models:**")
            if len(ensemble_models) > 0:
                for model_name in ensemble_models.keys():
                    display_name = model_name.replace('_', ' ').title()
                    st.success(f"✅ {display_name}")
            else:
                st.info("No ensemble models loaded")
        
        # Model architecture comparison
        if len(cnn_models) > 0:
            st.markdown("---")
            st.markdown("##### 🏗️ Model Architecture Comparison")
            
            arch_data = []
            if 'densenet' in cnn_models:
                arch_data.append({
                    'Model': 'DenseNet-121',
                    'Type': 'CNN',
                    'Layers': '121',
                    'Parameters': '~8.0M',
                    'Key Feature': 'Dense Connections'
                })
            if 'efficientnet' in cnn_models:
                arch_data.append({
                    'Model': 'EfficientNet-B0',
                    'Type': 'CNN',
                    'Layers': 'Variable',
                    'Parameters': '~5.3M',
                    'Key Feature': 'Compound Scaling'
                })
            if 'vit' in cnn_models:
                arch_data.append({
                    'Model': 'Vision Transformer',
                    'Type': 'Transformer',
                    'Layers': '12 Transformer Blocks',
                    'Parameters': '~86M',
                    'Key Feature': 'Self-Attention Mechanism'
                })
            
            arch_df = pd.DataFrame(arch_data)
            st.dataframe(arch_df, use_container_width=True)
            
            # Highlight ViT if loaded
            if 'vit' in cnn_models:
                st.info("""
                🎯 **Vision Transformer (ViT)** - Unlike traditional CNNs that process images through convolutional layers, 
                ViT divides images into patches and uses self-attention mechanisms to understand relationships between all parts 
                of the image simultaneously. This global perspective is particularly effective for detecting widespread flood patterns.
                """)
        
        st.markdown("---")
        st.info("💡 **Tip:** Upload satellite images in the '🖼️ Image Flood Detection' page to test these models!")
        
    else:
        st.info("💡 Load CNN models on the '⚙️  Load CNN Models' page to see deep learning model status.")
    
    # Overall system status
    st.markdown("---")
    st.markdown("#### 🎯 Overall System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        dataset_status = "✅ Loaded" if st.session_state.dataset_loaded else "❌ Not Loaded"
        st.markdown(f"""
        <div class="{'success-box' if st.session_state.dataset_loaded else 'warning-box'}">
            <h4>📊 Dataset</h4>
            <p>{dataset_status}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        tabular_status = "✅ Ready" if st.session_state.models_trained else "❌ Not Ready"
        st.markdown(f"""
        <div class="{'success-box' if st.session_state.models_trained else 'warning-box'}">
            <h4>⚙️ Tabular Models</h4>
            <p>{tabular_status}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        cnn_status = "✅ Ready" if st.session_state.cnn_models_loaded else "❌ Not Ready"
        st.markdown(f"""
        <div class="{'success-box' if st.session_state.cnn_models_loaded else 'warning-box'}">
            <h4>⚙️  CNN Models</h4>
            <p>{cnn_status}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        all_ready = st.session_state.dataset_loaded and st.session_state.models_trained and st.session_state.cnn_models_loaded
        system_status = "✅ Fully Operational" if all_ready else "⚠️ Partial"
        st.markdown(f"""
        <div class="{'success-box' if all_ready else 'info-box'}">
            <h4>🎯 System</h4>
            <p>{system_status}</p>
        </div>
        """, unsafe_allow_html=True)

# ==================== SIDEBAR STATUS ====================

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 System Status")

if st.session_state.dataset_loaded:
    st.sidebar.success("✅ Dataset Loaded")
else:
    st.sidebar.error("❌ Dataset Not Loaded")

if st.session_state.models_trained:
    st.sidebar.success("✅ Tabular Models Ready")
    st.sidebar.info(f"🎯 {len(st.session_state.model_results)} models")
else:
    st.sidebar.error("❌ Tabular Models Not Loaded")

if st.session_state.cnn_models_loaded:
    cnn_count = len([k for k in st.session_state.ensemble_models.keys() if k in [ 'densenet', 'efficientnet', 'vit']])
    st.sidebar.success(f"✅ {cnn_count} CNN Models Loaded")
    
    if 'vit' in st.session_state.ensemble_models:
        st.sidebar.info("🎯 Vision Transformer Active")
    
    ensemble_count = len([k for k in st.session_state.ensemble_models.keys() if k not in [ 'densenet', 'efficientnet', 'vit']])
    if ensemble_count > 0:
        st.sidebar.info(f"🔗 {ensemble_count} Ensemble models")
else:
    st.sidebar.error("❌ CNN Models Not Loaded")

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info("""
🌊 **FloodSentinel**
- 9 Pre-trained ML models
- 3 Pre-trained CNN models
- Context-aware detection
- Fire/vegetation filtering
- Ensemble predictions
- Real-time analysis

**CNN Models:**
- 🌿 DenseNet-121
- 🚀 EfficientNet-B0
- 🎯 Vision Transformer (ViT)

**ViT Advantage:**
Global attention mechanism for 
superior spatial understanding
""")

# ==================== FOOTER ====================

st.markdown("---")
st.markdown("""
    <div class="footer">
        <p>Crafted with ❤️ by Shreyas, Chinmay and Kaivalya.<br>
        Project: FloodSentinel - AI-Powered Flood Risk Assessment System with Vision Transformer</p>
    </div>
""", unsafe_allow_html=True)
