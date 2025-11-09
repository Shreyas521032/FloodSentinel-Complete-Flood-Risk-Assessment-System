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
if 'scaler' not in st.session_state:
    st.session_state.scaler = None

# ==================== DEEP LEARNING MODEL LOADING FUNCTIONS ====================

import torch
import torch.nn as nn
from torchvision import models as torch_models
import gzip
import tempfile
import os
import pickle
import streamlit as st

def convert_pth_to_pkl(pth_path, pkl_path):
    """
    Convert a PyTorch .pth file to a pickle .pkl file
    """
    try:
        if not os.path.exists(pth_path):
            print(f"❌ File not found: {pth_path}")
            return False
        
        file_size = os.path.getsize(pth_path)
        if file_size == 0:
            print(f"❌ File is empty: {pth_path}")
            return False
        
        print(f"📦 Converting {os.path.basename(pth_path)} ({file_size / (1024*1024):.2f} MB)...")
        
        # Load the .pth file
        data = torch.load(pth_path, map_location="cpu", weights_only=False)
        
        # Save as .pkl
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        pkl_size = os.path.getsize(pkl_path)
        print(f"✅ Converted successfully! PKL size: {pkl_size / (1024*1024):.2f} MB")
        return True
        
    except RuntimeError as e:
        print(f"❌ RuntimeError during torch.load: {e}")
        print("This often indicates a corrupted or incomplete .pth file.")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def decompress_and_load_model_v3(model_path, model_architecture, device):
    """
    Enhanced model loading with automatic .pth to .pkl conversion
    Handles both compressed (.gz) and uncompressed files
    """
    if not os.path.exists(model_path):
        print(f"❌ File not found: {model_path}")
        return None
    
    file_size = os.path.getsize(model_path)
    if file_size == 0:
        print(f"❌ File is empty: {model_path}")
        return None
    
    print(f"📦 Loading {os.path.basename(model_path)} ({file_size / (1024*1024):.2f} MB)...")
    
    # Check file extension
    is_gzipped = model_path.endswith('.gz')
    is_pth = model_path.endswith('.pth') or model_path.endswith('.pth.gz')
    
    try:
        # Step 1: Handle decompression if needed
        if is_gzipped:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
                try:
                    with gzip.open(model_path, 'rb') as f_in:
                        decompressed_data = f_in.read()
                        tmp_file.write(decompressed_data)
                        tmp_file.flush()
                    
                    print(f"✅ Decompressed to {len(decompressed_data) / (1024*1024):.2f} MB")
                    load_path = tmp_file.name
                except gzip.BadGzipFile:
                    print(f"⚠️ Not a valid gzip file, trying as regular file")
                    load_path = model_path
        else:
            load_path = model_path
        
        # Step 2: If it's a .pth file, try converting to .pkl first
        if is_pth:
            pkl_path = load_path.replace('.pth', '.pkl')
            if load_path.endswith('.pth.gz'):
                pkl_path = load_path.replace('.pth.gz', '.pkl')
            
            # Only convert if pkl doesn't exist or is older
            if not os.path.exists(pkl_path) or os.path.getmtime(load_path) > os.path.getmtime(pkl_path):
                print(f"🔄 Converting .pth to .pkl format...")
                if convert_pth_to_pkl(load_path, pkl_path):
                    load_path = pkl_path
                    print(f"✅ Using converted .pkl file")
        
        # Step 3: Try loading the model
        print(f"🔄 Attempting to load model from {os.path.basename(load_path)}...")
        
        # Method 1: Try loading as checkpoint
        try:
            checkpoint = torch.load(load_path, map_location=device, weights_only=False)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                # Try different possible keys for state dict
                state_dict = None
                for key in ['state_dict', 'model_state_dict', 'model']:
                    if key in checkpoint:
                        state_dict = checkpoint[key]
                        print(f"✅ Found state dict in key: '{key}'")
                        break
                
                if state_dict is None:
                    # Assume checkpoint is the state dict itself
                    state_dict = checkpoint
                    print(f"✅ Using checkpoint as state dict")
                
                # Try loading state dict with strict=False
                try:
                    model_architecture.load_state_dict(state_dict, strict=False)
                    model_architecture.to(device)
                    model_architecture.eval()
                    print("✅ Loaded as state dict (strict=False)")
                    return model_architecture
                except Exception as e:
                    print(f"⚠️ State dict loading failed: {str(e)[:100]}")
                    
                    # Try to match keys manually
                    try:
                        model_dict = model_architecture.state_dict()
                        # Filter out incompatible keys
                        filtered_dict = {k: v for k, v in state_dict.items() 
                                       if k in model_dict and v.shape == model_dict[k].shape}
                        
                        if len(filtered_dict) > 0:
                            model_dict.update(filtered_dict)
                            model_architecture.load_state_dict(model_dict, strict=False)
                            model_architecture.to(device)
                            model_architecture.eval()
                            print(f"✅ Loaded with key matching ({len(filtered_dict)}/{len(state_dict)} keys)")
                            return model_architecture
                        else:
                            print(f"❌ No matching keys found")
                            return None
                    except Exception as e2:
                        print(f"❌ Key matching also failed: {str(e2)[:100]}")
                        return None
            
            # If checkpoint is a full model object
            elif hasattr(checkpoint, 'eval'):
                checkpoint.to(device)
                checkpoint.eval()
                print("✅ Loaded as full model object")
                return checkpoint
            else:
                print(f"❌ Unexpected checkpoint type: {type(checkpoint)}")
                return None
        
        except Exception as e:
            print(f"❌ Loading failed: {str(e)[:150]}")
            return None
        
    finally:
        # Clean up temp files
        if is_gzipped and 'tmp_file' in locals():
            try:
                os.unlink(tmp_file.name)
            except:
                pass


def load_pretrained_dl_models_v3(models_dir="pretrained_models"):
    """
    Enhanced model loading with automatic .pth to .pkl conversion
    """
    loaded_models = {}
    
    if not os.path.exists(models_dir):
        st.warning(f"⚠️ Models directory '{models_dir}' not found.")
        st.info(f"💡 Please create the directory or check the path: {os.path.abspath(models_dir)}")
        return loaded_models
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.info(f"🖥️ Using device: {device}")
    
    # Model configurations with multiple possible paths
    model_configs = {
        'resnet': {
            'paths': [
                os.path.join(models_dir, "model_compressed_resnet_model_checkpoint.pth.gz"),
                os.path.join(models_dir, "resnet_model_checkpoint.pth"),
                os.path.join(models_dir, "resnet.pth"),
                os.path.join(models_dir, "resnet_model.pkl")
            ],
            'architecture': lambda: torch_models.resnet50(pretrained=False)
        },
        'densenet': {
            'paths': [
                os.path.join(models_dir, "model_compressed_densenet_model_checkpoint.pth.gz"),
                os.path.join(models_dir, "densenet_model_checkpoint.pth"),
                os.path.join(models_dir, "densenet.pth"),
                os.path.join(models_dir, "densenet_model.pkl")
            ],
            'architecture': lambda: torch_models.densenet121(pretrained=False)
        },
        'vit': {
            'paths': [
                os.path.join(models_dir, "model_compressed.pth.gz"),
                os.path.join(models_dir, "vit_model_checkpoint.pth"),
                os.path.join(models_dir, "vit.pth"),
                os.path.join(models_dir, "vit_model.pkl")
            ],
            'architecture': None  # Will use timm
        },
        'efficientnet': {
            'paths': [
                os.path.join(models_dir, "efficientnet_model_checkpoint.pth"),
                os.path.join(models_dir, "efficientnet.pth"),
                os.path.join(models_dir, "efficientnet_model.pkl")
            ],
            'architecture': lambda: torch_models.efficientnet_b0(pretrained=False)
        }
    }
    
    # Load ResNet
    st.markdown("---")
    st.markdown("#### 🔄 Loading ResNet-50...")
    for path in model_configs['resnet']['paths']:
        if os.path.exists(path):
            st.success(f"✅ Found: {os.path.basename(path)}")
            resnet = model_configs['resnet']['architecture']()
            # Modify final layer for binary classification
            resnet.fc = nn.Linear(resnet.fc.in_features, 2)
            loaded_model = decompress_and_load_model_v3(path, resnet, device)
            if loaded_model:
                loaded_models['resnet'] = loaded_model
                st.success("✅ ResNet-50 loaded successfully!")
                break
    else:
        st.warning(f"⚠️ ResNet not found. Checked: {[os.path.basename(p) for p in model_configs['resnet']['paths']]}")
    
    # Load DenseNet
    st.markdown("---")
    st.markdown("#### 🔄 Loading DenseNet-121...")
    for path in model_configs['densenet']['paths']:
        if os.path.exists(path):
            st.success(f"✅ Found: {os.path.basename(path)}")
            densenet = model_configs['densenet']['architecture']()
            densenet.classifier = nn.Linear(densenet.classifier.in_features, 2)
            loaded_model = decompress_and_load_model_v3(path, densenet, device)
            if loaded_model:
                loaded_models['densenet'] = loaded_model
                st.success("✅ DenseNet-121 loaded successfully!")
                break
    else:
        st.warning(f"⚠️ DenseNet not found. Checked: {[os.path.basename(p) for p in model_configs['densenet']['paths']]}")
    
    # Load ViT
    st.markdown("---")
    st.markdown("#### 🔄 Loading Vision Transformer...")
    try:
        import timm
        for path in model_configs['vit']['paths']:
            if os.path.exists(path):
                st.success(f"✅ Found: {os.path.basename(path)}")
                vit = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
                loaded_model = decompress_and_load_model_v3(path, vit, device)
                if loaded_model:
                    loaded_models['vit'] = loaded_model
                    st.success("✅ Vision Transformer loaded successfully!")
                    break
        else:
            st.warning(f"⚠️ ViT not found. Checked: {[os.path.basename(p) for p in model_configs['vit']['paths']]}")
    except ImportError:
        st.error("⚠️ timm library not installed. Install with: pip install timm")
    
    # Load EfficientNet
    st.markdown("---")
    st.markdown("#### 🔄 Loading EfficientNet-B0...")
    for path in model_configs['efficientnet']['paths']:
        if os.path.exists(path):
            st.success(f"✅ Found: {os.path.basename(path)}")
            efficientnet = model_configs['efficientnet']['architecture']()
            efficientnet.classifier[1] = nn.Linear(efficientnet.classifier[1].in_features, 2)
            loaded_model = decompress_and_load_model_v3(path, efficientnet, device)
            if loaded_model:
                loaded_models['efficientnet'] = loaded_model
                st.success("✅ EfficientNet-B0 loaded successfully!")
                break
    else:
        st.warning(f"⚠️ EfficientNet not found. Checked: {[os.path.basename(p) for p in model_configs['efficientnet']['paths']]}")
    
    # Load ensemble models
    st.markdown("---")
    st.markdown("#### 🔄 Loading ensemble models...")
    
    ensemble_files = {
        'meta_model.pkl': 'Meta Model',
        'xgb_meta_model.pkl': 'XGBoost Meta',
        'cnn_stacking_logistic.pkl': 'Stacking Logistic',
        'cnn_stacking_ensemble_xgb_model.pkl': 'Stacking XGBoost',
        'cnn_aggregator_ensemble_predictions.pkl': 'CNN Aggregator',
        'ensemble_metrics.pkl': 'Metrics'
    }
    
    for filename, display_name in ensemble_files.items():
        filepath = os.path.join(models_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    loaded_models[filename.replace('.pkl', '')] = pickle.load(f)
                st.success(f"✅ {display_name} loaded")
            except Exception as e:
                st.warning(f"⚠️ {display_name} failed: {str(e)[:60]}")
    
    # Summary
    st.markdown("---")
    st.markdown("### 📊 Loading Summary")
    
    cnn_models = [k for k in loaded_models.keys() if k in ['resnet', 'densenet', 'vit', 'efficientnet']]
    ensemble_models = [k for k in loaded_models.keys() if k not in cnn_models]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("🤖 CNN Models Loaded", f"{len(cnn_models)}/4")
        if cnn_models:
            st.success(f"Loaded: {', '.join(cnn_models)}")
        missing = [m for m in ['resnet', 'densenet', 'vit', 'efficientnet'] if m not in cnn_models]
        if missing:
            st.warning(f"Missing: {', '.join(missing)}")
    
    with col2:
        st.metric("🔗 Ensemble Components", len(ensemble_models))
        if ensemble_models:
            st.success(f"Loaded: {', '.join([e.replace('_', ' ').title() for e in ensemble_models[:3]])}")
    
    return loaded_models

# ==================== IMAGE PREPROCESSING & ANALYSIS ====================

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
    img_array = preprocess_for_flood_detection(image)
    if img_array is None:
        return None
    
    # Convert to tensor for PyTorch models
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    # Get predictions from each CNN model
    with torch.no_grad():
        for model_name in ['resnet', 'densenet', 'vit', 'efficientnet']:
            if model_name in models_dict:
                try:
                    output = models_dict[model_name](img_tensor)
                    prob = torch.softmax(output, dim=1)[0, 1].cpu().numpy()
                    predictions[model_name] = float(prob)
                    cnn_features.append(prob)
                except Exception as e:
                    st.warning(f"⚠️ Error with {model_name}: {str(e)}")
    
    # Try to use ensemble models if CNN features are available
    if len(cnn_features) > 0 and len(cnn_features) == 4:  # All 4 CNNs loaded
        cnn_feature_vector = np.array(cnn_features).reshape(1, -1)
        
        # Try meta models
        for meta_name in ['meta_model', 'xgb_meta_model']:
            if meta_name in models_dict:
                try:
                    meta_pred = models_dict[meta_name].predict(cnn_feature_vector)[0]
                    predictions[f"{meta_name}_ensemble"] = float(meta_pred)
                except Exception as e:
                    pass  # Silently skip if incompatible
        
        # Try stacking models
        for stack_name in ['cnn_stacking_logistic', 'cnn_stacking_ensemble_xgb_model']:
            if stack_name in models_dict:
                try:
                    stack_pred = models_dict[stack_name].predict(cnn_feature_vector)[0]
                    predictions[f"{stack_name}_ensemble"] = float(stack_pred)
                except Exception as e:
                    pass  # Silently skip if incompatible
    
    # Adjust predictions based on context
    if predictions:
        ensemble_pred = np.mean(list(predictions.values()))
        
        # Apply context-based adjustment
        if context['fire_percentage'] > 10:
            ensemble_pred *= 0.3  # Heavily reduce if fire detected
        elif context['vegetation_percentage'] > 50 and context['water_percentage'] < 15:
            ensemble_pred *= 0.5  # Reduce if mostly vegetation
        elif context['water_percentage'] > 25:
            ensemble_pred = min(ensemble_pred * 1.2, 1.0)  # Slightly increase if water detected
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
        with st.spinner("🔄 Downloading pretrained models..."):
            path = kagglehub.model_download("subhojeetroy01/flood-prediction-models-performance-comparison")
            st.success(f"✅ Models downloaded to: {path}")
                    
            flood_files = [os.path.join(root, file) for root, dirs, files in os.walk(path_tabular) for file in files if file.endswith('.csv')]
            if flood_files:
                df_flood = pd.read_csv(flood_files[0])
                st.success(f"✅ Loaded flood prediction dataset with {len(df_flood)} records")
            else:
                st.error("❌ No CSV files found in flood prediction dataset")
                return None, []
                
            sat_files = []
            st.info("Searching for satellite images...")
            
            for root, dirs, files in os.walk(path_sat):
                for file in files:
                    if file.lower().endswith(('.tif', '.png', '.jpg', '.jpeg')):
                        sat_files.append(os.path.join(root, file))
            
            if sat_files:
                st.success(f"✅ Found {len(sat_files)} satellite images!")
            else:
                st.warning("⚠️ No satellite images found.")
            
            return df_flood, sat_files
            
    except Exception as e:
        st.error(f"❌ Error loading datasets: {str(e)}")
        return None, []

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
                <li>⚙️ 12 Pre-trained ML algorithms</li>
                <li>🛰️ Ensemble deep learning models</li>
                <li>📊 Context-aware flood detection</li>
                <li>🎯 Fire/vegetation filtering</li>
                <li>📈 Interactive visualizations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Dataset Loading")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Load from Kaggle", type="primary", key="load_kaggle"):
            df_flood, sat_files = load_datasets_from_kaggle()
            if df_flood is not None:
                st.session_state.df_flood = df_flood
                st.session_state.sat_files = sat_files
                st.session_state.dataset_loaded = True
                st.rerun()
    
    with col2:
        if st.button("📊 Use Sample Data", type="secondary", key="load_sample"):
            st.session_state.df_flood = create_sample_data()
            st.session_state.sat_files = []
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
        st.markdown(f"""
        <div class="metric-container">
            <h3>🛰️ Images</h3>
            <h2>{len(st.session_state.sat_files)}</h2>
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
    st.markdown("### ⚙️ Load Pre-trained Models")
    
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
                
                # Display PCA info
                st.info(f"📊 PCA applied: Explained variance = {pca.explained_variance_ratio_.sum():.2%}")
            else:
                st.error("❌ No models were loaded. Check the directory path.")
    
    if st.session_state.models_trained:
        st.markdown("#### 📊 Loaded Models Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
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
        
        with col2:
            st.info("**Model Performance Legend:**")
            st.markdown("""
            - 🟢 Excellent (Accuracy > 85%)
            - 🟡 Good (Accuracy > 70%)
            - 🟠 Fair (Accuracy < 70%)
            
            **Note:** Missing models like Gradient Boosting won't affect predictions - the ensemble uses available models.
            """)

# ==================== PAGE: PREDICTIONS ====================

elif page == "🔮 Predictions":
    st.markdown("### 🔮 Flood Risk Predictions")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please load models first from the Model Training page")
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

# ==================== PAGE: SATELLITE ANALYSIS ====================

elif page == "🛰️ Satellite Analysis":
    st.markdown("### 🛰️ Deep Learning Models")
    
    st.markdown("""
    Load pre-trained deep learning models for satellite imagery analysis:
    - **ResNet-50**: Deep residual network (.pth or .pkl)
    - **DenseNet-121**: Densely connected network (.pth or .pkl)
    - **Vision Transformer (ViT)**: Attention-based model (.pth or .pkl)
    - **EfficientNet-B0**: Efficient convolutional network (.pth or .pkl)
    - **Ensemble Models**: Meta-learners and stacking models
    
    ℹ️ The app will automatically convert .pth files to .pkl format for better compatibility.
    """)
    
    st.markdown("#### 📁 Load Deep Learning Models")
    
    models_dir = st.text_input("DL models directory path:", value="pretrained_models")
    
    
    
    if st.button("🔄 Load DL Models", type="primary"):
        with st.spinner("Loading and converting models..."):
            # Use the new v3 function
            loaded_models = load_pretrained_dl_models_v3(models_dir)
            
            if loaded_models:
                st.session_state.ensemble_models = loaded_models
                st.session_state.models_loaded = True
                st.success(f"✅ Successfully loaded {len(loaded_models)} model components!")
            else:
                st.error("❌ No models were loaded.")
    
    # ... rest of the page code remains the same
    
    if st.session_state.models_loaded:
        st.markdown("#### 📊 Loaded Components")
        
        model_info = []
        cnn_count = 0
        ensemble_count = 0
        
        for model_name in st.session_state.ensemble_models.keys():
            if model_name in ['resnet', 'densenet', 'vit', 'efficientnet']:
                model_type = '🤖 CNN'
                cnn_count += 1
            else:
                model_type = '🔗 Ensemble'
                ensemble_count += 1
            
            model_info.append({
                'Component': model_name,
                'Type': model_type,
                'Status': '✅ Ready'
            })
        
        if model_info:
            st.dataframe(pd.DataFrame(model_info), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**🤖 CNN Models:** {cnn_count}\n\nThese models process raw images directly")
            with col2:
                st.info(f"**🔗 Ensemble Models:** {ensemble_count}\n\nThese combine CNN predictions")
            
            if cnn_count < 4:
                st.warning(f"⚠️ Only {cnn_count}/4 CNN models loaded. Ensemble models may not work optimally.")
                st.info("💡 Ensemble models require all 4 CNN models (ResNet, DenseNet, ViT, EfficientNet) to generate predictions.")
            else:
                st.success("✅ All CNN models loaded! Ensemble models can now combine their predictions.")
    else:
        st.info("👆 Click 'Load DL Models' to load compressed models")
        
        st.markdown("#### 📋 Required Files")
        st.markdown("""
        **In `pretrained_models/` directory:**
        
        **Compressed PyTorch Models (.pth.gz):**
        - `model_compressed_resnet_model_checkpoint.pth.gz`
        - `model_compressed_densenet_model_checkpoint.pth.gz`
        - `model_compressed.pth.gz` (ViT)
        
        **Uncompressed PyTorch Model:**
        - `efficientnet_model_checkpoint.pth`
        
        **Ensemble Models (.pkl):**
        - `meta_model.pkl`
        - `xgb_meta_model.pkl`
        - `cnn_stacking_logistic.pkl`
        - `cnn_stacking_ensemble_xgb_model.pkl`
        - `cnn_aggregator_ensemble_predictions.pkl`
        - `ensemble_metrics.pkl`
        
        **In `Saved_Model/` directory:**
        - All tabular ML models (.pkl files)
        """)

# ==================== PAGE: IMAGE FLOOD DETECTION ====================

elif page == "🖼️ Image Flood Detection":
    st.markdown("### 🖼️ Advanced Flood Detection")
    
    if not st.session_state.models_loaded:
        st.warning("⚠️ Deep learning models not loaded from 'Satellite Analysis'")
        st.info("✅ **You can still use context-aware analysis!**")
        st.markdown("""
        The app includes intelligent context analysis that works without deep learning models:
        - 🔥 **Fire detection** - Identifies fire/heat signatures
        - 🌿 **Vegetation analysis** - Detects green vegetation
        - 💧 **Water detection** - Identifies water bodies
        - 📊 **Smart classification** - Rule-based flood assessment
        
        Upload an image below to try it!
        """)
    
    st.markdown("""
    Upload a satellite or aerial image for flood detection. The system includes:
    - 🔥 Fire detection to avoid false positives
    - 🌿 Vegetation analysis
    - 💧 Water body detection
    - 🤖 Deep learning ensemble predictions (if models loaded)
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
            if st.session_state.models_loaded:
                st.markdown("#### 🤖 Deep Learning Analysis")
                
                # Show which models will be used
                available_cnns = [m for m in ['resnet', 'densenet', 'vit', 'efficientnet'] if m in st.session_state.ensemble_models]
                available_ensembles = [m for m in st.session_state.ensemble_models.keys() if m not in ['resnet', 'densenet', 'vit', 'efficientnet']]
                
                with st.expander("ℹ️ Available Models for Prediction"):
                    st.markdown(f"""
                    **🤖 CNN Models Ready:** {len(available_cnns)}/4
                    - {', '.join(available_cnns) if available_cnns else 'None'}
                    
                    **🔗 Ensemble Models Ready:** {len(available_ensembles)}
                    - {', '.join(available_ensembles[:3]) if available_ensembles else 'None'}
                    
                    **Note:** Ensemble models require all 4 CNN models to be loaded. If fewer than 4 CNNs are available, only CNN predictions will be shown.
                    """)
                
                with st.spinner("Running ensemble predictions..."):
                    result = predict_with_ensemble(image, st.session_state.ensemble_models)
                
                if result and not result['rejected']:
                    predictions = result['predictions']
                    ensemble_pred = result['ensemble_pred']
                    
                    # Show loading status
                    cnn_preds = {k: v for k, v in predictions.items() if k in ['resnet', 'densenet', 'vit', 'efficientnet']}
                    ensemble_preds = {k: v for k, v in predictions.items() if k not in ['resnet', 'densenet', 'vit', 'efficientnet']}
                    
                    if len(cnn_preds) < 4:
                        missing_cnns = [m for m in ['resnet', 'densenet', 'vit', 'efficientnet'] if m not in cnn_preds]
                        st.warning(f"⚠️ {len(missing_cnns)} CNN model(s) not loaded: {', '.join(missing_cnns)}")
                        st.info("💡 These models failed to load during the 'Satellite Analysis' step. Check the error messages there.")
                    
                    if len(ensemble_preds) == 0 and len(cnn_preds) < 4:
                        st.info("ℹ️ Ensemble models require all 4 CNNs. Showing CNN predictions only.")
                    
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
                        
                        if not cnn_preds and not ensemble_preds:
                            st.warning("No model predictions available")
                            st.info(f"Total models available: {len(predictions)}")
                    
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
                    st.markdown("#### 📊 Model Comparison")
                    
                    if len(predictions) > 1:
                        # Separate predictions for better visualization
                        cnn_preds = {k: v for k, v in predictions.items() if k in ['resnet', 'densenet', 'vit', 'efficientnet']}
                        ensemble_preds = {k: v for k, v in predictions.items() if k not in ['resnet', 'densenet', 'vit', 'efficientnet']}
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Combined bar chart
                            model_names = list(predictions.keys())
                            model_preds = [predictions[m] * 100 for m in model_names]
                            
                            # Create color coding for different model types
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
                            # Radar chart for model agreement
                            if len(predictions) >= 3:
                                fig_radar = go.Figure()
                                
                                fig_radar.add_trace(go.Scatterpolar(
                                    r=model_preds,
                                    theta=[name[:15] for name in model_names],  # Truncate long names
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
                            else:
                                # Show box plot if fewer models
                                fig_box = go.Figure()
                                fig_box.add_trace(go.Box(
                                    y=model_preds,
                                    name='Predictions',
                                    marker_color='rgb(102, 126, 234)',
                                    boxmean='sd'
                                ))
                                fig_box.update_layout(
                                    title="📊 Prediction Distribution",
                                    yaxis_title="Flood Probability (%)"
                                )
                                st.plotly_chart(fig_box, use_container_width=True)
                        
                        # Detailed breakdown
                        st.markdown("##### 📈 Prediction Statistics")
                        stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)
                        
                        with stat_col1:
                            st.metric("Mean", f"{np.mean(model_preds):.1f}%")
                        with stat_col2:
                            st.metric("Std Dev", f"{np.std(model_preds):.1f}%")
                        with stat_col3:
                            st.metric("Min", f"{np.min(model_preds):.1f}%")
                        with stat_col4:
                            st.metric("Max", f"{np.max(model_preds):.1f}%")
                        with stat_col5:
                            st.metric("Range", f"{np.ptp(model_preds):.1f}%")
                        
                        # Show model agreement analysis
                        std_dev = np.std(model_preds)
                        if std_dev < 10:
                            st.success(f"✅ **High Agreement** - Models show strong consensus (σ={std_dev:.1f}%)")
                        elif std_dev < 20:
                            st.info(f"ℹ️ **Moderate Agreement** - Models show reasonable consensus (σ={std_dev:.1f}%)")
                        else:
                            st.warning(f"⚠️ **Low Agreement** - Models show significant variation (σ={std_dev:.1f}%)")
                        
                        # Detailed predictions table
                        with st.expander("📋 View Detailed Predictions Table"):
                            pred_table = pd.DataFrame({
                                'Model': model_names,
                                'Type': colors,
                                'Probability': [f"{p:.2f}%" for p in model_preds],
                                'Deviation from Mean': [f"{p - np.mean(model_preds):.2f}%" for p in model_preds]
                            })
                            st.dataframe(pred_table, use_container_width=True)
                    
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
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please load models first from Model Training page")
        st.stop()
    
    results = st.session_state.model_results
    
    st.markdown("#### 🏆 Model Performance")
    
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
    
    # Multiple comparison charts
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
    
    # Additional metrics visualization
    st.markdown("#### 📉 Detailed Metrics Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Precision vs Recall scatter
        fig_scatter = px.scatter(
            perf_df,
            x='Precision',
            y='Recall',
            size='F1 Score',
            color='Accuracy',
            hover_data=['Model'],
            title='🎯 Precision vs Recall',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Error metrics
        fig_error = go.Figure()
        fig_error.add_trace(go.Bar(
            name='RMSE',
            x=perf_df['Model'],
            y=perf_df['RMSE'],
            marker_color='indianred'
        ))
        fig_error.add_trace(go.Bar(
            name='MAE',
            x=perf_df['Model'],
            y=perf_df['MAE'],
            marker_color='lightsalmon'
        ))
        fig_error.update_layout(
            title='📉 Error Metrics Comparison',
            barmode='group',
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_error, use_container_width=True)

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

if st.session_state.models_loaded:
    st.sidebar.success("✅ DL Models Loaded")
else:
    st.sidebar.error("❌ DL Models Not Loaded")

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info("""
🌊 **FloodSentinel**
- 12 Pre-trained ML models
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
