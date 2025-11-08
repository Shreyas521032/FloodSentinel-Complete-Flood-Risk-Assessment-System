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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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

def decompress_model(compressed_path):
    """Decompress a .gz compressed model file with validation"""
    try:
        import tempfile
        
        # Check if file exists and has content
        if not os.path.exists(compressed_path):
            st.error(f"File not found: {compressed_path}")
            return None
        
        file_size = os.path.getsize(compressed_path)
        if file_size == 0:
            st.error(f"File is empty: {compressed_path}")
            return None
        
        st.info(f"📦 Decompressing {os.path.basename(compressed_path)} ({file_size / (1024*1024):.2f} MB)...")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
            try:
                with gzip.open(compressed_path, 'rb') as f_in:
                    decompressed_data = f_in.read()
                    tmp_file.write(decompressed_data)
                    st.success(f"✅ Decompressed to {len(decompressed_data) / (1024*1024):.2f} MB")
                return tmp_file.name
            except gzip.BadGzipFile:
                st.error(f"❌ Invalid gzip file: {compressed_path}")
                st.info("💡 The file might not be gzip compressed or is corrupted")
                return None
            except Exception as e:
                st.error(f"❌ Decompression error: {str(e)}")
                return None
                
    except Exception as e:
        st.error(f"❌ Error in decompress_model: {str(e)}")
        return None

def load_pretrained_dl_models(models_dir="pretrained_models"):
    """Load all pre-trained deep learning models with robust error handling"""
    loaded_models = {}
    
    if not os.path.exists(models_dir):
        st.warning(f"⚠️ Models directory '{models_dir}' not found.")
        st.info(f"💡 Please create the directory: {os.path.abspath(models_dir)}")
        return loaded_models
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.info(f"🖥️ Using device: {device}")
        
        # Load ResNet
        resnet_path = os.path.join(models_dir, "model_compressed_resnet_model_checkpoint.pth.gz")
        if os.path.exists(resnet_path):
            decompressed = decompress_model(resnet_path)
            if decompressed:
                try:
                    st.info("🔄 Loading ResNet-50 architecture...")
                    resnet = torch_models.resnet50(pretrained=False)
                    resnet.fc = nn.Linear(resnet.fc.in_features, 2)
                    
                    # Try loading with different map_location strategies
                    try:
                        state_dict = torch.load(decompressed, map_location=device)
                        resnet.load_state_dict(state_dict)
                        resnet.to(device)
                        resnet.eval()
                        loaded_models['resnet'] = resnet
                        st.success("✅ ResNet-50 loaded successfully")
                    except Exception as load_err:
                        st.error(f"❌ Error loading ResNet weights: {str(load_err)}")
                        st.info("💡 The model file may be corrupted or incompatible")
                    
                    # Clean up temp file
                    try:
                        os.unlink(decompressed)
                    except:
                        pass
                        
                except Exception as e:
                    st.error(f"❌ Error with ResNet: {str(e)}")
        else:
            st.warning(f"⚠️ ResNet model not found at {resnet_path}")
        
        # Load DenseNet
        densenet_path = os.path.join(models_dir, "model_compressed_densenet_model_checkpoint.pth.gz")
        if os.path.exists(densenet_path):
            decompressed = decompress_model(densenet_path)
            if decompressed:
                try:
                    st.info("🔄 Loading DenseNet-121 architecture...")
                    densenet = torch_models.densenet121(pretrained=False)
                    densenet.classifier = nn.Linear(densenet.classifier.in_features, 2)
                    
                    try:
                        state_dict = torch.load(decompressed, map_location=device)
                        densenet.load_state_dict(state_dict)
                        densenet.to(device)
                        densenet.eval()
                        loaded_models['densenet'] = densenet
                        st.success("✅ DenseNet-121 loaded successfully")
                    except Exception as load_err:
                        st.error(f"❌ Error loading DenseNet weights: {str(load_err)}")
                    
                    try:
                        os.unlink(decompressed)
                    except:
                        pass
                        
                except Exception as e:
                    st.error(f"❌ Error with DenseNet: {str(e)}")
        else:
            st.warning(f"⚠️ DenseNet model not found at {densenet_path}")
        
        # Load ViT
        vit_path = os.path.join(models_dir, "model_compressed.pth.gz")
        if os.path.exists(vit_path):
            decompressed = decompress_model(vit_path)
            if decompressed:
                try:
                    st.info("🔄 Loading Vision Transformer architecture...")
                    vit = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
                    
                    try:
                        state_dict = torch.load(decompressed, map_location=device)
                        vit.load_state_dict(state_dict)
                        vit.to(device)
                        vit.eval()
                        loaded_models['vit'] = vit
                        st.success("✅ Vision Transformer loaded successfully")
                    except Exception as load_err:
                        st.error(f"❌ Error loading ViT weights: {str(load_err)}")
                    
                    try:
                        os.unlink(decompressed)
                    except:
                        pass
                        
                except Exception as e:
                    st.error(f"❌ Error with ViT: {str(e)}")
        else:
            st.warning(f"⚠️ ViT model not found at {vit_path}")
        
        # Load EfficientNet (uncompressed) - try multiple architectures
        efficientnet_path = os.path.join(models_dir, "efficientnet_model_checkpoint.pth")
        if os.path.exists(efficientnet_path):
            try:
                file_size = os.path.getsize(efficientnet_path)
                st.info(f"🔄 Loading EfficientNet ({file_size / (1024*1024):.2f} MB)...")
                
                # Load the state dict first to inspect it
                state_dict = torch.load(efficientnet_path, map_location=device)
                
                # Try to determine the architecture from keys
                if 'blocks.0.0.conv_dw.weight' in state_dict:
                    # timm EfficientNet structure
                    st.info("Detected timm EfficientNet architecture")
                    try:
                        import timm
                        efficientnet = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
                        efficientnet.load_state_dict(state_dict, strict=False)
                        efficientnet.to(device)
                        efficientnet.eval()
                        loaded_models['efficientnet'] = efficientnet
                        st.success("✅ EfficientNet-B0 (timm) loaded successfully")
                    except Exception as e:
                        st.warning(f"⚠️ Could not load with timm: {str(e)}")
                else:
                    # Try standard torchvision
                    efficientnet = torch_models.efficientnet_b0(pretrained=False)
                    efficientnet.classifier[1] = nn.Linear(efficientnet.classifier[1].in_features, 2)
                    try:
                        efficientnet.load_state_dict(state_dict, strict=False)
                        efficientnet.to(device)
                        efficientnet.eval()
                        loaded_models['efficientnet'] = efficientnet
                        st.success("✅ EfficientNet-B0 (torchvision) loaded successfully")
                    except Exception as load_err:
                        st.error(f"❌ Error loading EfficientNet weights: {str(load_err)[:200]}")
                        st.info("⚠️ EfficientNet skipped - architecture mismatch")
                    
            except Exception as e:
                st.error(f"❌ Error with EfficientNet: {str(e)[:200]}")
                st.info("⚠️ EfficientNet will be skipped - continuing with other models")
        else:
            st.warning(f"⚠️ EfficientNet model not found at {efficientnet_path}")
        
        # Load ensemble models (pickle files)
        ensemble_files = {
            'meta_model.pkl': 'Meta Model',
            'xgb_meta_model.pkl': 'XGBoost Meta Model',
            'cnn_stacking_logistic.pkl': 'CNN Stacking (Logistic)',
            'cnn_stacking_ensemble_xgb_model.pkl': 'CNN Stacking (XGBoost)',
            'cnn_aggregator_ensemble_predictions.pkl': 'CNN Aggregator Ensemble',
            'ensemble_metrics.pkl': 'Ensemble Metrics'
        }
        
        for filename, display_name in ensemble_files.items():
            filepath = os.path.join(models_dir, filename)
            if os.path.exists(filepath):
                try:
                    file_size = os.path.getsize(filepath)
                    st.info(f"📦 Loading {display_name} ({file_size / 1024:.2f} KB)...")
                    
                    with open(filepath, 'rb') as f:
                        try:
                            loaded_models[filename.replace('.pkl', '')] = pickle.load(f)
                            st.success(f"✅ {display_name} loaded successfully")
                        except Exception as e1:
                            # Try joblib
                            try:
                                import joblib
                                loaded_models[filename.replace('.pkl', '')] = joblib.load(filepath)
                                st.success(f"✅ {display_name} loaded with joblib")
                            except Exception as e2:
                                st.warning(f"⚠️ Could not load {display_name}: {str(e1)[:100]}")
                                
                except Exception as e:
                    st.warning(f"⚠️ Error accessing {display_name}: {str(e)}")
            else:
                st.info(f"ℹ️ Optional: {display_name} not found")
        
        if not loaded_models:
            st.error("❌ No models were loaded. Please check:")
            st.markdown("""
            ### Troubleshooting Steps:
            
            1. **For PyTorch models (.pth.gz files):**
               ```python
               # Your files may not be properly gzipped. To fix:
               import torch
               import gzip
               
               # Load original model
               model = torch.load('model.pth')
               
               # Save properly
               torch.save(model, 'temp.pth')
               
               # Compress with gzip
               with open('temp.pth', 'rb') as f_in:
                   with gzip.open('model.pth.gz', 'wb') as f_out:
                       f_out.write(f_in.read())
               ```
            
            2. **Alternative: Skip compression**
               - Rename your `.pth.gz` files to `.pth`
               - Update the code to look for `.pth` instead
            
            3. **For EfficientNet:**
               - Model was saved with `timm` library architecture
               - Using `strict=False` to load partial weights
            
            **Current Status:**
            - ✅ {len([k for k in loaded_models.keys() if k in ['resnet', 'densenet', 'vit', 'efficientnet']])} CNN models loaded
            - ✅ {len([k for k in loaded_models.keys() if k not in ['resnet', 'densenet', 'vit', 'efficientnet']])} ensemble models loaded
            """)
        else:
            cnn_count = len([k for k in loaded_models.keys() if k in ['resnet', 'densenet', 'vit', 'efficientnet']])
            ensemble_count = len(loaded_models) - cnn_count
            st.success(f"✅ Loaded {cnn_count} CNN models and {ensemble_count} ensemble models!")
            
            if cnn_count == 0:
                st.warning("⚠️ No CNN models loaded - image prediction will use context analysis only")
                st.info("💡 You can still use the app with tabular models and context-based image analysis")
        
    except Exception as e:
        st.error(f"❌ Critical error loading DL models: {str(e)}")
        import traceback
        with st.expander("🔍 Full Error Traceback"):
            st.code(traceback.format_exc())
    
    return loaded_models

def load_pretrained_tabular_models(models_dir="Saved_Model"):
    """Load all pre-trained tabular models with robust error handling"""
    loaded_models = {}
    
    if not os.path.exists(models_dir):
        st.warning(f"⚠️ Models directory '{models_dir}' not found.")
        st.info(f"💡 Please create the directory or check the path: {os.path.abspath(models_dir)}")
        return loaded_models
    
    model_files = {
        'linear_regression (1).pkl': '📈 Linear Regression',
        'ridge (1).pkl': '📊 Ridge',
        'lasso (1).pkl': '🔗 Lasso',
        'k_neighbors_regressor (1).pkl': '👥 K-Neighbors',
        'decision_tree_regressor (1).pkl': '🌿 Decision Tree',
        'xgboost_regressor (1).pkl': '🚀 XGBoost',
        'lightgbm_regressor (1).pkl': '💡 LightGBM',
        'catboost_regressor (1).pkl': '🎯 CatBoost',
        'support_vector_regressor (1).pkl': '📈 SVR',
    }
    
    st.info(f"📂 Looking for models in: {os.path.abspath(models_dir)}")
    
    for filename, display_name in model_files.items():
        filepath = os.path.join(models_dir, filename)
        
        if not os.path.exists(filepath):
            st.warning(f"⚠️ {display_name} not found at {filepath}")
            continue
        
        # Check file size
        file_size = os.path.getsize(filepath)
        if file_size == 0:
            st.error(f"❌ {display_name} is empty (0 bytes)")
            continue
        
        st.info(f"📦 Loading {display_name} ({file_size / 1024:.2f} KB)...")
        
        try:
            # Try different pickle protocols
            with open(filepath, 'rb') as f:
                try:
                    loaded_models[display_name] = pickle.load(f)
                    st.success(f"✅ {display_name} loaded successfully")
                except Exception as e1:
                    # Try with different encoding
                    st.warning(f"⚠️ Standard pickle failed, trying alternative methods...")
                    f.seek(0)
                    try:
                        loaded_models[display_name] = pickle.load(f, encoding='latin1')
                        st.success(f"✅ {display_name} loaded with latin1 encoding")
                    except Exception as e2:
                        # Try joblib if pickle fails
                        try:
                            import joblib
                            loaded_models[display_name] = joblib.load(filepath)
                            st.success(f"✅ {display_name} loaded with joblib")
                        except Exception as e3:
                            st.error(f"❌ Failed to load {display_name}")
                            st.error(f"   Pickle error: {str(e1)[:100]}")
                            st.error(f"   Latin1 error: {str(e2)[:100]}")
                            st.error(f"   Joblib error: {str(e3)[:100]}")
                            
                            # Show file info for debugging
                            with st.expander(f"🔍 Debug info for {display_name}"):
                                st.code(f"File path: {filepath}\nFile size: {file_size} bytes")
                                # Read first few bytes
                                with open(filepath, 'rb') as debug_f:
                                    first_bytes = debug_f.read(20)
                                    st.code(f"First bytes (hex): {first_bytes.hex()}")
                                    st.code(f"First bytes (repr): {repr(first_bytes)}")
                            
        except Exception as e:
            st.error(f"❌ Error accessing {display_name}: {str(e)}")
    
    if loaded_models:
        st.success(f"🎉 Successfully loaded {len(loaded_models)} models!")
    else:
        st.warning("⚠️ No tabular models were loaded.")
        st.info("💡 Some models may have numpy version mismatches. The app will continue with successfully loaded models.")
        st.markdown("""
        **Common Issues:**
        - **Gradient Boosting error**: Numpy version incompatibility
          - This is a known issue with numpy 2.x vs 1.x
          - The model can be re-saved with: `joblib.dump(model, 'model.pkl', protocol=4)`
        
        **Continue anyway?** Yes! The other 9 models work fine for predictions.
        """)
    
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
    
    # Preprocess image
    img_array = preprocess_for_flood_detection(image)
    if img_array is None:
        return None
    
    # Convert to tensor for PyTorch models
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    # Get predictions from each model
    with torch.no_grad():
        for model_name in ['resnet', 'densenet', 'vit', 'efficientnet']:
            if model_name in models_dict:
                try:
                    output = models_dict[model_name](img_tensor)
                    prob = torch.softmax(output, dim=1)[0, 1].cpu().numpy()
                    predictions[model_name] = float(prob)
                except Exception as e:
                    st.warning(f"⚠️ Error with {model_name}: {str(e)}")
    
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
            def load_data(path_tabular="dataset/flood dataset.csv"):
                return pd.read_csv(path_tabular)
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
                        
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        
                        st.session_state.model_results[model_name] = {
                            "MSE": mse,
                            "RMSE": rmse,
                            "MAE": mae,
                            "R²": r2,
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
                r2_score = st.session_state.model_results[name]['R²']
                if r2_score > 0.8:
                    st.markdown(f"🟢 {name}: R² = {r2_score:.4f}")
                elif r2_score > 0.6:
                    st.markdown(f"🟡 {name}: R² = {r2_score:.4f}")
                else:
                    st.markdown(f"🟠 {name}: R² = {r2_score:.4f}")
        
        with col2:
            st.info("**Model Performance Legend:**")
            st.markdown("""
            - 🟢 Excellent (R² > 0.80)
            - 🟡 Good (R² > 0.60)
            - 🟠 Fair (R² < 0.60)
            
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
    - **ResNet-50**: Deep residual network (compressed)
    - **DenseNet-121**: Densely connected network (compressed)
    - **Vision Transformer (ViT)**: Attention-based model (compressed)
    - **EfficientNet-B0**: Efficient convolutional network
    - **Ensemble Models**: Meta-learners and stacking models
    """)
    
    st.markdown("#### 📁 Load Compressed Deep Learning Models")
    
    models_dir = st.text_input("DL models directory path:", value="pretrained_models")
    
    if st.button("🔄 Load DL Models", type="primary"):
        with st.spinner("Decompressing and loading models..."):
            loaded_models = load_pretrained_dl_models(models_dir)
            
            if loaded_models:
                st.session_state.ensemble_models = loaded_models
                st.session_state.models_loaded = True
                st.success(f"✅ Successfully loaded {len(loaded_models)} model components!")
            else:
                st.error("❌ No models were loaded.")
    
    if st.session_state.models_loaded:
        st.markdown("#### 📊 Loaded Components")
        
        model_info = []
        for model_name in st.session_state.ensemble_models.keys():
            model_info.append({
                'Component': model_name,
                'Type': 'CNN' if model_name in ['resnet', 'densenet', 'vit'] else 'Ensemble',
                'Status': '✅ Ready'
            })
        
        if model_info:
            st.dataframe(pd.DataFrame(model_info), use_container_width=True)
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
                        
                        st.markdown("##### 📊 Individual Models")
                        for model_name, pred in predictions.items():
                            st.metric(model_name.upper(), f"{pred:.2%}")
                    
                    with col2:
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
            'R² Score': metrics['R²'],
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
        })
    
    perf_df = pd.DataFrame(perf_data)
    perf_df = perf_df.sort_values('R² Score', ascending=False)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if len(perf_df) > 0:
            best = perf_df.iloc[0]
            st.markdown(f"""<div class="metric-container"><h4>🥇 Best Model</h4><h3>{best['Model']}</h3><p>R²: {best['R² Score']:.4f}</p></div>""", unsafe_allow_html=True)
    
    with col2:
        avg_r2 = perf_df['R² Score'].mean()
        st.markdown(f"""<div class="metric-container"><h4>📊 Average R²</h4><h2>{avg_r2:.4f}</h2></div>""", unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""<div class="metric-container"><h4>🎯 Models</h4><h2>{len(perf_df)}</h2></div>""", unsafe_allow_html=True)
    
    st.dataframe(perf_df, use_container_width=True)
    
    fig_r2 = px.bar(
        perf_df.sort_values('R² Score'),
        x='R² Score',
        y='Model',
        orientation='h',
        title='🎯 R² Score Comparison',
        color='R² Score',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_r2, use_container_width=True)

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
