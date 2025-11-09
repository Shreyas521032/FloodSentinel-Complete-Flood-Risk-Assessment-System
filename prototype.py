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

# ==================== KAGGLE MODEL DOWNLOAD FUNCTIONS ====================

def download_models_from_kaggle_kernel(kernel_slug, output_dir="pretrained_models"):
    """
    Download model outputs from a Kaggle kernel
    
    Args:
        kernel_slug: Kaggle kernel slug (e.g., 'username/kernel-name')
        output_dir: Directory to save downloaded models
    """
    import subprocess
    import os
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        st.info(f"📥 Downloading models from Kaggle kernel: {kernel_slug}")
        st.info(f"📂 Saving to: {os.path.abspath(output_dir)}")
        
        # Run kaggle command
        cmd = f"kaggle kernels output {kernel_slug} -p {output_dir}"
        
        st.code(cmd, language="bash")
        
        with st.spinner("⏳ Downloading... This may take a few minutes..."):
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True
            )
        
        if result.returncode == 0:
            st.success("✅ Download completed successfully!")
            
            # List downloaded files
            files = os.listdir(output_dir)
            if files:
                st.success(f"✅ Downloaded {len(files)} files:")
                
                file_info = []
                for file in files:
                    filepath = os.path.join(output_dir, file)
                    size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    file_info.append({
                        'File': file,
                        'Size (MB)': f"{size_mb:.2f}",
                        'Type': '🤖 CNN Model' if any(x in file for x in ['resnet', 'densenet', 'vit', 'efficientnet']) else '🔗 Ensemble'
                    })
                
                st.dataframe(pd.DataFrame(file_info), use_container_width=True)
            else:
                st.warning("⚠️ No files found in output directory")
            
            return True, output_dir
        else:
            st.error(f"❌ Download failed!")
            st.error(f"Error: {result.stderr}")
            
            # Check for common issues
            if "401" in result.stderr or "authentication" in result.stderr.lower():
                st.warning("🔑 **Authentication Issue**")
                st.markdown("""
                Please ensure you have configured Kaggle API credentials:
                
                1. Go to Kaggle Account Settings: https://www.kaggle.com/settings
                2. Click "Create New API Token" to download `kaggle.json`
                3. Place it in `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\\Users\\<Username>\\.kaggle\\kaggle.json` (Windows)
                4. Run: `chmod 600 ~/.kaggle/kaggle.json` (Linux/Mac only)
                """)
            elif "404" in result.stderr or "not found" in result.stderr.lower():
                st.warning("🔍 **Kernel Not Found**")
                st.info(f"Please verify the kernel slug: `{kernel_slug}`")
                st.info("Format should be: `username/kernel-name`")
            
            return False, None
            
    except FileNotFoundError:
        st.error("❌ Kaggle CLI not found!")
        st.markdown("""
        **Please install Kaggle CLI:**
        
        ```bash
        pip install kaggle
        ```
        
        Then configure your API credentials as described above.
        """)
        return False, None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return False, None

def download_models_from_kaggle_dataset(dataset_slug, output_dir="pretrained_models"):
    """
    Download models from a Kaggle dataset
    
    Args:
        dataset_slug: Kaggle dataset slug (e.g., 'username/dataset-name')
        output_dir: Directory to save downloaded models
    """
    import subprocess
    import os
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        st.info(f"📥 Downloading from Kaggle dataset: {dataset_slug}")
        st.info(f"📂 Saving to: {os.path.abspath(output_dir)}")
        
        cmd = f"kaggle datasets download -d {dataset_slug} -p {output_dir} --unzip"
        
        st.code(cmd, language="bash")
        
        with st.spinner("⏳ Downloading... This may take several minutes..."):
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True
            )
        
        if result.returncode == 0:
            st.success("✅ Download completed successfully!")
            
            files = []
            for root, dirs, filenames in os.walk(output_dir):
                for file in filenames:
                    files.append(os.path.join(root, file))
            
            if files:
                st.success(f"✅ Downloaded {len(files)} files")
                
                file_info = []
                for filepath in files[:20]:  # Show first 20 files
                    size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    file_info.append({
                        'File': os.path.basename(filepath),
                        'Path': filepath,
                        'Size (MB)': f"{size_mb:.2f}"
                    })
                
                st.dataframe(pd.DataFrame(file_info), use_container_width=True)
                
                if len(files) > 20:
                    st.info(f"... and {len(files) - 20} more files")
            
            return True, output_dir
        else:
            st.error(f"❌ Download failed!")
            st.error(f"Error: {result.stderr}")
            return False, None
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return False, None

# ==================== DEEP LEARNING MODEL LOADING FUNCTIONS ====================

def try_decompress_and_load(compressed_path, device):
    """
    Try to decompress .gz file and load it directly as raw weights
    Returns state_dict or None
    """
    try:
        st.info(f"📦 Attempting to load: {os.path.basename(compressed_path)}")
        
        # Check if file exists
        if not os.path.exists(compressed_path):
            st.warning(f"⚠️ File not found: {compressed_path}")
            return None
        
        file_size = os.path.getsize(compressed_path)
        if file_size == 0:
            st.warning(f"⚠️ File is empty: {compressed_path}")
            return None
        
        st.info(f"   File size: {file_size / (1024*1024):.2f} MB")
        
        # Try to decompress
        try:
            with gzip.open(compressed_path, 'rb') as f:
                decompressed_data = f.read()
                st.info(f"   Decompressed: {len(decompressed_data) / (1024*1024):.2f} MB")
        except Exception as e:
            st.warning(f"⚠️ Decompression failed: {str(e)[:100]}")
            return None
        
        # Save to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
            tmp_file.write(decompressed_data)
            tmp_path = tmp_file.name
        
        # Try to load as PyTorch checkpoint
        try:
            state_dict = torch.load(tmp_path, map_location=device, weights_only=False)
            st.success(f"✅ Successfully loaded as PyTorch checkpoint")
            os.unlink(tmp_path)
            return state_dict
        except Exception as e1:
            st.warning(f"⚠️ Not a standard PyTorch checkpoint: {str(e1)[:100]}")
            
            # Try to load as pickle
            try:
                import pickle
                with open(tmp_path, 'rb') as f:
                    state_dict = pickle.load(f)
                st.success(f"✅ Successfully loaded as pickle file")
                os.unlink(tmp_path)
                return state_dict
            except Exception as e2:
                st.warning(f"⚠️ Not a pickle file: {str(e2)[:100]}")
                os.unlink(tmp_path)
                return None
    
    except Exception as e:
        st.error(f"❌ Error in try_decompress_and_load: {str(e)}")
        return None

def load_pretrained_dl_models(models_dir="pretrained_models"):
    """Load all pre-trained deep learning models with multiple fallback strategies"""
    loaded_models = {}
    
    if not os.path.exists(models_dir):
        st.warning(f"⚠️ Models directory '{models_dir}' not found.")
        st.info(f"💡 Please create the directory: {os.path.abspath(models_dir)}")
        return loaded_models
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.info(f"🖥️ Using device: {device}")
        
        # Strategy: Try multiple file patterns for each model
        model_patterns = {
            'resnet': [
                'model_compressed_resnet_model_checkpoint.pth.gz',
                'resnet_model_checkpoint.pth.gz',
                'resnet_model_checkpoint.pth',
                'resnet.pth.gz',
                'resnet.pth'
            ],
            'densenet': [
                'model_compressed_densenet_model_checkpoint.pth.gz',
                'densenet_model_checkpoint.pth.gz',
                'densenet_model_checkpoint.pth',
                'densenet.pth.gz',
                'densenet.pth'
            ],
            'vit': [
                'model_compressed.pth.gz',
                'vit_model_checkpoint.pth.gz',
                'vit_model_checkpoint.pth',
                'vit.pth.gz',
                'vit.pth'
            ],
            'efficientnet': [
                'efficientnet_model_checkpoint.pth',
                'efficientnet_model_checkpoint.pth.gz',
                'efficientnet.pth.gz',
                'efficientnet.pth'
            ]
        }
        
        # Load ResNet
        st.markdown("#### 🔄 Loading ResNet-50...")
        resnet_loaded = False
        for pattern in model_patterns['resnet']:
            resnet_path = os.path.join(models_dir, pattern)
            if os.path.exists(resnet_path):
                st.info(f"   Found: {pattern}")
                try:
                    # Create architecture
                    resnet = torch_models.resnet50(pretrained=False)
                    resnet.fc = nn.Linear(resnet.fc.in_features, 2)
                    
                    # Load weights
                    if pattern.endswith('.gz'):
                        state_dict = try_decompress_and_load(resnet_path, device)
                    else:
                        state_dict = torch.load(resnet_path, map_location=device, weights_only=False)
                    
                    if state_dict is not None:
                        # Handle different checkpoint formats
                        if isinstance(state_dict, dict):
                            if 'state_dict' in state_dict:
                                state_dict = state_dict['state_dict']
                            elif 'model_state_dict' in state_dict:
                                state_dict = state_dict['model_state_dict']
                        
                        resnet.load_state_dict(state_dict, strict=False)
                        resnet.to(device)
                        resnet.eval()
                        loaded_models['resnet'] = resnet
                        st.success("✅ ResNet-50 loaded successfully")
                        resnet_loaded = True
                        break
                except Exception as e:
                    st.warning(f"⚠️ Failed with {pattern}: {str(e)[:100]}")
                    continue
        
        if not resnet_loaded:
            st.error("❌ Could not load ResNet-50 from any file pattern")
        
        # Load DenseNet
        st.markdown("#### 🔄 Loading DenseNet-121...")
        densenet_loaded = False
        for pattern in model_patterns['densenet']:
            densenet_path = os.path.join(models_dir, pattern)
            if os.path.exists(densenet_path):
                st.info(f"   Found: {pattern}")
                try:
                    densenet = torch_models.densenet121(pretrained=False)
                    densenet.classifier = nn.Linear(densenet.classifier.in_features, 2)
                    
                    if pattern.endswith('.gz'):
                        state_dict = try_decompress_and_load(densenet_path, device)
                    else:
                        state_dict = torch.load(densenet_path, map_location=device, weights_only=False)
                    
                    if state_dict is not None:
                        if isinstance(state_dict, dict):
                            if 'state_dict' in state_dict:
                                state_dict = state_dict['state_dict']
                            elif 'model_state_dict' in state_dict:
                                state_dict = state_dict['model_state_dict']
                        
                        densenet.load_state_dict(state_dict, strict=False)
                        densenet.to(device)
                        densenet.eval()
                        loaded_models['densenet'] = densenet
                        st.success("✅ DenseNet-121 loaded successfully")
                        densenet_loaded = True
                        break
                except Exception as e:
                    st.warning(f"⚠️ Failed with {pattern}: {str(e)[:100]}")
                    continue
        
        if not densenet_loaded:
            st.error("❌ Could not load DenseNet-121 from any file pattern")
        
        # Load ViT
        st.markdown("#### 🔄 Loading Vision Transformer...")
        vit_loaded = False
        for pattern in model_patterns['vit']:
            vit_path = os.path.join(models_dir, pattern)
            if os.path.exists(vit_path):
                st.info(f"   Found: {pattern}")
                try:
                    vit = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
                    
                    if pattern.endswith('.gz'):
                        state_dict = try_decompress_and_load(vit_path, device)
                    else:
                        state_dict = torch.load(vit_path, map_location=device, weights_only=False)
                    
                    if state_dict is not None:
                        if isinstance(state_dict, dict):
                            if 'state_dict' in state_dict:
                                state_dict = state_dict['state_dict']
                            elif 'model_state_dict' in state_dict:
                                state_dict = state_dict['model_state_dict']
                        
                        vit.load_state_dict(state_dict, strict=False)
                        vit.to(device)
                        vit.eval()
                        loaded_models['vit'] = vit
                        st.success("✅ Vision Transformer loaded successfully")
                        vit_loaded = True
                        break
                except Exception as e:
                    st.warning(f"⚠️ Failed with {pattern}: {str(e)[:100]}")
                    continue
        
        if not vit_loaded:
            st.error("❌ Could not load ViT from any file pattern")
        
        # Load EfficientNet
        st.markdown("#### 🔄 Loading EfficientNet...")
        efficientnet_loaded = False
        for pattern in model_patterns['efficientnet']:
            efficientnet_path = os.path.join(models_dir, pattern)
            if os.path.exists(efficientnet_path):
                st.info(f"   Found: {pattern}")
                try:
                    if pattern.endswith('.gz'):
                        state_dict = try_decompress_and_load(efficientnet_path, device)
                    else:
                        state_dict = torch.load(efficientnet_path, map_location=device, weights_only=False)
                    
                    if state_dict is not None:
                        if isinstance(state_dict, dict):
                            if 'state_dict' in state_dict:
                                state_dict = state_dict['state_dict']
                            elif 'model_state_dict' in state_dict:
                                state_dict = state_dict['model_state_dict']
                        
                        # Try timm version first
                        try:
                            efficientnet = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
                            efficientnet.load_state_dict(state_dict, strict=False)
                            efficientnet.to(device)
                            efficientnet.eval()
                            loaded_models['efficientnet'] = efficientnet
                            st.success("✅ EfficientNet-B0 (timm) loaded successfully")
                            efficientnet_loaded = True
                            break
                        except:
                            # Try torchvision version
                            efficientnet = torch_models.efficientnet_b0(pretrained=False)
                            efficientnet.classifier[1] = nn.Linear(efficientnet.classifier[1].in_features, 2)
                            efficientnet.load_state_dict(state_dict, strict=False)
                            efficientnet.to(device)
                            efficientnet.eval()
                            loaded_models['efficientnet'] = efficientnet
                            st.success("✅ EfficientNet-B0 (torchvision) loaded successfully")
                            efficientnet_loaded = True
                            break
                except Exception as e:
                    st.warning(f"⚠️ Failed with {pattern}: {str(e)[:100]}")
                    continue
        
        if not efficientnet_loaded:
            st.error("❌ Could not load EfficientNet from any file pattern")
        
        # Load ensemble models (pickle files)
        st.markdown("#### 🔄 Loading Ensemble Models...")
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
        
        # Final summary
        cnn_count = len([k for k in loaded_models.keys() if k in ['resnet', 'densenet', 'vit', 'efficientnet']])
        ensemble_count = len([k for k in loaded_models.keys() if k not in ['resnet', 'densenet', 'vit', 'efficientnet']])
        
        if cnn_count == 0:
            st.error("❌ No CNN models loaded - image prediction will use context analysis only")
            st.info("💡 You can still use the app with tabular models and context-based analysis")
        else:
            st.success(f"✅ Loaded {cnn_count} CNN models and {ensemble_count} ensemble models!")
            
            if cnn_count < 4:
                st.warning(f"⚠️ Only {cnn_count}/4 CNN models loaded. Ensemble may not work optimally.")
        
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
    
    st.info(f"📂 Looking for models in: {os.path.abspath(models_dir)}")
    
    for filename, display_name in model_files.items():
        filepath = os.path.join(models_dir, filename)
        
        if not os.path.exists(filepath):
            st.warning(f"⚠️ {display_name} not found at {filepath}")
            continue
        
        file_size = os.path.getsize(filepath)
        if file_size == 0:
            st.error(f"❌ {display_name} is empty (0 bytes)")
            continue
        
        st.info(f"📦 Loading {display_name} ({file_size / 1024:.2f} KB)...")
        
        try:
            with open(filepath, 'rb') as f:
                try:
                    loaded_models[display_name] = pickle.load(f)
                    st.success(f"✅ {display_name} loaded successfully")
                except Exception as e1:
                    st.warning(f"⚠️ Standard pickle failed, trying alternative methods...")
                    f.seek(0)
                    try:
                        loaded_models[display_name] = pickle.load(f, encoding='latin1')
                        st.success(f"✅ {display_name} loaded with latin1 encoding")
                    except Exception as e2:
                        try:
                            import joblib
                            loaded_models[display_name] = joblib.load(filepath)
                            st.success(f"✅ {display_name} loaded with joblib")
                        except Exception as e3:
                            st.error(f"❌ Failed to load {display_name}")
                            
        except Exception as e:
            st.error(f"❌ Error accessing {display_name}: {str(e)}")
    
    if loaded_models:
        st.success(f"🎉 Successfully loaded {len(loaded_models)} models!")
    else:
        st.warning("⚠️ No tabular models were loaded.")
    
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
