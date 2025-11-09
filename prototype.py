import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
import os
import glob
import cv2
from PIL import Image
import time
import json
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models as torch_models, transforms
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
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE INITIALIZATION ====================
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'dataset_loaded' not in st.session_state:
    st.session_state.dataset_loaded = False
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}
if 'cnn_models' not in st.session_state:
    st.session_state.cnn_models = {}
if 'training_history' not in st.session_state:
    st.session_state.training_history = {}
if 'image_dataset' not in st.session_state:
    st.session_state.image_dataset = None
if 'flood_labels' not in st.session_state:
    st.session_state.flood_labels = {}

# ==================== DEVICE SETUP ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== IMAGE PREPROCESSING ====================
def create_preprocessing_pipeline(img_size=224):
    """Create image preprocessing pipeline"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# ==================== DATASET CLASS ====================
class FloodImageDataset(Dataset):
    """Custom Dataset for Flood Images"""
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
            # Try to open as regular image
            image = Image.open(img_path).convert('RGB')
        except:
            # If fails, create a dummy image
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ==================== MODEL CREATION FUNCTIONS ====================
def create_resnet50(num_classes=2):
    """Create ResNet-50 model"""
    model = torch_models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def create_densenet121(num_classes=2):
    """Create DenseNet-121 model"""
    model = torch_models.densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model

def create_efficientnet_b0(num_classes=2):
    """Create EfficientNet-B0 model"""
    model = timm.create_model('efficientnet_b0', pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model

def create_vit(num_classes=2):
    """Create Vision Transformer model"""
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    return model

# ==================== TRAINING FUNCTIONS ====================
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Handle different output formats
        if hasattr(outputs, 'logits'):
            outputs = outputs.logits
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits
            
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def train_model(model, train_loader, val_loader, model_name, num_epochs=10, lr=1e-4):
    """Train a model with progress tracking"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_placeholder = st.empty()
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Update progress
        progress = (epoch + 1) / num_epochs
        progress_bar.progress(progress)
        status_text.text(f"{model_name} - Epoch {epoch+1}/{num_epochs}")
        
        # Display metrics
        metrics_placeholder.markdown(f"""
        **Train Loss:** {train_loss:.4f} | **Train Acc:** {train_acc:.4f}  
        **Val Loss:** {val_loss:.4f} | **Val Acc:** {val_acc:.4f}
        """)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{model_name}_best.pth")
    
    status_text.text(f"✅ {model_name} training complete! Best Val Acc: {best_val_acc:.4f}")
    return model, history

# ==================== MAIN APPLICATION ====================
st.markdown('<h1 class="main-header">🌊 FloodSentinel</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Flood Risk Assessment with In-App Model Training</p>', unsafe_allow_html=True)

# ==================== SIDEBAR ====================
st.sidebar.markdown("### 🧭 Navigation")
page = st.sidebar.selectbox(
    "Choose a section:",
    ["🏠 Home", "📊 Dataset Preparation", "🤖 Train CNN Models", "🔮 Make Predictions", "📈 Results Dashboard"]
)

# ==================== PAGE: HOME ====================
if page == "🏠 Home":
    st.markdown("### 🎯 Project Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>🌊 Problem Statement</h4>
            <p>Floods cause widespread damage globally. This app uses deep learning to detect floods from satellite imagery.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-box">
            <h4>🎯 Our Solution</h4>
            <ul>
                <li>✅ Train 4 CNN models in the app</li>
                <li>✅ No need for external training</li>
                <li>✅ Use your own image dataset</li>
                <li>✅ Real-time training monitoring</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
            <h4>🚀 Quick Start Guide</h4>
            <ol>
                <li>📊 Upload your image dataset</li>
                <li>🤖 Train CNN models (ResNet, DenseNet, etc.)</li>
                <li>🔮 Make predictions on new images</li>
                <li>📈 View comprehensive results</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.info(f"**Current Device:** {device}")
        if torch.cuda.is_available():
            st.success(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        else:
            st.warning("⚠️ CPU Mode (Training will be slower)")

# ==================== PAGE: DATASET PREPARATION ====================
elif page == "📊 Dataset Preparation":
    st.markdown("### 📊 Prepare Your Image Dataset")
    
    st.markdown("""
    Upload your flood detection image dataset. The app expects:
    - **Images:** JPG, PNG, or TIFF format
    - **Labels:** JSON file mapping folder/filename to flood status (0 or 1)
    - **Structure:** Images organized in folders or with clear naming
    """)
    
    # Option 1: Upload images directory
    st.markdown("#### Option 1: Upload Image Folder")
    uploaded_folder = st.file_uploader(
        "Upload a ZIP file containing images",
        type=['zip'],
        help="ZIP file should contain images organized by class or with clear naming"
    )
    
    # Option 2: Scan local directory
    st.markdown("#### Option 2: Use Local Directory")
    local_dir = st.text_input(
        "Enter path to image directory:",
        placeholder="/path/to/your/images",
        help="Directory containing flood detection images"
    )
    
    # Label file upload
    st.markdown("#### Upload Labels (Optional)")
    label_file = st.file_uploader(
        "Upload label JSON file",
        type=['json'],
        help="JSON mapping: {'folder_name': 1, 'another_folder': 0}"
    )
    
    if st.button("🔍 Scan and Prepare Dataset", type="primary"):
        image_paths = []
        
        # Scan local directory
        if local_dir and os.path.exists(local_dir):
            with st.spinner("Scanning directory..."):
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
                    image_paths.extend(glob.glob(os.path.join(local_dir, "**", ext), recursive=True))
                
                st.success(f"✅ Found {len(image_paths)} images!")
        
        # Extract ZIP file
        elif uploaded_folder:
            import zipfile
            import tempfile
            
            with st.spinner("Extracting ZIP file..."):
                with tempfile.TemporaryDirectory() as temp_dir:
                    with zipfile.ZipFile(uploaded_folder, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    
                    for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
                        image_paths.extend(glob.glob(os.path.join(temp_dir, "**", ext), recursive=True))
                
                st.success(f"✅ Extracted {len(image_paths)} images!")
        
        if image_paths:
            # Load labels
            labels = {}
            if label_file:
                labels = json.load(label_file)
                st.success(f"✅ Loaded labels for {len(labels)} classes")
            else:
                st.info("ℹ️ No labels provided. Inferring from folder structure...")
                # Infer labels from folder names containing "flood" or "non_flood"
                for path in image_paths:
                    folder = os.path.basename(os.path.dirname(path))
                    if 'flood' in folder.lower() and 'non' not in folder.lower():
                        labels[folder] = 1
                    else:
                        labels[folder] = 0
                
                st.info(f"ℹ️ Inferred labels: {dict(list(labels.items())[:5])}...")
            
            # Create label array
            image_labels = []
            for path in image_paths:
                folder = os.path.basename(os.path.dirname(path))
                image_labels.append(labels.get(folder, 0))
            
            # Store in session state
            st.session_state.image_paths = image_paths
            st.session_state.image_labels = image_labels
            st.session_state.dataset_loaded = True
            
            # Display statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Images", len(image_paths))
            with col2:
                st.metric("Flood Images", sum(image_labels))
            with col3:
                st.metric("Non-Flood Images", len(image_labels) - sum(image_labels))
            
            # Show sample images
            st.markdown("#### 📸 Sample Images")
            cols = st.columns(5)
            for i, path in enumerate(image_paths[:5]):
                with cols[i]:
                    try:
                        img = Image.open(path)
                        st.image(img, caption=f"Label: {image_labels[i]}", use_container_width=True)
                    except:
                        st.error(f"Can't load image {i+1}")

# ==================== PAGE: TRAIN CNN MODELS ====================
elif page == "🤖 Train CNN Models":
    st.markdown("### 🤖 Train Deep Learning Models")
    
    if not st.session_state.dataset_loaded:
        st.warning("⚠️ Please prepare your dataset first!")
        st.stop()
    
    # Display dataset info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Total Images", len(st.session_state.image_paths))
    with col2:
        st.metric("🌊 Flood Images", sum(st.session_state.image_labels))
    with col3:
        st.metric("🏞️ Non-Flood", len(st.session_state.image_labels) - sum(st.session_state.image_labels))
    
    # Training configuration
    st.markdown("#### ⚙️ Training Configuration")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        batch_size = st.selectbox("Batch Size", [8, 16, 32, 64], index=2)
    with col2:
        num_epochs = st.slider("Number of Epochs", 1, 50, 10)
    with col3:
        learning_rate = st.select_slider("Learning Rate", 
                                         options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
                                         value=1e-4,
                                         format_func=lambda x: f"{x:.0e}")
    
    # Model selection
    st.markdown("#### 🎯 Select Models to Train")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        train_resnet = st.checkbox("✅ ResNet-50", value=True)
    with col2:
        train_densenet = st.checkbox("✅ DenseNet-121", value=True)
    with col3:
        train_efficientnet = st.checkbox("✅ EfficientNet-B0", value=True)
    with col4:
        train_vit = st.checkbox("✅ ViT-Base", value=True)
    
    # Start training button
    if st.button("🚀 Start Training", type="primary"):
        # Prepare dataset
        with st.spinner("Preparing dataset..."):
            transform = create_preprocessing_pipeline(224)
            
            # Create dataset
            dataset = FloodImageDataset(
                st.session_state.image_paths,
                st.session_state.image_labels,
                transform=transform
            )
            
            # Split dataset
            train_size = int(0.7 * len(dataset))
            val_size = int(0.15 * len(dataset))
            test_size = len(dataset) - train_size - val_size
            
            train_dataset, val_dataset, test_dataset = random_split(
                dataset, [train_size, val_size, test_size]
            )
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
            
            st.success(f"✅ Dataset prepared: Train={train_size}, Val={val_size}, Test={test_size}")
        
        # Store test loader for later evaluation
        st.session_state.test_loader = test_loader
        
        # Train selected models
        models_to_train = []
        if train_resnet:
            models_to_train.append(("ResNet-50", create_resnet50()))
        if train_densenet:
            models_to_train.append(("DenseNet-121", create_densenet121()))
        if train_efficientnet:
            models_to_train.append(("EfficientNet-B0", create_efficientnet_b0()))
        if train_vit:
            models_to_train.append(("ViT-Base", create_vit()))
        
        # Train each model
        for model_name, model in models_to_train:
            st.markdown(f"### 🔥 Training {model_name}")
            
            trained_model, history = train_model(
                model, train_loader, val_loader, 
                model_name, num_epochs, learning_rate
            )
            
            # Store trained model and history
            st.session_state.cnn_models[model_name] = trained_model
            st.session_state.training_history[model_name] = history
            
            # Plot training curves
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            epochs = range(1, len(history['train_loss']) + 1)
            
            ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
            ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title(f'{model_name} - Loss')
            ax1.legend()
            ax1.grid(True)
            
            ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
            ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title(f'{model_name} - Accuracy')
            ax2.legend()
            ax2.grid(True)
            
            st.pyplot(fig)
            plt.close()
        
        st.success("🎉 All models trained successfully!")
        st.balloons()

# ==================== PAGE: MAKE PREDICTIONS ====================
elif page == "🔮 Make Predictions":
    st.markdown("### 🔮 Flood Detection Predictions")
    
    if not st.session_state.cnn_models:
        st.warning("⚠️ Please train models first!")
        st.stop()
    
    st.markdown("#### 📤 Upload Image for Prediction")
    uploaded_image = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_image:
        # Display image
        image = Image.open(uploaded_image)
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            if st.button("🔮 Predict", type="primary"):
                # Preprocess image
                transform = create_preprocessing_pipeline(224)
                image_tensor = transform(image).unsqueeze(0).to(device)
                
                # Get predictions from all models
                predictions = {}
                
                with torch.no_grad():
                    for model_name, model in st.session_state.cnn_models.items():
                        model.eval()
                        output = model(image_tensor)
                        
                        if hasattr(output, 'logits'):
                            output = output.logits
                        
                        prob = torch.softmax(output, dim=1)[0, 1].item()
                        predictions[model_name] = prob
                
                # Calculate ensemble prediction
                ensemble_pred = np.mean(list(predictions.values()))
                
                # Display results
                st.markdown("#### 🎯 Prediction Results")
                
                # Main result
                if ensemble_pred < 0.3:
                    risk = "🟢 Low Flood Risk"
                    color = "#4facfe"
                elif ensemble_pred < 0.7:
                    risk = "🟡 Moderate Flood Risk"
                    color = "#fee140"
                else:
                    risk = "🔴 High Flood Risk"
                    color = "#fa709a"
                
                st.markdown(f"""
                <div class="metric-container" style="background: {color};">
                    <h3>Ensemble Prediction</h3>
                    <h1>{ensemble_pred:.1%}</h1>
                    <h4>{risk}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Individual model predictions
                st.markdown("##### 📊 Individual Model Predictions")
                pred_df = pd.DataFrame({
                    'Model': list(predictions.keys()),
                    'Probability': [f"{p:.2%}" for p in predictions.values()]
                })
                st.dataframe(pred_df, use_container_width=True)
                
                # Visualization
                fig = px.bar(
                    x=list(predictions.keys()),
                    y=list(predictions.values()),
                    title="Model Predictions Comparison",
                    labels={'x': 'Model', 'y': 'Flood Probability'},
                    color=list(predictions.values()),
                    color_continuous_scale='RdYlGn_r'
                )
                fig.add_hline(y=ensemble_pred, line_dash="dash", 
                             line_color="red", annotation_text="Ensemble")
                st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: RESULTS DASHBOARD ====================
elif page == "📈 Results Dashboard":
    st.markdown("### 📈 Training Results & Model Performance")
    
    if not st.session_state.training_history:
        st.warning("⚠️ No training history available. Please train models first!")
        st.stop()
    
    # Summary metrics
    st.markdown("#### 📊 Training Summary")
    
    summary_data = []
    for model_name, history in st.session_state.training_history.items():
        summary_data.append({
            'Model': model_name,
            'Final Train Acc': f"{history['train_acc'][-1]:.4f}",
            'Final Val Acc': f"{history['val_acc'][-1]:.4f}",
            'Best Val Acc': f"{max(history['val_acc']):.4f}",
            'Final Train Loss': f"{history['train_loss'][-1]:.4f}",
            'Final Val Loss': f"{history['val_loss'][-1]:.4f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    # Comparative plots
    st.markdown("#### 📈 Comparative Training Curves")
    
    # Accuracy comparison
    fig_acc = go.Figure()
    for model_name, history in st.session_state.training_history.items():
        epochs = list(range(1, len(history['val_acc']) + 1))
        fig_acc.add_trace(go.Scatter(
            x=epochs, y=history['val_acc'],
            mode='lines+markers',
            name=model_name
        ))
    
    fig_acc.update_layout(
        title='Validation Accuracy Comparison',
        xaxis_title='Epoch',
        yaxis_title='Accuracy',
        hovermode='x unified'
    )
    st.plotly_chart(fig_acc, use_container_width=True)
    
    # Loss comparison
    fig_loss = go.Figure()
    for model_name, history in st.session_state.training_history.items():
        epochs = list(range(1, len(history['val_loss']) + 1))
        fig_loss.add_trace(go.Scatter(
            x=epochs, y=history['val_loss'],
            mode='lines+markers',
            name=model_name
        ))
    
    fig_loss.update_layout(
        title='Validation Loss Comparison',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        hovermode='x unified'
    )
    st.plotly_chart(fig_loss, use_container_width=True)
    
    # Model download section
    st.markdown("#### 💾 Download Trained Models")
    
    for model_name in st.session_state.cnn_models.keys():
        if os.path.exists(f"{model_name}_best.pth"):
            with open(f"{model_name}_best.pth", 'rb') as f:
                st.download_button(
                    label=f"📥 Download {model_name}",
                    data=f,
                    file_name=f"{model_name}_best.pth",
                    mime="application/octet-stream"
                )

# ==================== SIDEBAR STATUS ====================
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Status")

if st.session_state.dataset_loaded:
    st.sidebar.success("✅ Dataset Loaded")
    st.sidebar.info(f"📸 {len(st.session_state.get('image_paths', []))} images")
else:
    st.sidebar.error("❌ Dataset Not Loaded")

if st.session_state.cnn_models:
    st.sidebar.success(f"✅ {len(st.session_state.cnn_models)} Models Trained")
else:
    st.sidebar.error("❌ No Models Trained")

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ System Info")
st.sidebar.info(f"""
**Device:** {device}  
**PyTorch:** {torch.__version__}  
**CUDA Available:** {torch.cuda.is_available()}
""")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #888; padding: 10px;">
        <p>🌊 FloodSentinel - Crafted with ❤️ by Shreyas, Chinmay and Kaivalya</p>
    </div>
""", unsafe_allow_html=True)
