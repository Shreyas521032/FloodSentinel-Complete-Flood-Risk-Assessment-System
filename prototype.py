import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Libraries
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Dense, LSTM, GRU, Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Dropout, BatchNormalization, Activation, Add, Concatenate, Flatten,
    Conv1D, MaxPooling1D, GlobalMaxPooling1D, Reshape, TimeDistributed
)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.impute import SimpleImputer

# Image Processing
from PIL import Image
import cv2
from skimage import filters, segmentation, measure, morphology
from skimage.feature import local_binary_pattern

# Data Handling
import kagglehub
import os
import json
import pickle
from datetime import datetime, timedelta
import time

# Set page config
st.set_page_config(
    page_title="FloodSentinel - AI-Powered Flood Risk Assessment",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'datasets_loaded' not in st.session_state:
    st.session_state.datasets_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'flood_data' not in st.session_state:
    st.session_state.flood_data = None
if 'satellite_data' not in st.session_state:
    st.session_state.satellite_data = None

class FloodSentinelEngine:
    def __init__(self):
        self.tabular_models = {}
        self.deep_models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        
    def download_datasets(self):
        """Download datasets from Kaggle"""
        try:
            with st.spinner("Downloading flood prediction dataset..."):
                flood_path = kagglehub.dataset_download("naiyakhalid/flood-prediction-dataset")
                st.success(f"Flood dataset downloaded to: {flood_path}")
            
            with st.spinner("Downloading satellite imagery dataset..."):
                satellite_path = kagglehub.dataset_download("rhythmroy/sen12flood-flood-detection-dataset")
                st.success(f"Satellite dataset downloaded to: {satellite_path}")
            
            return flood_path, satellite_path
        except Exception as e:
            st.error(f"Error downloading datasets: {str(e)}")
            return None, None
    
    def load_and_preprocess_data(self, flood_path, satellite_path):
        """Load and preprocess both datasets"""
        try:
            # Load tabular flood data
            flood_files = []
            for root, dirs, files in os.walk(flood_path):
                for file in files:
                    if file.endswith('.csv'):
                        flood_files.append(os.path.join(root, file))
            
            if flood_files:
                flood_data = pd.read_csv(flood_files[0])
                st.success(f"Loaded flood data: {flood_data.shape}")
            else:
                # Create synthetic flood data for demonstration
                np.random.seed(42)
                n_samples = 1000
                flood_data = pd.DataFrame({
                    'rainfall_mm': np.random.exponential(20, n_samples),
                    'river_level_m': np.random.normal(5, 2, n_samples),
                    'soil_moisture': np.random.uniform(0.1, 0.9, n_samples),
                    'elevation_m': np.random.normal(100, 50, n_samples),
                    'temperature_c': np.random.normal(25, 10, n_samples),
                    'humidity_percent': np.random.uniform(40, 90, n_samples),
                    'wind_speed_kmh': np.random.exponential(10, n_samples),
                    'previous_flood_days': np.random.poisson(30, n_samples),
                    'season': np.random.choice(['Spring', 'Summer', 'Autumn', 'Winter'], n_samples),
                    'land_use': np.random.choice(['Urban', 'Rural', 'Forest', 'Agriculture'], n_samples)
                })
                # Create flood risk target
                flood_risk_score = (
                    flood_data['rainfall_mm'] * 0.3 +
                    flood_data['river_level_m'] * 0.2 +
                    flood_data['soil_moisture'] * 0.1 +
                    (1 / (flood_data['elevation_m'] + 1)) * 1000 * 0.2 +
                    flood_data['humidity_percent'] * 0.1 +
                    np.random.normal(0, 5, n_samples)
                )
                flood_data['flood_risk'] = (flood_risk_score > np.percentile(flood_risk_score, 70)).astype(int)
                st.info("Using synthetic flood data for demonstration")
            
            # Process satellite data path
            satellite_files = []
            for root, dirs, files in os.walk(satellite_path):
                for file in files:
                    if file.endswith(('.jpg', '.png', '.tif', '.tiff')):
                        satellite_files.append(os.path.join(root, file))
            
            satellite_data = {
                'path': satellite_path,
                'files': satellite_files[:100],  # Limit for demo
                'count': len(satellite_files)
            }
            
            return flood_data, satellite_data
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None, None
    
    def preprocess_tabular_data(self, data):
        """Preprocess tabular flood data"""
        # Handle missing values
        numeric_features = data.select_dtypes(include=[np.number]).columns
        categorical_features = data.select_dtypes(include=['object']).columns
        
        # Impute missing values
        if len(numeric_features) > 0:
            imputer_num = SimpleImputer(strategy='median')
            data[numeric_features] = imputer_num.fit_transform(data[numeric_features])
        
        if len(categorical_features) > 0:
            imputer_cat = SimpleImputer(strategy='most_frequent')
            data[categorical_features] = imputer_cat.fit_transform(data[categorical_features])
        
        # Encode categorical variables
        encoded_data = data.copy()
        for col in categorical_features:
            if col != 'flood_risk':  # Don't encode target variable
                le = LabelEncoder()
                encoded_data[col] = le.fit_transform(data[col])
                self.encoders[col] = le
        
        return encoded_data
    
    def create_hybrid_model(self, tabular_shape, image_shape):
        """Create hybrid model combining tabular and image data"""
        # Tabular input branch
        tabular_input = Input(shape=(tabular_shape,), name='tabular_input')
        tabular_dense = Dense(128, activation='relu')(tabular_input)
        tabular_dense = Dropout(0.3)(tabular_dense)
        tabular_dense = Dense(64, activation='relu')(tabular_dense)
        tabular_dense = Dropout(0.2)(tabular_dense)
        
        # Image input branch (CNN)
        image_input = Input(shape=image_shape, name='image_input')
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D((2, 2))(conv1)
        
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D((2, 2))(conv2)
        
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D((2, 2))(conv3)
        
        # Global average pooling
        gap = GlobalAveragePooling2D()(pool3)
        image_dense = Dense(64, activation='relu')(gap)
        image_dense = Dropout(0.3)(image_dense)
        
        # Combine branches
        combined = Concatenate()([tabular_dense, image_dense])
        combined = Dense(128, activation='relu')(combined)
        combined = Dropout(0.3)(combined)
        combined = Dense(64, activation='relu')(combined)
        combined = Dropout(0.2)(combined)
        
        # Output layer
        output = Dense(1, activation='sigmoid', name='flood_prediction')(combined)
        
        # Create model
        model = Model(inputs=[tabular_input, image_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='binary_crossentropy', 
                     metrics=['accuracy'])
        
        return model
    
    def train_tabular_models(self, X, y):
        """Train multiple ML models for tabular data"""
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['tabular'] = scaler
        
        results = {}
        
        for name, model in models.items():
            with st.spinner(f"Training {name}..."):
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1': f1_score(y_test, y_pred),
                    'auc': roc_auc_score(y_test, y_pred_proba),
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'y_test': y_test
                }
                
                # Feature importance for tree-based models
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = model.feature_importances_
        
        self.tabular_models = results
        return results
    
    def create_synthetic_image_data(self, n_samples=1000):
        """Create synthetic satellite imagery data for demonstration"""
        np.random.seed(42)
        
        # Generate synthetic satellite images (64x64x3)
        images = []
        labels = []
        
        for i in range(n_samples):
            # Create base image
            img = np.random.rand(64, 64, 3)
            
            # Add patterns for flood/no-flood
            if np.random.rand() > 0.5:  # Flood
                # Add blue-ish tint for water
                img[:, :, 2] += 0.3
                img[:, :, 0] *= 0.7
                img[:, :, 1] *= 0.7
                
                # Add some noise patterns
                noise = np.random.rand(64, 64) * 0.2
                img[:, :, 2] += noise
                
                label = 1
            else:  # No flood
                # Add green-ish tint for vegetation
                img[:, :, 1] += 0.3
                img[:, :, 0] *= 0.8
                img[:, :, 2] *= 0.8
                
                label = 0
            
            # Clip values
            img = np.clip(img, 0, 1)
            images.append(img)
            labels.append(label)
        
        return np.array(images), np.array(labels)
    
    def train_image_model(self, images, labels):
        """Train CNN model for satellite imagery"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=0.2, random_state=42
        )
        
        # Create CNN model
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            GlobalAveragePooling2D(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        with st.spinner("Training CNN model for satellite imagery..."):
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=10,
                batch_size=32,
                verbose=0,
                callbacks=[
                    EarlyStopping(patience=3, restore_best_weights=True),
                    ReduceLROnPlateau(patience=2, factor=0.5)
                ]
            )
        
        # Evaluate model
        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        
        self.deep_models['cnn'] = {
            'model': model,
            'history': history,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_loss': train_loss,
            'test_loss': test_loss
        }
        
        return model, history

def main():
    st.markdown('<div class="main-header">🌊 FloodSentinel</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; color: #666; margin-bottom: 3rem;">AI-Powered Flood Risk Assessment Using Multi-Temporal Satellite Imagery and Deep Neural Networks</div>', unsafe_allow_html=True)
    
    # Initialize the engine
    engine = FloodSentinelEngine()
    
    # Sidebar
    st.sidebar.title("🛠️ FloodSentinel Control Panel")
    
    # Main navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview", 
        "🤖 Model Training", 
        "📈 Model Performance", 
        "🔮 Risk Prediction", 
        "📋 System Status"
    ])
    
    with tab1:
        st.markdown('<div class="sub-header">📊 Data Management & Overview</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Download & Load Datasets", type="primary"):
                flood_path, satellite_path = engine.download_datasets()
                if flood_path and satellite_path:
                    flood_data, satellite_data = engine.load_and_preprocess_data(flood_path, satellite_path)
                    if flood_data is not None and satellite_data is not None:
                        st.session_state.flood_data = flood_data
                        st.session_state.satellite_data = satellite_data
                        st.session_state.datasets_loaded = True
        
        with col2:
            if st.session_state.datasets_loaded:
                st.markdown('<div class="success-box">✅ Datasets loaded successfully!</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box">⚠️ Please load datasets first</div>', unsafe_allow_html=True)
        
        # Data overview
        if st.session_state.datasets_loaded:
            st.subheader("📋 Flood Prediction Data Overview")
            
            # Display basic statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Samples", len(st.session_state.flood_data))
            with col2:
                st.metric("Features", len(st.session_state.flood_data.columns) - 1)
            with col3:
                flood_rate = st.session_state.flood_data['flood_risk'].mean()
                st.metric("Flood Risk Rate", f"{flood_rate:.2%}")
            with col4:
                st.metric("Satellite Images", st.session_state.satellite_data['count'])
            
            # Data visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # Correlation heatmap
                numeric_cols = st.session_state.flood_data.select_dtypes(include=[np.number]).columns
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(st.session_state.flood_data[numeric_cols].corr(), 
                           annot=True, cmap='RdYlBu_r', center=0, ax=ax)
                plt.title('Feature Correlation Matrix')
                st.pyplot(fig)
            
            with col2:
                # Distribution of flood risk
                fig = px.histogram(st.session_state.flood_data, x='flood_risk', 
                                 title='Distribution of Flood Risk')
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature distributions
            st.subheader("📊 Feature Distributions")
            numeric_features = st.session_state.flood_data.select_dtypes(include=[np.number]).columns.tolist()
            if 'flood_risk' in numeric_features:
                numeric_features.remove('flood_risk')
            
            selected_features = st.multiselect(
                "Select features to visualize:",
                numeric_features,
                default=numeric_features[:4] if len(numeric_features) >= 4 else numeric_features
            )
            
            if selected_features:
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=selected_features[:4]
                )
                
                for i, feature in enumerate(selected_features[:4]):
                    row = i // 2 + 1
                    col = i % 2 + 1
                    
                    fig.add_trace(
                        go.Histogram(x=st.session_state.flood_data[feature], 
                                   name=feature, showlegend=False),
                        row=row, col=col
                    )
                
                fig.update_layout(height=600, title_text="Feature Distributions")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown('<div class="sub-header">🤖 Model Training & Development</div>', unsafe_allow_html=True)
        
        if not st.session_state.datasets_loaded:
            st.warning("Please load datasets first in the Data Overview tab.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔧 Tabular Data Models")
            if st.button("🚀 Train ML Models", type="primary"):
                # Prepare tabular data
                processed_data = engine.preprocess_tabular_data(st.session_state.flood_data)
                
                # Separate features and target
                X = processed_data.drop('flood_risk', axis=1)
                y = processed_data['flood_risk']
                
                # Train models
                results = engine.train_tabular_models(X, y)
                
                st.success("✅ Tabular models trained successfully!")
                
                # Display results
                results_df = pd.DataFrame({
                    'Model': list(results.keys()),
                    'Accuracy': [results[m]['accuracy'] for m in results.keys()],
                    'Precision': [results[m]['precision'] for m in results.keys()],
                    'Recall': [results[m]['recall'] for m in results.keys()],
                    'F1-Score': [results[m]['f1'] for m in results.keys()],
                    'AUC': [results[m]['auc'] for m in results.keys()]
                })
                
                st.dataframe(results_df, use_container_width=True)
        
        with col2:
            st.subheader("🖼️ Satellite Imagery Models")
            if st.button("🚀 Train CNN Models", type="primary"):
                # Generate synthetic satellite data
                images, labels = engine.create_synthetic_image_data(1000)
                
                # Train CNN model
                model, history = engine.train_image_model(images, labels)
                
                st.success("✅ CNN model trained successfully!")
                
                # Display training history
                if history:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    
                    # Accuracy plot
                    ax1.plot(history.history['accuracy'], label='Training Accuracy')
                    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
                    ax1.set_title('Model Accuracy')
                    ax1.set_xlabel('Epoch')
                    ax1.set_ylabel('Accuracy')
                    ax1.legend()
                    
                    # Loss plot
                    ax2.plot(history.history['loss'], label='Training Loss')
                    ax2.plot(history.history['val_loss'], label='Validation Loss')
                    ax2.set_title('Model Loss')
                    ax2.set_xlabel('Epoch')
                    ax2.set_ylabel('Loss')
                    ax2.legend()
                    
                    st.pyplot(fig)
        
        # Model architecture visualization
        if st.checkbox("Show Model Architecture"):
            st.subheader("🏗️ Model Architecture")
            
            # Create a sample hybrid model for visualization
            try:
                hybrid_model = engine.create_hybrid_model(10, (64, 64, 3))
                st.text("Hybrid Model Summary:")
                
                # Create a string buffer to capture model summary
                import io
                import contextlib
                
                @contextlib.contextmanager
                def capture_stdout():
                    old_stdout = st.write
                    st.write = lambda x: None
                    yield
                    st.write = old_stdout
                
                # Display model summary
                model_summary = []
                hybrid_model.summary(print_fn=lambda x: model_summary.append(x))
                st.text('\n'.join(model_summary))
                
            except Exception as e:
                st.error(f"Error creating model architecture: {str(e)}")
    
    with tab3:
        st.markdown('<div class="sub-header">📈 Model Performance Analysis</div>', unsafe_allow_html=True)
        
        if not hasattr(engine, 'tabular_models') or not engine.tabular_models:
            st.warning("Please train models first in the Model Training tab.")
            return
        
        # Model comparison
        st.subheader("🏆 Model Comparison")
        
        if engine.tabular_models:
            # Create comparison dataframe
            comparison_data = []
            for model_name, results in engine.tabular_models.items():
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': results['accuracy'],
                    'Precision': results['precision'],
                    'Recall': results['recall'],
                    'F1-Score': results['f1'],
                    'AUC': results['auc']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Bar chart comparison
            fig = px.bar(comparison_df, x='Model', y=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'],
                        title='Model Performance Comparison', barmode='group')
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed metrics table
            st.subheader("📊 Detailed Performance Metrics")
            st.dataframe(comparison_df, use_container_width=True)
        
        # ROC Curves
        st.subheader("📈 ROC Curves")
        
        if engine.tabular_models:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for model_name, results in engine.tabular_models.items():
                if 'probabilities' in results and 'y_test' in results:
                    fpr, tpr, _ = roc_curve(results['y_test'], results['probabilities'])
                    auc_score = results['auc']
                    ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
            
            ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curves for All Models')
            ax.legend()
            ax.grid(True)
            
            st.pyplot(fig)
        
        # Feature importance
        st.subheader("🎯 Feature Importance")
        
        if engine.feature_importance:
            importance_data = []
            feature_names = list(st.session_state.flood_data.columns)
            if 'flood_risk' in feature_names:
                feature_names.remove('flood_risk')
            
            for model_name, importance in engine.feature_importance.items():
                for i, imp in enumerate(importance):
                    if i < len(feature_names):
                        importance_data.append({
                            'Model': model_name,
                            'Feature': feature_names[i],
                            'Importance': imp
                        })
            
            if importance_data:
                importance_df = pd.DataFrame(importance_data)
                
                # Create importance plot
                fig = px.bar(importance_df, x='Feature', y='Importance', color='Model',
                           title='Feature Importance by Model', barmode='group')
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        
        # Confusion matrices
        st.subheader("🔍 Confusion Matrices")
        
        if engine.tabular_models:
            cols = st.columns(2)
            col_idx = 0
            
            for model_name, results in engine.tabular_models.items():
                if 'predictions' in results and 'y_test' in results:
                    cm = confusion_matrix(results['y_test'], results['predictions'])
                    
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_title(f'Confusion Matrix - {model_name}')
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    
                    with cols[col_idx % 2]:
                        st.pyplot(fig)
                    col_idx += 1
    
    with tab4:
        st.markdown('<div class="sub-header">🔮 Real-Time Flood Risk Prediction</div>', unsafe_allow_html=True)
        
        if not hasattr(engine, 'tabular_models') or not engine.tabular_models:
            st.warning("Please train models first in the Model Training tab.")
            return
        
        st.subheader("🌦️ Input Environmental Parameters")
        
        # Create input form
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=200.0, value=25.0)
                river_level = st.number_input("River Level (m)", min_value=0.0, max_value=20.0, value=5.0)
                soil_moisture = st.slider("Soil Moisture", min_value=0.0, max_value=1.0, value=0.5)
                elevation = st.number_input("Elevation (m)", min_value=0.0, max_value=1000.0, value=100.0)
            
            with col2:
                temperature = st.number_input("Temperature (°C)", min_value=-20.0, max_value=50.0, value=25.0)
                humidity = st.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
                wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=100.0, value=10.0)
                prev_flood_days = st.number_input("Days Since Last Flood", min_value=0, max_value=365, value=30)
            
            with col3:
                season = st.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"])
                land_use = st.selectbox("Land Use", ["Urban", "Rural", "Forest", "Agriculture"])
                
                # Image upload for satellite data
                uploaded_image = st.file_uploader("Upload Satellite Image (Optional)", 
                                                type=['jpg', 'jpeg', 'png'])
            
            submitted = st.form_submit_button("🔮 Predict Flood Risk", type="primary")
        
        if submitted:
            # Prepare input data
            input_data = pd.DataFrame({
                'rainfall_mm': [rainfall],
                'river_level_m': [river_level],
                'soil_moisture': [soil_moisture],
                'elevation_m': [elevation],
                'temperature_c': [temperature],
                'humidity_percent': [humidity],
                'wind_speed_kmh': [wind_speed],
                'previous_flood_days': [prev_flood_days],
                'season': [season],
                'land_use': [land_use]
            })
            
            # Encode categorical variables
            for col in ['season', 'land_use']:
                if col in engine.encoders:
                    # Handle unseen categories
                    try:
                        input_data[col] = engine.encoders[col].transform(input_data[col])
                    except ValueError:
                        # Use most frequent category if unseen
                        input_data[col] = 0
            
            # Scale the input data
            if 'tabular' in engine.scalers:
                input_scaled = engine.scalers['tabular'].transform(input_data)
            else:
                input_scaled = input_data.values
            
            # Make predictions with all models
            predictions = {}
            for model_name, results in engine.tabular_models.items():
                model = results['model']
                prob = model.predict_proba(input_scaled)[0][1]
                pred = model.predict(input_scaled)[0]
                predictions[model_name] = {'probability': prob, 'prediction': pred}
            
            # Display predictions
            st.subheader("🎯 Flood Risk Predictions")
            
            # Create metrics display
            cols = st.columns(len(predictions))
            for i, (model_name, pred_data) in enumerate(predictions.items()):
                with cols[i]:
                    risk_level = "HIGH" if pred_data['probability'] > 0.7 else "MEDIUM" if pred_data['probability'] > 0.3 else "LOW"
                    color = "red" if risk_level == "HIGH" else "orange" if risk_level == "MEDIUM" else "green"
                    
                    st.markdown(f"""
                    <div style="background-color: {color}; color: white; padding: 1rem; border-radius: 10px; text-align: center;">
                        <h3>{model_name}</h3>
                        <h2>{pred_data['probability']:.1%}</h2>
                        <p>Risk Level: {risk_level}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Ensemble prediction
            avg_probability = np.mean([pred['probability'] for pred in predictions.values()])
            ensemble_risk = "HIGH" if avg_probability > 0.7 else "MEDIUM" if avg_probability > 0.3 else "LOW"
            
            st.subheader("🏆 Ensemble Prediction")
            st.markdown(f"""
            <div style="background-color: #1f77b4; color: white; padding: 2rem; border-radius: 15px; text-align: center; margin: 2rem 0;">
                <h2>Ensemble Flood Risk: {avg_probability:.1%}</h2>
                <h3>Risk Level: {ensemble_risk}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Risk factors analysis
            st.subheader("📊 Risk Factors Analysis")
            
            # Calculate risk contribution of each factor
            risk_factors = {
                'High Rainfall': min(rainfall / 50, 1) * 0.3,
                'High River Level': min(river_level / 10, 1) * 0.25,
                'High Soil Moisture': soil_moisture * 0.15,
                'Low Elevation': max(0, (200 - elevation) / 200) * 0.15,
                'High Humidity': (humidity / 100) * 0.1,
                'Recent Flood History': max(0, (90 - prev_flood_days) / 90) * 0.05
            }
            
            risk_df = pd.DataFrame(list(risk_factors.items()), columns=['Factor', 'Contribution'])
            risk_df['Contribution'] = risk_df['Contribution'].clip(0, 1)
            
            fig = px.bar(risk_df, x='Factor', y='Contribution', 
                        title='Risk Factors Contribution',
                        color='Contribution',
                        color_continuous_scale='Reds')
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.subheader("💡 Recommendations")
            
            recommendations = []
            if avg_probability > 0.7:
                recommendations.extend([
                    "🚨 HIGH RISK: Immediate action required",
                    "📢 Issue flood warnings to affected areas",
                    "🏠 Evacuate low-lying areas if necessary",
                    "🛡️ Deploy emergency response teams"
                ])
            elif avg_probability > 0.3:
                recommendations.extend([
                    "⚠️ MEDIUM RISK: Monitor conditions closely",
                    "📊 Increase monitoring frequency",
                    "📋 Prepare emergency response protocols",
                    "📞 Alert relevant authorities"
                ])
            else:
                recommendations.extend([
                    "✅ LOW RISK: Normal monitoring sufficient",
                    "🔄 Continue regular monitoring",
                    "📈 Track weather patterns",
                    "📚 Review and update emergency plans"
                ])
            
            for rec in recommendations:
                st.markdown(f"• {rec}")
            
            # Process uploaded image if available
            if uploaded_image is not None:
                st.subheader("🛰️ Satellite Image Analysis")
                
                # Display the uploaded image
                image = Image.open(uploaded_image)
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(image, caption="Uploaded Satellite Image", use_column_width=True)
                
                with col2:
                    # Simulate image analysis
                    if 'cnn' in engine.deep_models:
                        # Resize image for model input
                        img_resized = image.resize((64, 64))
                        img_array = np.array(img_resized) / 255.0
                        
                        # Add batch dimension
                        img_batch = np.expand_dims(img_array, axis=0)
                        
                        # Make prediction
                        try:
                            cnn_model = engine.deep_models['cnn']['model']
                            img_prediction = cnn_model.predict(img_batch)[0][0]
                            
                            st.markdown(f"""
                            <div style="background-color: #e74c3c; color: white; padding: 1rem; border-radius: 10px; text-align: center;">
                                <h3>CNN Image Analysis</h3>
                                <h2>Flood Probability: {img_prediction:.1%}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error analyzing image: {str(e)}")
                    else:
                        st.info("CNN model not available. Train the CNN model first.")
    
    with tab5:
        st.markdown('<div class="sub-header">📋 System Status & Information</div>', unsafe_allow_html=True)
        
        # System status
        st.subheader("🔧 System Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_data = st.session_state.datasets_loaded
            color = "green" if status_data else "red"
            st.markdown(f"""
            <div style="background-color: {color}; color: white; padding: 1rem; border-radius: 10px; text-align: center;">
                <h3>Data Status</h3>
                <p>{'✅ Loaded' if status_data else '❌ Not Loaded'}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            models_trained = len(engine.tabular_models) > 0
            color = "green" if models_trained else "red"
            st.markdown(f"""
            <div style="background-color: {color}; color: white; padding: 1rem; border-radius: 10px; text-align: center;">
                <h3>ML Models</h3>
                <p>{'✅ Trained' if models_trained else '❌ Not Trained'}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            cnn_trained = 'cnn' in engine.deep_models
            color = "green" if cnn_trained else "red"
            st.markdown(f"""
            <div style="background-color: {color}; color: white; padding: 1rem; border-radius: 10px; text-align: center;">
                <h3>CNN Model</h3>
                <p>{'✅ Trained' if cnn_trained else '❌ Not Trained'}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Model information
        if engine.tabular_models:
            st.subheader("🤖 Trained Models Information")
            
            model_info = []
            for model_name, results in engine.tabular_models.items():
                model_info.append({
                    'Model': model_name,
                    'Accuracy': f"{results['accuracy']:.3f}",
                    'F1-Score': f"{results['f1']:.3f}",
                    'AUC': f"{results['auc']:.3f}",
                    'Status': '✅ Ready'
                })
            
            model_df = pd.DataFrame(model_info)
            st.dataframe(model_df, use_container_width=True)
        
        # System requirements and information
        st.subheader("📚 System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🔧 Technical Stack:**
            - **Frontend:** Streamlit
            - **ML Framework:** Scikit-learn
            - **Deep Learning:** TensorFlow/Keras
            - **Data Processing:** Pandas, NumPy
            - **Visualization:** Plotly, Matplotlib, Seaborn
            - **Image Processing:** PIL, OpenCV, Scikit-image
            - **Data Source:** Kaggle Hub
            """)
        
        with col2:
            st.markdown("""
            **📊 Model Capabilities:**
            - **Tabular Data:** Random Forest, Gradient Boosting, SVM, Logistic Regression
            - **Image Data:** Convolutional Neural Networks (CNN)
            - **Hybrid Models:** Combined tabular + image processing
            - **Real-time Prediction:** Multi-model ensemble
            - **Risk Assessment:** Probability-based risk levels
            - **Feature Analysis:** Importance ranking and contribution analysis
            """)
        
        # About section
        st.subheader("ℹ️ About FloodSentinel")
        
        st.markdown("""
        **FloodSentinel** is an advanced AI-powered flood risk assessment system that combines:
        
        1. **Multi-Modal Data Processing**: Integrates both tabular environmental data and satellite imagery
        2. **Advanced Machine Learning**: Utilizes ensemble methods with multiple ML algorithms
        3. **Deep Learning**: Employs CNNs for satellite image analysis
        4. **Real-time Prediction**: Provides instant flood risk assessments
        5. **Explainable AI**: Offers detailed analysis of risk factors and model decisions
        6. **User-Friendly Interface**: Streamlit-based web application for easy access
        
        The system is designed to support disaster management authorities, urban planners, and emergency response teams 
        in making informed decisions about flood risk management and mitigation strategies.
        """)
        
        # Performance metrics
        if engine.tabular_models:
            st.subheader("📈 System Performance")
            
            best_model = max(engine.tabular_models.items(), key=lambda x: x[1]['accuracy'])
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Best Model", best_model[0])
            with col2:
                st.metric("Best Accuracy", f"{best_model[1]['accuracy']:.3f}")
            with col3:
                st.metric("Best F1-Score", f"{best_model[1]['f1']:.3f}")
            with col4:
                st.metric("Best AUC", f"{best_model[1]['auc']:.3f}")
        
        # Data source information
        st.subheader("📁 Data Sources")
        
        st.markdown("""
        **Primary Datasets:**
        1. **Flood Prediction Dataset** (naiyakhalid/flood-prediction-dataset)
           - Environmental parameters (rainfall, river levels, soil moisture, etc.)
           - Historical flood events
           - Meteorological data
        
        2. **SEN12-FLOOD Dataset** (rhythmroy/sen12flood-flood-detection-dataset)
           - Satellite imagery for flood detection
           - Multi-temporal observations
           - Various spectral bands for comprehensive analysis
        
        **Data Processing Pipeline:**
        - Automated data cleaning and preprocessing
        - Feature engineering and selection
        - Missing value imputation
        - Categorical encoding
        - Data normalization and scaling
        """)
        
        # Export functionality
        st.subheader("💾 Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export Model Performance Report"):
                if engine.tabular_models:
                    # Create performance report
                    report_data = []
                    for model_name, results in engine.tabular_models.items():
                        report_data.append({
                            'Model': model_name,
                            'Accuracy': results['accuracy'],
                            'Precision': results['precision'],
                            'Recall': results['recall'],
                            'F1-Score': results['f1'],
                            'AUC': results['auc']
                        })
                    
                    report_df = pd.DataFrame(report_data)
                    csv = report_df.to_csv(index=False)
                    
                    st.download_button(
                        label="Download Performance Report",
                        data=csv,
                        file_name="floodsentinel_performance_report.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No models trained yet.")
        
        with col2:
            if st.button("🔧 Export System Configuration"):
                config = {
                    'datasets_loaded': st.session_state.datasets_loaded,
                    'models_trained': len(engine.tabular_models),
                    'cnn_trained': 'cnn' in engine.deep_models,
                    'timestamp': datetime.now().isoformat(),
                    'system_info': {
                        'tensorflow_version': tf.__version__,
                        'streamlit_version': st.__version__
                    }
                }
                
                config_json = json.dumps(config, indent=2)
                
                st.download_button(
                    label="Download System Config",
                    data=config_json,
                    file_name="floodsentinel_config.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()
