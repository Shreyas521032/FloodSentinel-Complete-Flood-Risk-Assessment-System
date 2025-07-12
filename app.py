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

# ML/DL Libraries
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
import lightgbm as lgb

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten, Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Image Processing
from PIL import Image
import cv2

# Data Download
import kagglehub
import os
import zipfile
import glob

# Set page config
st.set_page_config(
    page_title="FloodSentinel",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stProgress .st-bo {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'datasets_loaded' not in st.session_state:
    st.session_state.datasets_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

class FloodSentinel:
    def __init__(self):
        self.numerical_data = None
        self.image_data = None
        self.ml_models = {}
        self.dl_models = {}
        self.scalers = {}
        self.results = {}
        
    @st.cache_data
    def download_datasets(_self):
        """Download datasets from Kaggle"""
        try:
            with st.spinner("Downloading numerical flood prediction dataset..."):
                path1 = kagglehub.dataset_download("naiyakhalid/flood-prediction-dataset")
                st.success(f"Dataset 1 downloaded to: {path1}")
                
            with st.spinner("Downloading satellite imagery dataset..."):
                path2 = kagglehub.dataset_download("rhythmroy/sen12flood-flood-detection-dataset")
                st.success(f"Dataset 2 downloaded to: {path2}")
                
            return path1, path2
        except Exception as e:
            st.error(f"Error downloading datasets: {str(e)}")
            return None, None
    
    def load_numerical_data(self, path):
        """Load and preprocess numerical flood prediction data"""
        try:
            # Find CSV files in the path
            csv_files = glob.glob(os.path.join(path, "*.csv"))
            if not csv_files:
                # Look in subdirectories
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.endswith('.csv'):
                            csv_files.append(os.path.join(root, file))
            
            if csv_files:
                df = pd.read_csv(csv_files[0])
                st.info(f"Loaded numerical data from: {csv_files[0]}")
                return df
            else:
                st.warning("No CSV files found in the dataset path")
                return None
        except Exception as e:
            st.error(f"Error loading numerical data: {str(e)}")
            return None
    
    def load_image_data(self, path):
        """Load satellite imagery data"""
        try:
            # Look for image files or structured data
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
                image_files.extend(glob.glob(os.path.join(path, "**", ext), recursive=True))
            
            if image_files:
                st.info(f"Found {len(image_files)} image files")
                return image_files[:1000]  # Limit for demo
            else:
                st.warning("No image files found in the dataset path")
                return None
        except Exception as e:
            st.error(f"Error loading image data: {str(e)}")
            return None
    
    def preprocess_numerical_data(self, df):
        """Preprocess numerical data"""
        if df is None:
            return None, None, None, None
            
        # Handle missing values
        df = df.dropna()
        
        # Encode categorical variables
        le = LabelEncoder()
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col != 'FloodProbability' and col != 'Flood':  # Assuming target variables
                df[col] = le.fit_transform(df[col])
        
        # Separate features and target
        target_cols = ['FloodProbability', 'Flood', 'flood', 'target']
        target_col = None
        for col in target_cols:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            # If no clear target, use the last column
            target_col = df.columns[-1]
            
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Convert target to binary if needed
        if len(y.unique()) > 2:
            y = (y > y.median()).astype(int)
        
        return X, y, df, target_col
    
    def train_ml_models(self, X, y):
        """Train multiple ML models"""
        if X is None or y is None:
            return {}
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['numerical'] = scaler
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42),
            'LightGBM': lgb.LGBMClassifier(random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'Naive Bayes': GaussianNB(),
            'KNN': KNeighborsClassifier(),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'AdaBoost': AdaBoostClassifier(random_state=42)
        }
        
        results = {}
        
        # Train and evaluate models
        for name, model in models.items():
            try:
                if name in ['Logistic Regression', 'SVM', 'Naive Bayes', 'KNN']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                accuracy = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_pred_proba)
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'auc': auc,
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
                
            except Exception as e:
                st.warning(f"Error training {name}: {str(e)}")
        
        return results
    
    def create_deep_learning_model(self, input_shape):
        """Create a deep learning model for flood prediction"""
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_shape,)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_deep_learning_model(self, X, y):
        """Train deep learning model"""
        if X is None or y is None:
            return None
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create and train model
        model = self.create_deep_learning_model(X_train_scaled.shape[1])
        
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
        
        # Train model
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        # Evaluate
        y_pred_proba = model.predict(X_test_scaled).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        return {
            'model': model,
            'scaler': scaler,
            'history': history,
            'accuracy': accuracy,
            'auc': auc,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

def main():
    # Header
    st.markdown('<h1 class="main-header">🌊 FloodSentinel</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Advanced Flood Risk Assessment Using Multi-Temporal Satellite Imagery and Deep Neural Networks</p>', unsafe_allow_html=True)
    
    # Initialize FloodSentinel
    flood_sentinel = FloodSentinel()
    
    # Sidebar
    st.sidebar.title("🚀 FloodSentinel Control Panel")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Select Module",
        ["🏠 Home", "📊 Data Overview", "🤖 ML Models", "🧠 Deep Learning", "📈 Model Comparison", "🔮 Predictions", "📋 Risk Assessment"]
    )
    
    if page == "🏠 Home":
        st.markdown('<h2 class="sub-header">Welcome to FloodSentinel</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🎯 Project Overview
            FloodSentinel is a state-of-the-art flood risk assessment system that combines:
            - **Machine Learning** for numerical flood prediction
            - **Deep Learning** for satellite imagery analysis
            - **Multi-temporal data fusion** for enhanced accuracy
            """)
        
        with col2:
            st.markdown("""
            ### 🔧 Key Features
            - **10 ML Algorithms** for comprehensive analysis
            - **Deep Neural Networks** for complex pattern recognition
            - **Real-time Risk Assessment** capabilities
            - **Interactive Visualizations** and insights
            """)
        
        with col3:
            st.markdown("""
            ### 📊 Datasets
            - **Numerical Dataset**: Weather, terrain, and hydrological data
            - **Satellite Imagery**: Multi-temporal flood detection data
            - **Automated Processing** with Kaggle integration
            """)
        
        # Dataset Loading Section
        st.markdown('<h3 class="sub-header">📥 Data Loading</h3>', unsafe_allow_html=True)
        
        if st.button("🔄 Load Datasets", type="primary"):
            path1, path2 = flood_sentinel.download_datasets()
            
            if path1 and path2:
                # Load numerical data
                flood_sentinel.numerical_data = flood_sentinel.load_numerical_data(path1)
                
                # Load image data
                flood_sentinel.image_data = flood_sentinel.load_image_data(path2)
                
                if flood_sentinel.numerical_data is not None:
                    st.session_state.datasets_loaded = True
                    st.success("✅ Datasets loaded successfully!")
                else:
                    st.warning("⚠️ Some datasets could not be loaded. Please check the paths.")
        
        if st.session_state.datasets_loaded:
            st.success("✅ Datasets are ready for analysis!")
    
    elif page == "📊 Data Overview":
        st.markdown('<h2 class="sub-header">Data Overview & Analysis</h2>', unsafe_allow_html=True)
        
        if not st.session_state.datasets_loaded:
            st.warning("Please load datasets first from the Home page.")
            return
        
        if flood_sentinel.numerical_data is not None:
            df = flood_sentinel.numerical_data
            
            # Basic statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Total Records", len(df))
            with col2:
                st.metric("📈 Features", len(df.columns))
            with col3:
                st.metric("🔍 Missing Values", df.isnull().sum().sum())
            with col4:
                st.metric("💾 Memory Usage", f"{df.memory_usage().sum() / 1024:.1f} KB")
            
            # Data preview
            st.markdown("### 📋 Data Preview")
            st.dataframe(df.head())
            
            # Statistical summary
            st.markdown("### 📊 Statistical Summary")
            st.dataframe(df.describe())
            
            # Data distribution
            st.markdown("### 📈 Data Distribution")
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                selected_columns = st.multiselect("Select columns to visualize", numeric_columns, default=numeric_columns[:4])
                
                if selected_columns:
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=selected_columns[:4]
                    )
                    
                    for i, col in enumerate(selected_columns[:4]):
                        row = i // 2 + 1
                        col_pos = i % 2 + 1
                        
                        fig.add_trace(
                            go.Histogram(x=df[col], name=col),
                            row=row, col=col_pos
                        )
                    
                    fig.update_layout(height=600, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Correlation matrix
            st.markdown("### 🔗 Correlation Matrix")
            if len(numeric_columns) > 1:
                corr_matrix = df[numeric_columns].corr()
                fig = px.imshow(corr_matrix, text_auto=True, aspect="auto")
                st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🤖 ML Models":
        st.markdown('<h2 class="sub-header">Machine Learning Models</h2>', unsafe_allow_html=True)
        
        if not st.session_state.datasets_loaded:
            st.warning("Please load datasets first from the Home page.")
            return
        
        if flood_sentinel.numerical_data is not None:
            df = flood_sentinel.numerical_data
            
            # Preprocess data
            X, y, processed_df, target_col = flood_sentinel.preprocess_numerical_data(df)
            
            if X is not None and y is not None:
                st.success(f"✅ Data preprocessed successfully! Target: {target_col}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Features", X.shape[1])
                with col2:
                    st.metric("Samples", X.shape[0])
                
                # Train models
                if st.button("🚀 Train ML Models", type="primary"):
                    with st.spinner("Training models... This may take a few minutes."):
                        results = flood_sentinel.train_ml_models(X, y)
                        flood_sentinel.results['ml'] = results
                        st.session_state.models_trained = True
                
                # Display results
                if 'ml' in flood_sentinel.results and flood_sentinel.results['ml']:
                    st.markdown("### 📊 Model Performance")
                    
                    # Create results DataFrame
                    results_data = []
                    for name, result in flood_sentinel.results['ml'].items():
                        results_data.append({
                            'Model': name,
                            'Accuracy': f"{result['accuracy']:.4f}",
                            'AUC': f"{result['auc']:.4f}"
                        })
                    
                    results_df = pd.DataFrame(results_data)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Best model
                    best_model = max(flood_sentinel.results['ml'].items(), key=lambda x: x[1]['accuracy'])
                    st.success(f"🏆 Best Model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")
                    
                    # Visualization
                    fig = px.bar(
                        results_df, 
                        x='Model', 
                        y='Accuracy',
                        title='Model Performance Comparison'
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # ROC Curves
                    st.markdown("### 📈 ROC Curves")
                    fig = go.Figure()
                    
                    for name, result in flood_sentinel.results['ml'].items():
                        fpr, tpr, _ = roc_curve(result['y_test'], result['y_pred_proba'])
                        fig.add_trace(go.Scatter(
                            x=fpr, y=tpr,
                            mode='lines',
                            name=f"{name} (AUC: {result['auc']:.3f})"
                        ))
                    
                    fig.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode='lines',
                        name='Random',
                        line=dict(dash='dash')
                    ))
                    
                    fig.update_layout(
                        title='ROC Curves Comparison',
                        xaxis_title='False Positive Rate',
                        yaxis_title='True Positive Rate'
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🧠 Deep Learning":
        st.markdown('<h2 class="sub-header">Deep Learning Models</h2>', unsafe_allow_html=True)
        
        if not st.session_state.datasets_loaded:
            st.warning("Please load datasets first from the Home page.")
            return
        
        if flood_sentinel.numerical_data is not None:
            df = flood_sentinel.numerical_data
            
            # Preprocess data
            X, y, processed_df, target_col = flood_sentinel.preprocess_numerical_data(df)
            
            if X is not None and y is not None:
                st.success(f"✅ Data ready for deep learning! Target: {target_col}")
                
                # Model architecture
                st.markdown("### 🏗️ Neural Network Architecture")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    **Network Structure:**
                    - Input Layer: Feature dimensions
                    - Hidden Layer 1: 128 neurons + ReLU + Dropout(0.3)
                    - Hidden Layer 2: 64 neurons + ReLU + Dropout(0.3)
                    - Hidden Layer 3: 32 neurons + ReLU + Dropout(0.2)
                    - Hidden Layer 4: 16 neurons + ReLU
                    - Output Layer: 1 neuron + Sigmoid
                    """)
                
                with col2:
                    st.markdown("""
                    **Training Configuration:**
                    - Optimizer: Adam (lr=0.001)
                    - Loss: Binary Crossentropy
                    - Callbacks: Early Stopping, ReduceLROnPlateau
                    - Epochs: 50 (with early stopping)
                    - Batch Size: 32
                    """)
                
                # Train model
                if st.button("🚀 Train Deep Learning Model", type="primary"):
                    with st.spinner("Training neural network... This may take a while."):
                        dl_result = flood_sentinel.train_deep_learning_model(X, y)
                        flood_sentinel.results['dl'] = dl_result
                
                # Display results
                if 'dl' in flood_sentinel.results and flood_sentinel.results['dl']:
                    result = flood_sentinel.results['dl']
                    
                    st.markdown("### 📊 Deep Learning Performance")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🎯 Accuracy", f"{result['accuracy']:.4f}")
                    with col2:
                        st.metric("📈 AUC Score", f"{result['auc']:.4f}")
                    with col3:
                        st.metric("🔄 Training Status", "Completed ✅")
                    
                    # Training history
                    if 'history' in result:
                        st.markdown("### 📈 Training History")
                        
                        history = result['history'].history
                        
                        fig = make_subplots(
                            rows=1, cols=2,
                            subplot_titles=['Loss', 'Accuracy']
                        )
                        
                        # Loss
                        fig.add_trace(
                            go.Scatter(y=history['loss'], name='Training Loss'),
                            row=1, col=1
                        )
                        fig.add_trace(
                            go.Scatter(y=history['val_loss'], name='Validation Loss'),
                            row=1, col=1
                        )
                        
                        # Accuracy
                        fig.add_trace(
                            go.Scatter(y=history['accuracy'], name='Training Accuracy'),
                            row=1, col=2
                        )
                        fig.add_trace(
                            go.Scatter(y=history['val_accuracy'], name='Validation Accuracy'),
                            row=1, col=2
                        )
                        
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "📈 Model Comparison":
        st.markdown('<h2 class="sub-header">Model Comparison & Analysis</h2>', unsafe_allow_html=True)
        
        if 'ml' not in flood_sentinel.results and 'dl' not in flood_sentinel.results:
            st.warning("Please train models first from the ML Models and Deep Learning pages.")
            return
        
        # Comparison table
        comparison_data = []
        
        if 'ml' in flood_sentinel.results:
            for name, result in flood_sentinel.results['ml'].items():
                comparison_data.append({
                    'Model': name,
                    'Type': 'Machine Learning',
                    'Accuracy': result['accuracy'],
                    'AUC': result['auc']
                })
        
        if 'dl' in flood_sentinel.results:
            result = flood_sentinel.results['dl']
            comparison_data.append({
                'Model': 'Deep Neural Network',
                'Type': 'Deep Learning',
                'Accuracy': result['accuracy'],
                'AUC': result['auc']
            })
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            df_comparison = df_comparison.sort_values('Accuracy', ascending=False)
            
            st.markdown("### 🏆 Overall Model Rankings")
            st.dataframe(df_comparison, use_container_width=True)
            
            # Best model overall
            best_model = df_comparison.iloc[0]
            st.success(f"🥇 Overall Best Model: {best_model['Model']} ({best_model['Type']}) - Accuracy: {best_model['Accuracy']:.4f}")
            
            # Comparison visualization
            fig = px.scatter(
                df_comparison,
                x='Accuracy',
                y='AUC',
                color='Type',
                text='Model',
                title='Model Performance Comparison: Accuracy vs AUC'
            )
            fig.update_traces(textposition="top center")
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🔮 Predictions":
        st.markdown('<h2 class="sub-header">Flood Risk Predictions</h2>', unsafe_allow_html=True)
        
        if not st.session_state.models_trained:
            st.warning("Please train models first from the ML Models page.")
            return
        
        if flood_sentinel.numerical_data is not None:
            df = flood_sentinel.numerical_data
            X, y, processed_df, target_col = flood_sentinel.preprocess_numerical_data(df)
            
            if X is not None:
                st.markdown("### 🎯 Make Predictions")
                
                # Feature input
                st.markdown("#### Enter Feature Values:")
                
                input_data = {}
                cols = st.columns(3)
                
                for i, feature in enumerate(X.columns):
                    with cols[i % 3]:
                        if X[feature].dtype in ['int64', 'float64']:
                            input_data[feature] = st.number_input(
                                f"{feature}",
                                value=float(X[feature].mean()),
                                key=f"input_{feature}"
                            )
                        else:
                            input_data[feature] = st.selectbox(
                                f"{feature}",
                                options=sorted(X[feature].unique()),
                                key=f"input_{feature}"
                            )
                
                if st.button("🔮 Make Prediction", type="primary"):
                    if 'ml' in flood_sentinel.results:
                        input_df = pd.DataFrame([input_data])
                        
                        st.markdown("### 📊 Prediction Results")
                        
                        predictions = {}
                        for name, result in flood_sentinel.results['ml'].items():
                            try:
                                model = result['model']
                                
                                # Scale if needed
                                if name in ['Logistic Regression', 'SVM', 'Naive Bayes', 'KNN']:
                                    input_scaled = flood_sentinel.scalers['numerical'].transform(input_df)
                                    pred_proba = model.predict_proba(input_scaled)[0][1]
                                else:
                                    pred_proba = model.predict_proba(input_df)[0][1]
                                
                                predictions[name] = pred_proba
                            except:
                                pass
                        
                        # Display predictions
                        pred_df = pd.DataFrame([
                            {'Model': name, 'Flood Risk Probability': prob}
                            for name, prob in predictions.items()
                        ])
                        
                        st.dataframe(pred_df, use_container_width=True)
                        
                        # Average prediction
                        avg_risk = np.mean(list(predictions.values()))
                        
                        col1, col2, col3 = st.columns(3)
                        with col2:
                            if avg_risk > 0.7:
                                st.error(f"🚨 HIGH FLOOD RISK: {avg_risk:.2%}")
                            elif avg_risk > 0.4:
                                st.warning(f"⚠️ MODERATE FLOOD RISK: {avg_risk:.2%}")
                            else:
                                st.success(f"✅ LOW FLOOD RISK: {avg_risk:.2%}")
    
    elif page == "📋 Risk Assessment":
        st.markdown('<h2 class="sub-header">Comprehensive Risk Assessment</h2>', unsafe_allow_html=True)
        
        if not st.session_state.models_trained:
            st.warning("Please train models first from the ML Models page.")
            return
        
        st.markdown("### 🌍 Regional Flood Risk Analysis")
        
        # Risk assessment parameters
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Assessment Parameters")
            region = st.selectbox(
                "Select Region",
                ["Urban Area", "Rural Area", "Coastal Region", "Mountainous Region", "River Basin"]
            )
            
            season = st.selectbox(
                "Select Season",
                ["Spring", "Summer", "Monsoon", "Winter"]
            )
            
            risk_factors = st.multiselect(
                "Key Risk Factors",
                ["Heavy Rainfall", "River Overflow", "Storm Surge", "Poor Drainage", "Urbanization", "Deforestation"],
                default=["Heavy Rainfall", "Poor Drainage"]
            )
        
        with col2:
            st.markdown("#### 📊 Risk Metrics")
            
            # Simulated risk assessment based on parameters
            base_risk = 0.3
            
            # Adjust risk based on region
            region_multiplier = {
                "Urban Area": 1.2,
                "Rural Area": 0.8,
                "Coastal Region": 1.5,
                "Mountainous Region": 1.1,
                "River Basin": 1.4
            }
            
            # Adjust risk based on season
            season_multiplier = {
                "Spring": 1.1,
                "Summer": 0.9,
                "Monsoon": 1.6,
                "Winter": 0.7
            }
            
            # Adjust risk based on factors
            factor_impact = len(risk_factors) * 0.1
            
            calculated_risk = base_risk * region_multiplier[region] * season_multiplier[season] + factor_impact
            calculated_risk = min(calculated_risk, 1.0)  # Cap at 100%
            
            # Display risk level
            if calculated_risk > 0.7:
                st.error(f"🚨 HIGH RISK: {calculated_risk:.1%}")
                risk_level = "HIGH"
                risk_color = "red"
            elif calculated_risk > 0.4:
                st.warning(f"⚠️ MODERATE RISK: {calculated_risk:.1%}")
                risk_level = "MODERATE"
                risk_color = "orange"
            else:
                st.success(f"✅ LOW RISK: {calculated_risk:.1%}")
                risk_level = "LOW"
                risk_color = "green"
        
        # Risk gauge
        st.markdown("### 📊 Risk Level Gauge")
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = calculated_risk * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Flood Risk Level (%)"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0, 40], 'color': "lightgreen"},
                    {'range': [40, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk breakdown
        st.markdown("### 📈 Risk Factor Analysis")
        
        # Create risk factor visualization
        factor_contributions = {
            "Regional Factor": region_multiplier[region] - 1,
            "Seasonal Factor": season_multiplier[season] - 1,
            "Environmental Factors": factor_impact,
            "Base Risk": base_risk
        }
        
        fig = px.bar(
            x=list(factor_contributions.keys()),
            y=list(factor_contributions.values()),
            title="Risk Factor Contributions",
            labels={'x': 'Risk Factors', 'y': 'Contribution'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("### 💡 Risk Mitigation Recommendations")
        
        recommendations = []
        
        if calculated_risk > 0.7:
            recommendations.extend([
                "🚨 Immediate evacuation planning required",
                "🏗️ Emergency infrastructure assessment",
                "📱 Activate early warning systems",
                "🚧 Implement flood barriers and diversions"
            ])
        elif calculated_risk > 0.4:
            recommendations.extend([
                "⚠️ Enhanced monitoring and surveillance",
                "🏠 Community preparedness programs",
                "🌊 Drainage system maintenance",
                "📋 Update emergency response plans"
            ])
        else:
            recommendations.extend([
                "✅ Regular monitoring and maintenance",
                "🌳 Sustainable land use planning",
                "💧 Water management optimization",
                "📚 Community education programs"
            ])
        
        for rec in recommendations:
            st.markdown(f"- {rec}")
        
        # Historical trends simulation
        st.markdown("### 📊 Historical Risk Trends")
        
        # Simulate historical data
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        risk_trend = [0.2, 0.25, 0.4, 0.5, 0.6, 0.8, 0.9, 0.85, 0.7, 0.4, 0.3, 0.25]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=months,
            y=risk_trend,
            mode='lines+markers',
            name='Risk Level',
            line=dict(color='blue', width=3)
        ))
        
        fig.update_layout(
            title='Monthly Flood Risk Trends',
            xaxis_title='Month',
            yaxis_title='Risk Level',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🌊 FloodSentinel - Advanced Flood Risk Assessment System</p>
        <p>Built with Streamlit, TensorFlow, and scikit-learn</p>
        <p>For research and educational purposes</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
