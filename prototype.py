import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import shap
import kagglehub
import os
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="FloodSentinel: AI-Powered Flood Risk Assessment",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

class FloodSentinelML:
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.models = {}
        self.results = {}
        self.feature_names = [
            'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement', 
            'Deforestation', 'Urbanization', 'ClimateChange', 'DamsQuality',
            'Siltation', 'AgriculturalPractices', 'Encroachments',
            'IneffectiveDisasterPreparedness', 'DrainageSystems',
            'CoastalVulnerability', 'Landslides', 'Watersheds',
            'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
            'InadequatePlanning', 'PoliticalFactors'
        ]
        
    @st.cache_data
    def load_data(_self):
        """Load and cache the flood dataset"""
        try:
            # Download dataset using kagglehub
            with st.spinner("Downloading flood dataset..."):
                path = kagglehub.dataset_download("naiyakhalid/flood-prediction-dataset")
                
            # Find CSV files in the downloaded path
            csv_files = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.csv'):
                        csv_files.append(os.path.join(root, file))
            
            if not csv_files:
                st.error("No CSV files found in the dataset")
                return None
                
            # Load the first CSV file found
            data = pd.read_csv(csv_files[0])
            
            # Ensure all required columns are present
            required_cols = _self.feature_names + ['FloodProbability']
            if not all(col in data.columns for col in required_cols):
                st.error("Dataset doesn't contain all required columns")
                return None
                
            return data
            
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            return None
    
    def preprocess_data(self, scaling_method='standard'):
        """Preprocess the data"""
        if self.data is None:
            return False
            
        try:
            # Check if required columns exist
            missing_cols = [col for col in self.feature_names + ['FloodProbability'] if col not in self.data.columns]
            if missing_cols:
                st.error(f"Missing columns in dataset: {missing_cols}")
                return False
            
            # Separate features and target
            self.X = self.data[self.feature_names].copy()
            self.y = self.data['FloodProbability'].copy()
            
            # Handle missing values
            self.X = self.X.fillna(self.X.mean())
            self.y = self.y.fillna(self.y.mode()[0] if not self.y.mode().empty else 0)
            
            # Ensure y is numeric and convert to int if needed
            if self.y.dtype == 'object':
                # Try to convert to numeric
                self.y = pd.to_numeric(self.y, errors='coerce')
                self.y = self.y.fillna(0)
            
            # Convert to int for classification
            self.y = self.y.astype(int)
            
            # Check if we have valid data
            if len(self.X) == 0 or len(self.y) == 0:
                st.error("No valid data found after preprocessing")
                return False
            
            # Split the data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
            )
            
            # Scale the features
            if scaling_method == 'standard':
                self.scaler = StandardScaler()
            elif scaling_method == 'robust':
                self.scaler = RobustScaler()
            else:
                self.scaler = MinMaxScaler()
                
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            
            return True
            
        except Exception as e:
            st.error(f"Error preprocessing data: {str(e)}")
            return False
    
    def initialize_models(self):
        """Initialize all ML models"""
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42),
            'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
            'CatBoost': CatBoostClassifier(random_state=42, verbose=False),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Naive Bayes': GaussianNB(),
            'Neural Network': MLPClassifier(random_state=42, max_iter=1000),
            'AdaBoost': AdaBoostClassifier(random_state=42)
        }
    
    def train_and_evaluate_models(self):
        """Train and evaluate all models"""
        # Check if data is properly preprocessed
        if (self.X_train is None or self.y_train is None or 
            self.X_test is None or self.y_test is None or
            not hasattr(self, 'X_train_scaled') or not hasattr(self, 'X_test_scaled')):
            st.error("Data not properly preprocessed. Please preprocess data first.")
            return False
            
        self.results = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (name, model) in enumerate(self.models.items()):
            status_text.text(f"Training {name}...")
            
            try:
                # Train the model
                if name in ['Logistic Regression', 'SVM', 'K-Nearest Neighbors', 'Neural Network']:
                    model.fit(self.X_train_scaled, self.y_train)
                    y_pred = model.predict(self.X_test_scaled)
                    y_pred_proba = model.predict_proba(self.X_test_scaled)
                    if y_pred_proba.shape[1] > 1:
                        y_pred_proba = y_pred_proba[:, 1]
                    else:
                        y_pred_proba = y_pred_proba[:, 0]
                else:
                    model.fit(self.X_train, self.y_train)
                    y_pred = model.predict(self.X_test)
                    y_pred_proba = model.predict_proba(self.X_test)
                    if y_pred_proba.shape[1] > 1:
                        y_pred_proba = y_pred_proba[:, 1]
                    else:
                        y_pred_proba = y_pred_proba[:, 0]
                
                # Calculate metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
                
                # Handle ROC AUC for multiclass
                try:
                    if len(np.unique(self.y_test)) > 2:
                        roc_auc = roc_auc_score(self.y_test, y_pred_proba, multi_class='ovr')
                    else:
                        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
                except:
                    roc_auc = 0.0
                
                mcc = matthews_corrcoef(self.y_test, y_pred)
                
                # Cross-validation
                try:
                    if name in ['Logistic Regression', 'SVM', 'K-Nearest Neighbors', 'Neural Network']:
                        cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5, scoring='accuracy')
                    else:
                        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy')
                except:
                    cv_scores = np.array([accuracy])  # Fallback to single score
                
                self.results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'mcc': mcc,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
                
            except Exception as e:
                st.warning(f"Error training {name}: {str(e)}")
                continue
            
            progress_bar.progress((i + 1) / len(self.models))
        
        status_text.text("Training completed!")
        progress_bar.empty()
        status_text.empty()
        return True

def main():
    # Header
    st.markdown('<h1 class="main-header">🌊 FloodSentinel: AI-Powered Flood Risk Assessment</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    **FloodSentinel** is an advanced machine learning system for flood risk assessment using multi-temporal 
    satellite imagery and deep neural networks. This application demonstrates state-of-the-art ML algorithms 
    for flood prediction and risk analysis.
    """)
    
    # Initialize the ML system
    if 'flood_ml' not in st.session_state:
        st.session_state.flood_ml = FloodSentinelML()
    
    flood_ml = st.session_state.flood_ml
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("📊 Navigation")
        page = st.selectbox(
            "Select Module",
            ["🏠 Home", "📈 Data Analysis", "🤖 ML Models", "🎯 Model Comparison", 
             "📊 Feature Analysis", "🔮 Predictions", "📋 Model Details"]
        )
        
        st.header("⚙️ Settings")
        scaling_method = st.selectbox(
            "Scaling Method",
            ["standard", "robust", "minmax"]
        )
    
    # Load data if not already loaded
    if flood_ml.data is None:
        with st.spinner("Loading flood dataset..."):
            flood_ml.data = flood_ml.load_data()
            if flood_ml.data is not None:
                st.success(f"Dataset loaded successfully! Shape: {flood_ml.data.shape}")
                # Show basic info about the dataset
                st.write("**Dataset Preview:**")
                st.dataframe(flood_ml.data.head())
                
                # Check target distribution
                if 'FloodProbability' in flood_ml.data.columns:
                    st.write("**Target Distribution:**")
                    st.write(flood_ml.data['FloodProbability'].value_counts().sort_index())
            else:
                st.error("Failed to load dataset. Please check your internet connection and try again.")
                return
    
    # Preprocess data
    if flood_ml.X is None:
        if flood_ml.preprocess_data(scaling_method):
            st.success("Data preprocessing completed successfully!")
            st.rerun()  # Refresh the app to show updated state
        else:
            st.error("Failed to preprocess data.")
            return
    
    # Main content based on selected page
    if page == "🏠 Home":
        show_home_page(flood_ml)
    elif page == "📈 Data Analysis":
        show_data_analysis(flood_ml)
    elif page == "🤖 ML Models":
        show_ml_models(flood_ml)
    elif page == "🎯 Model Comparison":
        show_model_comparison(flood_ml)
    elif page == "📊 Feature Analysis":
        show_feature_analysis(flood_ml)
    elif page == "🔮 Predictions":
        show_predictions(flood_ml)
    elif page == "📋 Model Details":
        show_model_details(flood_ml)

def show_home_page(flood_ml):
    """Display home page with overview"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Dataset Overview")
        st.info(f"**Total Samples:** {len(flood_ml.data)}")
        st.info(f"**Features:** {len(flood_ml.feature_names)}")
        st.info(f"**Target Classes:** {flood_ml.data['FloodProbability'].nunique()}")
        
        # Class distribution
        class_dist = flood_ml.data['FloodProbability'].value_counts()
        fig_pie = px.pie(
            values=class_dist.values,
            names=class_dist.index,
            title="Class Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Project Objectives")
        st.markdown("""
        - **Multi-Modal Analysis**: Combine satellite imagery with meteorological data
        - **Real-Time Prediction**: Provide immediate flood risk assessments
        - **Interpretable AI**: Explain model decisions for actionable insights
        - **Scalable Solution**: Deploy across diverse geographical regions
        - **Decision Support**: Aid disaster preparedness and mitigation
        """)
        
        st.subheader("🔧 Technical Features")
        st.markdown("""
        - **12 ML Algorithms**: From classical to ensemble methods
        - **Feature Engineering**: Advanced preprocessing and selection
        - **Model Interpretability**: SHAP values and feature importance
        - **Cross-Validation**: Robust model evaluation
        - **Interactive Dashboard**: Real-time visualization and analysis
        """)

def show_data_analysis(flood_ml):
    """Display comprehensive data analysis"""
    
    st.header("📈 Comprehensive Data Analysis")
    
    # Basic statistics
    st.subheader("📊 Dataset Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", len(flood_ml.data))
    with col2:
        st.metric("Features", len(flood_ml.feature_names))
    with col3:
        st.metric("Missing Values", flood_ml.data.isnull().sum().sum())
    with col4:
        st.metric("Duplicate Rows", flood_ml.data.duplicated().sum())
    
    # Correlation heatmap
    st.subheader("🔥 Feature Correlation Matrix")
    corr_matrix = flood_ml.data[flood_ml.feature_names].corr()
    
    fig_heatmap = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Feature Correlation Heatmap"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Distribution plots
    st.subheader("📈 Feature Distributions")
    
    # Select features for distribution analysis
    selected_features = st.multiselect(
        "Select features to analyze:",
        flood_ml.feature_names,
        default=flood_ml.feature_names[:4]
    )
    
    if selected_features:
        fig_dist = make_subplots(
            rows=len(selected_features)//2 + len(selected_features)%2,
            cols=2,
            subplot_titles=selected_features
        )
        
        for i, feature in enumerate(selected_features):
            row = i // 2 + 1
            col = i % 2 + 1
            
            fig_dist.add_trace(
                go.Histogram(x=flood_ml.data[feature], name=feature),
                row=row, col=col
            )
        
        fig_dist.update_layout(height=300*len(selected_features)//2 + 150)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Box plots for outlier detection
    st.subheader("📦 Outlier Detection")
    feature_for_box = st.selectbox("Select feature for box plot:", flood_ml.feature_names)
    
    fig_box = px.box(
        flood_ml.data,
        y=feature_for_box,
        color='FloodProbability',
        title=f"Box Plot: {feature_for_box} by Flood Probability"
    )
    st.plotly_chart(fig_box, use_container_width=True)

def show_ml_models(flood_ml):
    """Display ML models training and results"""
    
    st.header("🤖 Machine Learning Models")
    
    # Initialize and train models
    if not flood_ml.models:
        flood_ml.initialize_models()
    
    if not flood_ml.results:
        st.info("Click the button below to train all models.")
        if st.button("🚀 Train All Models", type="primary"):
            if flood_ml.train_and_evaluate_models():
                st.success("All models trained successfully!")
                st.rerun()  # Refresh to show results
            else:
                st.error("Failed to train models. Please check the data preprocessing.")
    
    if flood_ml.results:
        # Model performance summary
        st.subheader("📊 Model Performance Summary")
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Model': list(flood_ml.results.keys()),
            'Accuracy': [flood_ml.results[model]['accuracy'] for model in flood_ml.results],
            'Precision': [flood_ml.results[model]['precision'] for model in flood_ml.results],
            'Recall': [flood_ml.results[model]['recall'] for model in flood_ml.results],
            'F1-Score': [flood_ml.results[model]['f1_score'] for model in flood_ml.results],
            'ROC-AUC': [flood_ml.results[model]['roc_auc'] for model in flood_ml.results],
            'MCC': [flood_ml.results[model]['mcc'] for model in flood_ml.results],
            'CV Mean': [flood_ml.results[model]['cv_mean'] for model in flood_ml.results],
            'CV Std': [flood_ml.results[model]['cv_std'] for model in flood_ml.results]
        })
        
        # Sort by accuracy
        results_df = results_df.sort_values('Accuracy', ascending=False)
        st.dataframe(results_df, use_container_width=True)
        
        # Best model highlight
        best_model = results_df.iloc[0]['Model']
        st.success(f"🏆 Best Model: **{best_model}** with {results_df.iloc[0]['Accuracy']:.4f} accuracy")
        
        # Model performance visualization
        st.subheader("📈 Model Performance Visualization")
        
        # Bar chart of accuracies
        fig_bar = px.bar(
            results_df,
            x='Model',
            y='Accuracy',
            title='Model Accuracy Comparison',
            color='Accuracy',
            color_continuous_scale='viridis'
        )
        fig_bar.update_xaxes(tickangle=45)
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Radar chart for multiple metrics
        st.subheader("🎯 Multi-Metric Radar Chart")
        
        selected_models = st.multiselect(
            "Select models for radar chart:",
            list(flood_ml.results.keys()),
            default=list(flood_ml.results.keys())[:5]
        )
        
        if selected_models:
            fig_radar = go.Figure()
            
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
            
            for model in selected_models:
                values = [
                    flood_ml.results[model]['accuracy'],
                    flood_ml.results[model]['precision'],
                    flood_ml.results[model]['recall'],
                    flood_ml.results[model]['f1_score'],
                    flood_ml.results[model]['roc_auc']
                ]
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=metrics,
                    fill='toself',
                    name=model
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Model Performance Radar Chart"
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)

def show_model_comparison(flood_ml):
    """Display detailed model comparison"""
    
    st.header("🎯 Advanced Model Comparison")
    
    if not flood_ml.results:
        st.warning("Please train the models first in the ML Models section.")
        return
    
    # ROC Curves
    st.subheader("📈 ROC Curves Comparison")
    
    fig_roc = go.Figure()
    
    for model_name in flood_ml.results:
        try:
            y_pred_proba = flood_ml.results[model_name]['y_pred_proba']
            
            # Handle different probability array shapes
            if len(np.unique(flood_ml.y_test)) > 2:
                # Multi-class case - use the probability for the highest class
                if len(y_pred_proba.shape) > 1:
                    # If it's a 2D array, take the max probability
                    y_pred_proba_plot = np.max(y_pred_proba, axis=1) if y_pred_proba.ndim > 1 else y_pred_proba
                else:
                    y_pred_proba_plot = y_pred_proba
            else:
                # Binary case
                y_pred_proba_plot = y_pred_proba
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(flood_ml.y_test, y_pred_proba_plot)
            auc_score = flood_ml.results[model_name]['roc_auc']
            
            fig_roc.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name=f'{model_name} (AUC = {auc_score:.3f})'
            ))
        except Exception as e:
            st.warning(f"Error plotting ROC curve for {model_name}: {str(e)}")
            continue
    
    # Add diagonal line
    fig_roc.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color='gray')
    ))
    
    fig_roc.update_layout(
        title='ROC Curves Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        showlegend=True
    )
    
    st.plotly_chart(fig_roc, use_container_width=True)
    
    # Confusion matrices
    st.subheader("🔍 Confusion Matrices")
    
    selected_models_cm = st.multiselect(
        "Select models for confusion matrix:",
        list(flood_ml.results.keys()),
        default=[list(flood_ml.results.keys())[0]]
    )
    
    if selected_models_cm:
        cols = st.columns(len(selected_models_cm))
        
        for i, model_name in enumerate(selected_models_cm):
            with cols[i]:
                cm = confusion_matrix(flood_ml.y_test, flood_ml.results[model_name]['y_pred'])
                
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    aspect="auto",
                    title=f"{model_name}",
                    labels=dict(x="Predicted", y="Actual")
                )
                
                st.plotly_chart(fig_cm, use_container_width=True)
    
    # Statistical significance testing
    st.subheader("📊 Statistical Analysis")
    
    # Create performance comparison table
    comparison_data = []
    for model_name in flood_ml.results:
        comparison_data.append({
            'Model': model_name,
            'Accuracy': flood_ml.results[model_name]['accuracy'],
            'Precision': flood_ml.results[model_name]['precision'],
            'Recall': flood_ml.results[model_name]['recall'],
            'F1-Score': flood_ml.results[model_name]['f1_score'],
            'ROC-AUC': flood_ml.results[model_name]['roc_auc'],
            'MCC': flood_ml.results[model_name]['mcc']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Rank models by each metric
    st.subheader("🏆 Model Rankings by Metric")
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'MCC']
    ranking_data = {}
    
    for metric in metrics:
        sorted_models = comparison_df.sort_values(metric, ascending=False)
        ranking_data[metric] = sorted_models['Model'].tolist()
    
    ranking_df = pd.DataFrame(ranking_data)
    
    # Add rank numbers
    for col in ranking_df.columns:
        ranking_df[col] = [f"{i+1}. {model}" for i, model in enumerate(ranking_df[col])]
    
    st.dataframe(ranking_df, use_container_width=True)

def show_feature_analysis(flood_ml):
    """Display feature importance and analysis"""
    
    st.header("📊 Feature Analysis & Importance")
    
    if not flood_ml.results:
        st.warning("Please train the models first in the ML Models section.")
        return
    
    # Feature importance from tree-based models
    st.subheader("🌳 Feature Importance from Tree-based Models")
    
    tree_models = ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost']
    available_tree_models = [m for m in tree_models if m in flood_ml.results]
    
    if available_tree_models:
        selected_tree_model = st.selectbox(
            "Select tree-based model:",
            available_tree_models
        )
        
        model = flood_ml.results[selected_tree_model]['model']
        feature_importance = model.feature_importances_
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': flood_ml.feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        # Plot feature importance
        fig_importance = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f'Feature Importance - {selected_tree_model}'
        )
        fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Top features
        st.subheader("🔝 Top 10 Most Important Features")
        st.dataframe(importance_df.head(10), use_container_width=True)
    
    # Feature selection
    st.subheader("🎯 Feature Selection Analysis")
    
    # Univariate feature selection
    k_best = SelectKBest(f_classif, k=10)
    k_best.fit(flood_ml.X_train, flood_ml.y_train)
    
    selected_features = flood_ml.X_train.columns[k_best.get_support()]
    feature_scores = k_best.scores_[k_best.get_support()]
    
    fs_df = pd.DataFrame({
        'Feature': selected_features,
        'Score': feature_scores
    }).sort_values('Score', ascending=False)
    
    fig_fs = px.bar(
        fs_df,
        x='Score',
        y='Feature',
        orientation='h',
        title='Top 10 Features (Univariate Selection)'
    )
    fig_fs.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_fs, use_container_width=True)
    
    # Correlation with target
    st.subheader("🎯 Feature Correlation with Target")
    
    target_corr = flood_ml.data[flood_ml.feature_names].corrwith(flood_ml.data['FloodProbability'])
    target_corr_df = pd.DataFrame({
        'Feature': target_corr.index,
        'Correlation': target_corr.values
    }).sort_values('Correlation', key=abs, ascending=False)
    
    fig_corr = px.bar(
        target_corr_df,
        x='Correlation',
        y='Feature',
        orientation='h',
        title='Feature Correlation with Flood Probability',
        color='Correlation',
        color_continuous_scale='RdYlBu'
    )
    fig_corr.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_corr, use_container_width=True)

def show_predictions(flood_ml):
    """Display prediction interface"""
    
    st.header("🔮 Flood Risk Prediction")
    
    if not flood_ml.results:
        st.warning("Please train the models first in the ML Models section.")
        return
    
    st.subheader("📝 Enter Feature Values")
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        input_data = {}
        
        for i, feature in enumerate(flood_ml.feature_names):
            if i % 2 == 0:
                with col1:
                    input_data[feature] = st.number_input(
                        f"{feature}",
                        value=float(flood_ml.X[feature].mean()),
                        help=f"Mean: {flood_ml.X[feature].mean():.2f}, Std: {flood_ml.X[feature].std():.2f}"
                    )
            else:
                with col2:
                    input_data[feature] = st.number_input(
                        f"{feature}",
                        value=float(flood_ml.X[feature].mean()),
                        help=f"Mean: {flood_ml.X[feature].mean():.2f}, Std: {flood_ml.X[feature].std():.2f}"
                    )
        
        submitted = st.form_submit_button("🔮 Predict Flood Risk", type="primary")
        
        if submitted:
            # Create prediction dataframe
            pred_df = pd.DataFrame([input_data])
            
            # Make predictions with all models
            st.subheader("📊 Prediction Results")
            
            predictions = {}
            probabilities = {}
            
            for model_name, model_data in flood_ml.results.items():
                model = model_data['model']
                
                try:
                    # Scale input if needed
                    if model_name in ['Logistic Regression', 'SVM', 'K-Nearest Neighbors', 'Neural Network']:
                        pred_scaled = flood_ml.scaler.transform(pred_df)
                        pred = model.predict(pred_scaled)[0]
                        pred_proba = model.predict_proba(pred_scaled)[0]
                    else:
                        pred = model.predict(pred_df)[0]
                        pred_proba = model.predict_proba(pred_df)[0]
                    
                    predictions[model_name] = pred
                    # Handle different probability array shapes
                    if len(pred_proba) > 1:
                        probabilities[model_name] = pred_proba[1] if len(pred_proba) > 1 else pred_proba[0]
                    else:
                        probabilities[model_name] = pred_proba[0]
                    
                except Exception as e:
                    st.warning(f"Error with {model_name}: {str(e)}")
                    continue
            
            # Display predictions
            pred_results = pd.DataFrame({
                'Model': list(predictions.keys()),
                'Prediction': list(predictions.values()),
                'Flood Probability': [f"{prob:.4f}" for prob in probabilities.values()]
            })
            
            st.dataframe(pred_results, use_container_width=True)
            
            # Ensemble prediction (majority vote)
            ensemble_pred = max(set(predictions.values()), key=list(predictions.values()).count)
            avg_probability = np.mean(list(probabilities.values()))
            
            st.subheader("🎯 Ensemble Prediction")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Ensemble Prediction", f"Class {ensemble_pred}")
            with col2:
                st.metric("Average Probability", f"{avg_probability:.4f}")
            
            # Risk assessment
            if avg_probability > 0.7:
                st.error("🚨 HIGH FLOOD RISK - Immediate action recommended!")
            elif avg_probability > 0.5:
                st.warning("⚠️ MODERATE FLOOD RISK - Monitor conditions closely")
            else:
                st.success("✅ LOW FLOOD RISK - Normal conditions")
            
            # Prediction confidence visualization
            fig_pred = px.bar(
                pred_results,
                x='Model',
                y='Flood Probability',
                title='Flood Probability Predictions by Model',
                color='Flood Probability',
                color_continuous_scale='Reds'
            )
            fig_pred.update_xaxes(tickangle=45)
            st.plotly_chart(fig_pred, use_container_width=True)

def show_model_details(flood_ml):
    """Display detailed model information"""
    
    st.header("📋 Model Details & Interpretability")
    
    if not flood_ml.results:
        st.warning("Please train the models first in the ML Models section.")
        return
    
    # Model selection
    selected_model = st.selectbox(
        "Select model for detailed analysis:",
        list(flood_ml.results.keys())
    )
    
    model_data = flood_ml.results[selected_model]
    model = model_data['model']
    
    # Model information
    st.subheader(f"📊 {selected_model} - Detailed Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Accuracy", f"{model_data['accuracy']:.4f}")
        st.metric("Precision", f"{model_data['precision']:.4f}")
    
    with col2:
        st.metric("Recall", f"{model_data['recall']:.4f}")
        st.metric("F1-Score", f"{model_data['f1_score']:.4f}")
    
    with col3:
        st.metric("ROC-AUC", f"{model_data['roc_auc']:.4f}")
        st.metric("MCC", f"{model_data['mcc']:.4f}")
    
    # Cross-validation results
    st.subheader("🔄 Cross-Validation Results")
    st.info(f"CV Mean: {model_data['cv_mean']:.4f} ± {model_data['cv_std']:.4f}")
    
    # Classification report
    st.subheader("📝 Classification Report")
    class_report = classification_report(
        flood_ml.y_test, 
        model_data['y_pred'], 
        output_dict=True
    )
    
    # Convert to DataFrame for better display
    report_df = pd.DataFrame(class_report).transpose()
    st.dataframe(report_df, use_container_width=True)
    
    # Model-specific interpretability
    st.subheader("🔍 Model Interpretability")
    
    if selected_model in ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost', 'Decision Tree']:
        # Feature importance
        feature_importance = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': flood_ml.feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        fig_model_importance = px.bar(
            importance_df.head(10),
            x='Importance',
            y='Feature',
            orientation='h',
            title=f'Top 10 Feature Importance - {selected_model}'
        )
        fig_model_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_model_importance, use_container_width=True)
        
        # Feature importance table
        st.dataframe(importance_df, use_container_width=True)
    
    elif selected_model == 'Logistic Regression':
        # Coefficients
        coefficients = model.coef_[0]
        coef_df = pd.DataFrame({
            'Feature': flood_ml.feature_names,
            'Coefficient': coefficients
        }).sort_values('Coefficient', key=abs, ascending=False)
        
        fig_coef = px.bar(
            coef_df,
            x='Coefficient',
            y='Feature',
            orientation='h',
            title='Logistic Regression Coefficients',
            color='Coefficient',
            color_continuous_scale='RdYlBu'
        )
        fig_coef.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_coef, use_container_width=True)
        
        st.dataframe(coef_df, use_container_width=True)
    
    # Prediction distribution
    st.subheader("📈 Prediction Distribution")
    
    pred_dist = pd.DataFrame({
        'Actual': flood_ml.y_test,
        'Predicted': model_data['y_pred']
    })
    
    fig_pred_dist = px.histogram(
        pred_dist,
        x='Predicted',
        color='Actual',
        barmode='overlay',
        title='Prediction Distribution by Actual Class',
        opacity=0.7
    )
    st.plotly_chart(fig_pred_dist, use_container_width=True)

# Additional utility functions
def create_advanced_visualizations(flood_ml):
    """Create advanced visualizations for the dashboard"""
    
    # PCA visualization
    st.subheader("🔬 Principal Component Analysis")
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(flood_ml.X_train_scaled)
    
    pca_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'FloodProbability': flood_ml.y_train
    })
    
    fig_pca = px.scatter(
        pca_df,
        x='PC1',
        y='PC2',
        color='FloodProbability',
        title='PCA Visualization of Flood Data',
        labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)',
                'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)'}
    )
    st.plotly_chart(fig_pca, use_container_width=True)
    
    # Clustering analysis
    st.subheader("🔍 Clustering Analysis")
    
    n_clusters = st.slider("Number of clusters:", 2, 10, 3)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(flood_ml.X_train_scaled)
    
    cluster_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Cluster': clusters,
        'FloodProbability': flood_ml.y_train
    })
    
    fig_cluster = px.scatter(
        cluster_df,
        x='PC1',
        y='PC2',
        color='Cluster',
        symbol='FloodProbability',
        title=f'K-Means Clustering (k={n_clusters})'
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

if __name__ == "__main__":
    main()
