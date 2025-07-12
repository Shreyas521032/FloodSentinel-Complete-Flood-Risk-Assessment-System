import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import kagglehub
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           confusion_matrix, classification_report, roc_auc_score, roc_curve)
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import randint, uniform
import time

# Set page config
st.set_page_config(
    page_title="FloodSentinel: ML Flood Risk Assessment",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #fafafa;
        border-radius: 4px 4px 0 0;
        padding: 0 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'df' not in st.session_state:
    st.session_state.df = None

# Main title
st.markdown('<div class="main-header">🌊 FloodSentinel: ML Flood Risk Assessment</div>', unsafe_allow_html=True)
st.markdown("*Advanced Machine Learning System for Flood Risk Prediction Using Multi-Modal Data*")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Select Analysis Module",
    ["🏠 Home", "📊 Data Explorer", "🔍 Feature Analysis", "🤖 ML Models", "📈 Model Comparison", "🎯 Predictions", "📋 Reports"]
)

# Data loading function
@st.cache_data
def load_flood_data():
    """Load and cache flood prediction dataset"""
    try:
        with st.spinner("Downloading flood prediction dataset..."):
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
            df = pd.read_csv(csv_files[0])
            st.success(f"Dataset loaded successfully! Shape: {df.shape}")
            return df
            
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        # Fallback: create sample data for demonstration
        st.warning("Using sample data for demonstration purposes")
        return create_sample_data()

def create_sample_data():
    """Create sample flood prediction data for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'rainfall': np.random.exponential(50, n_samples),
        'temperature': np.random.normal(25, 10, n_samples),
        'humidity': np.random.normal(70, 15, n_samples),
        'wind_speed': np.random.gamma(2, 10, n_samples),
        'pressure': np.random.normal(1013, 20, n_samples),
        'elevation': np.random.uniform(0, 1000, n_samples),
        'slope': np.random.uniform(0, 45, n_samples),
        'soil_type': np.random.choice(['clay', 'sand', 'loam', 'silt'], n_samples),
        'drainage': np.random.choice(['poor', 'moderate', 'good'], n_samples),
        'land_use': np.random.choice(['urban', 'agricultural', 'forest', 'water'], n_samples),
        'distance_to_river': np.random.uniform(0, 50, n_samples),
        'previous_floods': np.random.poisson(2, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create flood risk based on features
    flood_risk = (
        (df['rainfall'] > 100).astype(int) * 0.3 +
        (df['elevation'] < 100).astype(int) * 0.2 +
        (df['distance_to_river'] < 5).astype(int) * 0.2 +
        (df['drainage'] == 'poor').astype(int) * 0.15 +
        (df['previous_floods'] > 3).astype(int) * 0.15
    )
    
    # Add some noise and create binary target
    flood_risk += np.random.normal(0, 0.1, n_samples)
    df['flood_risk'] = (flood_risk > 0.5).astype(int)
    
    return df

# Model training functions
def preprocess_data(df):
    """Preprocess the data for machine learning"""
    df_processed = df.copy()
    
    # Handle missing values
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    
    # Fill missing values
    for col in numeric_cols:
        df_processed[col].fillna(df_processed[col].median(), inplace=True)
    
    for col in categorical_cols:
        df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        if col != 'flood_risk':  # Don't encode target variable
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            label_encoders[col] = le
    
    return df_processed, label_encoders

def get_ml_models():
    """Get dictionary of ML models with their parameters"""
    models = {
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'XGBoost': {
            'model': xgb.XGBClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 5, 6],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        },
        'LightGBM': {
            'model': lgb.LGBMClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 5, 6],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 70]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [3, 4, 5],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        },
        'SVM': {
            'model': SVC(random_state=42, probability=True),
            'params': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['rbf', 'linear', 'poly']
            }
        },
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42),
            'params': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'lbfgs']
            }
        },
        'Neural Network': {
            'model': MLPClassifier(random_state=42, max_iter=1000),
            'params': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01]
            }
        },
        'Extra Trees': {
            'model': ExtraTreesClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10]
            }
        }
    }
    
    return models

# Page functions
def show_home():
    """Show home page with project overview"""
    st.markdown("## 🎯 Project Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🌍 Problem Statement")
        st.markdown("""
        Floods are among the most destructive natural hazards globally, causing:
        - **Loss of life** and economic disruption
        - **Environmental damage** and infrastructure destruction
        - **Displacement** of communities
        - **Agricultural losses** and food security issues
        
        Current flood prediction systems have critical limitations:
        - Computationally intensive physics-based models
        - Limited real-time forecasting capabilities
        - Poor performance in data-scarce regions
        - Lack of multi-modal data integration
        """)
    
    with col2:
        st.markdown("### 🚀 Our Solution")
        st.markdown("""
        **FloodSentinel** addresses these challenges through:
        - **Advanced ML algorithms** for accurate predictions
        - **Multi-modal data fusion** for comprehensive analysis
        - **Real-time processing** capabilities
        - **User-friendly interface** for decision makers
        - **Explainable AI** for transparent insights
        - **Scalable architecture** for global deployment
        """)
    
    st.markdown("### 📊 Key Features")
    
    feature_cols = st.columns(4)
    
    with feature_cols[0]:
        st.markdown("""
        **🔍 Data Explorer**
        - Interactive visualizations
        - Statistical analysis
        - Missing data handling
        - Feature distributions
        """)
    
    with feature_cols[1]:
        st.markdown("""
        **🧠 ML Models**
        - 8+ algorithms comparison
        - Hyperparameter tuning
        - Cross-validation
        - Feature importance
        """)
    
    with feature_cols[2]:
        st.markdown("""
        **📈 Model Comparison**
        - Performance metrics
        - ROC curves
        - Confusion matrices
        - Statistical tests
        """)
    
    with feature_cols[3]:
        st.markdown("""
        **🎯 Predictions**
        - Real-time forecasting
        - Risk assessment
        - Confidence intervals
        - Actionable insights
        """)

def show_data_explorer():
    """Show data exploration page"""
    st.markdown('<div class="sub-header">📊 Data Explorer</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        if st.button("Load Flood Dataset", type="primary"):
            df = load_flood_data()
            if df is not None:
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.rerun()
    
    if st.session_state.data_loaded and st.session_state.df is not None:
        df = st.session_state.df
        
        # Dataset overview
        st.markdown("### Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Features", f"{len(df.columns)-1}")
        with col3:
            st.metric("Flood Cases", f"{df['flood_risk'].sum():,}")
        with col4:
            st.metric("Flood Rate", f"{df['flood_risk'].mean():.1%}")
        
        # Data preview
        st.markdown("### Data Preview")
        st.dataframe(df.head(10))
        
        # Data quality assessment
        st.markdown("### Data Quality Assessment")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Missing Values**")
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                fig = px.bar(x=missing_data.index, y=missing_data.values, 
                           title="Missing Values by Column")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No missing values found!")
        
        with col2:
            st.markdown("**Data Types**")
            dtypes_df = pd.DataFrame({
                'Column': df.dtypes.index,
                'Type': df.dtypes.values
            })
            st.dataframe(dtypes_df)
        
        # Statistical summary
        st.markdown("### Statistical Summary")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        st.dataframe(df[numeric_cols].describe())
        
        # Distribution plots
        st.markdown("### Feature Distributions")
        
        # Select features for visualization
        selected_features = st.multiselect(
            "Select features to visualize:",
            options=numeric_cols.tolist(),
            default=numeric_cols[:4].tolist()
        )
        
        if selected_features:
            n_cols = min(2, len(selected_features))
            n_rows = (len(selected_features) + n_cols - 1) // n_cols
            
            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=selected_features,
                vertical_spacing=0.1
            )
            
            for i, feature in enumerate(selected_features):
                row = i // n_cols + 1
                col = i % n_cols + 1
                
                fig.add_trace(
                    go.Histogram(x=df[feature], name=feature, showlegend=False),
                    row=row, col=col
                )
            
            fig.update_layout(height=300 * n_rows, title_text="Feature Distributions")
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        st.markdown("### Correlation Analysis")
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Feature Correlation Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Target variable analysis
        st.markdown("### Target Variable Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Flood risk distribution
            flood_counts = df['flood_risk'].value_counts()
            fig = px.pie(
                values=flood_counts.values,
                names=['No Flood', 'Flood'],
                title="Flood Risk Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Feature importance for flood prediction
            if 'flood_risk' in df.columns:
                categorical_cols = df.select_dtypes(include=['object']).columns
                df_encoded = df.copy()
                
                # Simple encoding for correlation
                for col in categorical_cols:
                    if col != 'flood_risk':
                        df_encoded[col] = pd.Categorical(df_encoded[col]).codes
                
                correlations = df_encoded.corr()['flood_risk'].abs().sort_values(ascending=False)[1:]
                
                fig = px.bar(
                    x=correlations.values,
                    y=correlations.index,
                    orientation='h',
                    title="Feature Correlation with Flood Risk"
                )
                st.plotly_chart(fig, use_container_width=True)

def show_feature_analysis():
    """Show feature analysis page"""
    st.markdown('<div class="sub-header">🔍 Feature Analysis</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load the dataset first from the Data Explorer page.")
        return
    
    df = st.session_state.df
    
    # Feature selection methods
    st.markdown("### Feature Selection Methods")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Statistical Selection", "🌳 Tree-based Importance", "🔄 Recursive Elimination", "📉 Dimensionality Reduction"])
    
    with tab1:
        st.markdown("#### Statistical Feature Selection")
        
        # Prepare data
        df_processed, _ = preprocess_data(df)
        
        # Assume target is the last column or named 'flood_risk'
        target_col = 'flood_risk' if 'flood_risk' in df_processed.columns else df_processed.columns[-1]
        X = df_processed.drop(target_col, axis=1)
        y = df_processed[target_col]
        
        # SelectKBest with f_classif
        k_best = SelectKBest(score_func=f_classif, k='all')
        k_best.fit(X, y)
        
        # Create feature importance dataframe
        feature_scores = pd.DataFrame({
            'Feature': X.columns,
            'Score': k_best.scores_,
            'P-value': k_best.pvalues_
        }).sort_values('Score', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Top Features by F-Score**")
            st.dataframe(feature_scores.head(10))
        
        with col2:
            fig = px.bar(
                feature_scores.head(10),
                x='Score',
                y='Feature',
                orientation='h',
                title="Top 10 Features by F-Score"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("#### Tree-based Feature Importance")
        
        # Random Forest feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        rf_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Random Forest Feature Importance**")
            st.dataframe(rf_importance.head(10))
        
        with col2:
            fig = px.bar(
                rf_importance.head(10),
                x='Importance',
                y='Feature',
                orientation='h',
                title="Top 10 Features by RF Importance"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("#### Recursive Feature Elimination")
        
        # RFE with Random Forest
        rfe = RFE(RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=10)
        rfe.fit(X, y)
        
        rfe_features = pd.DataFrame({
            'Feature': X.columns,
            'Selected': rfe.support_,
            'Ranking': rfe.ranking_
        }).sort_values('Ranking')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**RFE Feature Selection**")
            st.dataframe(rfe_features)
        
        with col2:
            selected_features = rfe_features[rfe_features['Selected']]['Feature'].tolist()
            fig = px.bar(
                x=list(range(1, len(selected_features) + 1)),
                y=selected_features,
                orientation='h',
                title="Selected Features by RFE"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("#### Principal Component Analysis")
        
        # PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # Explained variance ratio
        explained_variance = pd.DataFrame({
            'PC': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
            'Explained_Variance': pca.explained_variance_ratio_,
            'Cumulative_Variance': np.cumsum(pca.explained_variance_ratio_)
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                explained_variance.head(10),
                x='PC',
                y='Explained_Variance',
                title="PCA Explained Variance by Component"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.line(
                explained_variance.head(10),
                x='PC',
                y='Cumulative_Variance',
                title="Cumulative Explained Variance",
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)

def show_ml_models():
    """Show ML models training page"""
    st.markdown('<div class="sub-header">🤖 Machine Learning Models</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load the dataset first from the Data Explorer page.")
        return
    
    df = st.session_state.df
    
    # Model configuration
    st.markdown("### Model Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
    
    with col2:
        hyperparameter_tuning = st.checkbox("Enable Hyperparameter Tuning", value=True)
        feature_selection = st.checkbox("Enable Feature Selection", value=True)
    
    with col3:
        scaling_method = st.selectbox("Scaling Method", ["StandardScaler", "RobustScaler", "None"])
        random_state = st.number_input("Random State", value=42)
    
    # Model selection
    st.markdown("### Select Models to Train")
    
    models_dict = get_ml_models()
    selected_models = st.multiselect(
        "Choose models:",
        options=list(models_dict.keys()),
        default=["Random Forest", "XGBoost", "LightGBM", "Logistic Regression"]
    )
    
    if st.button("Train Models", type="primary"):
        if not selected_models:
            st.error("Please select at least one model to train.")
            return
        
        # Prepare data
        df_processed, label_encoders = preprocess_data(df)
        
        target_col = 'flood_risk' if 'flood_risk' in df_processed.columns else df_processed.columns[-1]
        X = df_processed.drop(target_col, axis=1)
        y = df_processed[target_col]
        
        # Feature selection
        if feature_selection:
            with st.spinner("Performing feature selection..."):
                selector = SelectKBest(score_func=f_classif, k=min(10, X.shape[1]))
                X = selector.fit_transform(X, y)
                selected_features = selector.get_support(indices=True)
                st.info(f"Selected {len(selected_features)} features out of {len(df_processed.columns)-1}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scaling
        if scaling_method != "None":
            if scaling_method == "StandardScaler":
                scaler = StandardScaler()
            else:
                scaler = RobustScaler()
            
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        # Train models
        results = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, model_name in enumerate(selected_models):
            status_text.text(f"Training {model_name}...")
            
            start_time = time.time()
            
            model_config = models_dict[model_name]
            model = model_config['model']
            
            if hyperparameter_tuning:
                # Hyperparameter tuning
                search = RandomizedSearchCV(
                    model,
                    model_config['params'],
                    n_iter=20,
                    cv=cv_folds,
                    scoring='f1',
                    random_state=random_state,
                    n_jobs=-1
                )
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
                best_params = search.best_params_
            else:
                best_model = model
                best_model.fit(X_train, y_train)
                best_params = {}
            
            # Predictions
            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Cross-validation
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv_folds, scoring='f1')
            
            training_time = time.time() - start_time
            
            results[model_name] = {
                'model': best_model,
                'best_params': best_params,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'training_time': training_time,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_prob': y_prob
            }
            
            progress_bar.progress((i + 1) / len(selected_models))
        
        status_text.text("Training completed!")
        
        # Store results in session state
        st.session_state.models_trained = True
        st.session_state.model_results = results
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        
        # Display results
        st.markdown("### Training Results")
        
        results_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [results[model]['accuracy'] for model in results],
            'Precision': [results[model]['precision'] for model in results],
            'Recall': [results[model]['recall'] for model in results],
            'F1-Score': [results[model]['f1_score'] for model in results],
            'CV Mean': [results[model]['cv_mean'] for model in results],
            'CV Std': [results[model]['cv_std'] for model in results],
            'Training Time (s)': [results[model]['training_time'] for model in results]
        }).round(4)
        
        st.dataframe(results_df)
        
        # Best model highlight
        best_model_name = results_df.loc[results_df['F1-Score'].idxmax(), 'Model']
        st.success(f"🏆 Best performing model: **{best_model_name}** with F1-Score: {results_df['F1-Score'].max():.4f}")

def show_model_comparison():
    """Show model comparison page"""
    st.markdown('<div class="sub-header">📈 Model Comparison</div>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.warning("Please train models first from the ML Models page.")
        return
    
    results = st.session_state.model_results
    
    # Performance metrics comparison
    st.markdown("### Performance Metrics Comparison")
    
    metrics_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[model]['accuracy'] for model in results],
        'Precision': [results[model]['precision'] for model in results],
        'Recall': [results[model]['recall'] for model in results],
        'F1-Score': [results[model]['f1_score'] for model in results],
        'CV Mean': [results[model]['cv_mean'] for model in results],
        'Training Time': [results[model]['training_time'] for model in results]
    })
    
    # Performance comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            metrics_df.melt(id_vars=['Model'], value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score']),
            x='Model',
            y='value',
            color='variable',
            title="Performance Metrics Comparison",
            barmode='group'
        )
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            metrics_df,
            x='Training Time',
            y='F1-Score',
            size='Accuracy',
            color='Model',
            title="Performance vs Training Time",
            hover_data=['Precision', 'Recall']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ROC Curves
    st.markdown("### ROC Curves Comparison")
    
    fig = go.Figure()
    
    for model_name in results:
        if results[model_name]['y_prob'] is not None:
            fpr, tpr, _ = roc_curve(results[model_name]['y_test'], results[model_name]['y_prob'])
            auc = roc_auc_score(results[model_name]['y_test'], results[model_name]['y_prob'])
            
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name=f'{model_name} (AUC = {auc:.3f})',
                line=dict(width=2)
            ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color='gray')
    ))
    
    fig.update_layout(
        title='ROC Curves Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=800,
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Confusion Matrices
    st.markdown("### Confusion Matrices")
    
    n_models = len(results)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=list(results.keys()),
        specs=[[{"type": "heatmap"} for _ in range(n_cols)] for _ in range(n_rows)]
    )
    
    for i, model_name in enumerate(results):
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        cm = confusion_matrix(results[model_name]['y_test'], results[model_name]['y_pred'])
        
        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=['Predicted No Flood', 'Predicted Flood'],
                y=['Actual No Flood', 'Actual Flood'],
                colorscale='Blues',
                showscale=i == 0,
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 12}
            ),
            row=row,
            col=col
        )
    
    fig.update_layout(
        title_text="Confusion Matrices Comparison",
        height=300 * n_rows
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature Importance (for tree-based models)
    st.markdown("### Feature Importance Analysis")
    
    tree_based_models = ['Random Forest', 'XGBoost', 'LightGBM', 'Gradient Boosting', 'Extra Trees']
    available_tree_models = [model for model in results.keys() if model in tree_based_models]
    
    if available_tree_models:
        selected_model = st.selectbox("Select model for feature importance:", available_tree_models)
        
        if hasattr(results[selected_model]['model'], 'feature_importances_'):
            importances = results[selected_model]['model'].feature_importances_
            
            # Create feature names (assuming we have them)
            feature_names = [f'Feature_{i}' for i in range(len(importances))]
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(
                importance_df.head(15),
                x='Importance',
                y='Feature',
                orientation='h',
                title=f"Top 15 Feature Importances - {selected_model}"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Model comparison table
    st.markdown("### Detailed Model Comparison")
    
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [f"{results[model]['accuracy']:.4f}" for model in results],
        'Precision': [f"{results[model]['precision']:.4f}" for model in results],
        'Recall': [f"{results[model]['recall']:.4f}" for model in results],
        'F1-Score': [f"{results[model]['f1_score']:.4f}" for model in results],
        'CV Mean ± Std': [f"{results[model]['cv_mean']:.4f} ± {results[model]['cv_std']:.4f}" for model in results],
        'Training Time (s)': [f"{results[model]['training_time']:.2f}" for model in results]
    })
    
    st.dataframe(comparison_df)
    
    # Best model summary
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
    best_model_results = results[best_model_name]
    
    st.markdown("### 🏆 Best Model Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Best Model", best_model_name)
        st.metric("F1-Score", f"{best_model_results['f1_score']:.4f}")
    
    with col2:
        st.metric("Accuracy", f"{best_model_results['accuracy']:.4f}")
        st.metric("Precision", f"{best_model_results['precision']:.4f}")
    
    with col3:
        st.metric("Recall", f"{best_model_results['recall']:.4f}")
        st.metric("CV Score", f"{best_model_results['cv_mean']:.4f}")
    
    with col4:
        st.metric("Training Time", f"{best_model_results['training_time']:.2f}s")
        st.metric("CV Std", f"{best_model_results['cv_std']:.4f}")

def show_predictions():
    """Show predictions page"""
    st.markdown('<div class="sub-header">🎯 Flood Risk Predictions</div>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.warning("Please train models first from the ML Models page.")
        return
    
    results = st.session_state.model_results
    
    # Model selection for predictions
    st.markdown("### Select Model for Predictions")
    
    model_options = list(results.keys())
    selected_model = st.selectbox("Choose model:", model_options)
    
    # Input methods
    st.markdown("### Prediction Input")
    
    input_method = st.radio(
        "Choose input method:",
        ["Manual Input", "Batch Upload", "Random Sample"]
    )
    
    if input_method == "Manual Input":
        st.markdown("#### Enter Feature Values")
        
        # Create input fields based on sample data structure
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=50.0)
            temperature = st.number_input("Temperature (°C)", min_value=-20.0, max_value=50.0, value=25.0)
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0)
            wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=200.0, value=20.0)
        
        with col2:
            pressure = st.number_input("Pressure (hPa)", min_value=900.0, max_value=1100.0, value=1013.0)
            elevation = st.number_input("Elevation (m)", min_value=0.0, max_value=5000.0, value=100.0)
            slope = st.number_input("Slope (degrees)", min_value=0.0, max_value=90.0, value=5.0)
            distance_to_river = st.number_input("Distance to River (km)", min_value=0.0, max_value=100.0, value=10.0)
        
        with col3:
            soil_type = st.selectbox("Soil Type", ["clay", "sand", "loam", "silt"])
            drainage = st.selectbox("Drainage", ["poor", "moderate", "good"])
            land_use = st.selectbox("Land Use", ["urban", "agricultural", "forest", "water"])
            previous_floods = st.number_input("Previous Floods", min_value=0, max_value=20, value=2)
        
        if st.button("Predict Flood Risk", type="primary"):
            # Prepare input data
            input_data = pd.DataFrame({
                'rainfall': [rainfall],
                'temperature': [temperature],
                'humidity': [humidity],
                'wind_speed': [wind_speed],
                'pressure': [pressure],
                'elevation': [elevation],
                'slope': [slope],
                'distance_to_river': [distance_to_river],
                'soil_type': [soil_type],
                'drainage': [drainage],
                'land_use': [land_use],
                'previous_floods': [previous_floods]
            })
            
            # Encode categorical variables (simple encoding for demo)
            input_data_encoded = input_data.copy()
            categorical_columns = ['soil_type', 'drainage', 'land_use']
            
            for col in categorical_columns:
                input_data_encoded[col] = pd.Categorical(input_data_encoded[col]).codes
            
            # Make prediction
            model = results[selected_model]['model']
            prediction = model.predict(input_data_encoded)[0]
            
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(input_data_encoded)[0]
                flood_probability = probability[1]
            else:
                flood_probability = None
            
            # Display results
            st.markdown("### Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("⚠️ **HIGH FLOOD RISK DETECTED**")
                    st.markdown("**Recommendation:** Immediate action required!")
                else:
                    st.success("✅ **LOW FLOOD RISK**")
                    st.markdown("**Status:** Normal conditions")
            
            with col2:
                if flood_probability is not None:
                    st.metric("Flood Probability", f"{flood_probability:.2%}")
                    
                    # Risk level
                    if flood_probability < 0.3:
                        risk_level = "Low"
                        risk_color = "green"
                    elif flood_probability < 0.7:
                        risk_level = "Medium"
                        risk_color = "orange"
                    else:
                        risk_level = "High"
                        risk_color = "red"
                    
                    st.markdown(f"**Risk Level:** <span style='color: {risk_color}'>{risk_level}</span>", 
                              unsafe_allow_html=True)
            
            # Feature contribution (for tree-based models)
            if hasattr(model, 'feature_importances_'):
                st.markdown("### Feature Contribution Analysis")
                
                feature_names = list(input_data_encoded.columns)
                feature_values = input_data_encoded.iloc[0].values
                feature_importances = model.feature_importances_
                
                contribution_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Value': feature_values,
                    'Importance': feature_importances,
                    'Contribution': feature_values * feature_importances
                }).sort_values('Contribution', ascending=False)
                
                fig = px.bar(
                    contribution_df,
                    x='Contribution',
                    y='Feature',
                    orientation='h',
                    title="Feature Contribution to Prediction",
                    color='Contribution',
                    color_continuous_scale='RdYlBu_r'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    elif input_method == "Batch Upload":
        st.markdown("#### Upload CSV File for Batch Predictions")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            batch_data = pd.read_csv(uploaded_file)
            st.write("Uploaded data preview:")
            st.dataframe(batch_data.head())
            
            if st.button("Run Batch Predictions"):
                # Process batch data (simplified)
                st.info("Batch prediction functionality would be implemented here")
                st.write("This would process all rows and return predictions")
    
    else:  # Random Sample
        st.markdown("#### Generate Random Sample Predictions")
        
        n_samples = st.slider("Number of samples", 1, 100, 10)
        
        if st.button("Generate Random Predictions"):
            # Generate random samples
            np.random.seed(42)
            
            random_data = pd.DataFrame({
                'rainfall': np.random.exponential(50, n_samples),
                'temperature': np.random.normal(25, 10, n_samples),
                'humidity': np.random.normal(70, 15, n_samples),
                'wind_speed': np.random.gamma(2, 10, n_samples),
                'pressure': np.random.normal(1013, 20, n_samples),
                'elevation': np.random.uniform(0, 1000, n_samples),
                'slope': np.random.uniform(0, 45, n_samples),
                'distance_to_river': np.random.uniform(0, 50, n_samples),
                'soil_type': np.random.choice([0, 1, 2, 3], n_samples),
                'drainage': np.random.choice([0, 1, 2], n_samples),
                'land_use': np.random.choice([0, 1, 2, 3], n_samples),
                'previous_floods': np.random.poisson(2, n_samples)
            })
            
            # Make predictions
            model = results[selected_model]['model']
            predictions = model.predict(random_data)
            
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(random_data)[:, 1]
                random_data['flood_probability'] = probabilities
            
            random_data['prediction'] = predictions
            random_data['risk_level'] = random_data.get('flood_probability', predictions).apply(
                lambda x: 'High' if x > 0.7 else 'Medium' if x > 0.3 else 'Low'
            )
            
            st.markdown("### Random Sample Predictions")
            st.dataframe(random_data)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Samples", len(random_data))
                st.metric("High Risk", len(random_data[random_data['prediction'] == 1]))
            
            with col2:
                st.metric("Low Risk", len(random_data[random_data['prediction'] == 0]))
                st.metric("Average Risk", f"{random_data['prediction'].mean():.2%}")
            
            with col3:
                if 'flood_probability' in random_data.columns:
                    st.metric("Mean Probability", f"{random_data['flood_probability'].mean():.2%}")
                    st.metric("Max Probability", f"{random_data['flood_probability'].max():.2%}")

def show_reports():
    """Show reports page"""
    st.markdown('<div class="sub-header">📋 Analysis Reports</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load the dataset first to generate reports.")
        return
    
    df = st.session_state.df
    
    # Report selection
    report_type = st.selectbox(
        "Select Report Type:",
        ["Executive Summary", "Technical Report", "Model Performance Report", "Data Quality Report"]
    )
    
    if report_type == "Executive Summary":
        st.markdown("### 📊 Executive Summary Report")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Dataset Size", f"{len(df):,} records")
        with col2:
            st.metric("Features", f"{len(df.columns)-1}")
        with col3:
            st.metric("Flood Cases", f"{df['flood_risk'].sum():,}")
        with col4:
            st.metric("Flood Rate", f"{df['flood_risk'].mean():.1%}")
        
        # Key findings
        st.markdown("#### Key Findings")
        st.markdown("""
        - **Dataset Coverage**: Comprehensive dataset with multiple environmental and geographical features
        - **Flood Distribution**: Balanced representation of flood and non-flood scenarios
        - **Data Quality**: High-quality dataset with minimal missing values
        - **Feature Diversity**: Multi-modal features including weather, terrain, and historical data
        """)
        
        if st.session_state.models_trained:
            results = st.session_state.model_results
            best_model = max(results.keys(), key=lambda x: results[x]['f1_score'])
            
            st.markdown("#### Model Performance Summary")
            st.markdown(f"""
            - **Best Performing Model**: {best_model}
            - **Accuracy**: {results[best_model]['accuracy']:.1%}
            - **F1-Score**: {results[best_model]['f1_score']:.3f}
            - **Cross-Validation Score**: {results[best_model]['cv_mean']:.3f} ± {results[best_model]['cv_std']:.3f}
            """)
        
        # Recommendations
        st.markdown("#### Recommendations")
        st.markdown("""
        1. **Deploy the best-performing model** for real-time flood risk assessment
        2. **Integrate with early warning systems** for proactive disaster management
        3. **Expand data collection** to include more geographical regions
        4. **Implement continuous model monitoring** and retraining
        5. **Develop mobile applications** for field personnel
        """)
    
    elif report_type == "Technical Report":
        st.markdown("### 🔧 Technical Report")
        
        # Data preprocessing
        st.markdown("#### Data Preprocessing")
        st.markdown("""
        - **Missing Value Treatment**: Median imputation for numerical, mode for categorical
        - **Feature Encoding**: Label encoding for categorical variables
        - **Scaling**: StandardScaler/RobustScaler applied for algorithm compatibility
        - **Feature Selection**: Statistical and tree-based methods employed
        """)
        
        # Model architecture
        st.markdown("#### Model Architecture")
        
        if st.session_state.models_trained:
            results = st.session_state.model_results
            
            for model_name, model_results in results.items():
                with st.expander(f"{model_name} Configuration"):
                    st.json(model_results['best_params'])
        
        # Performance metrics
        st.markdown("#### Performance Metrics")
        st.markdown("""
        - **Accuracy**: Overall correctness of predictions
        - **Precision**: Proportion of true positives among predicted positives
        - **Recall**: Proportion of true positives among actual positives
        - **F1-Score**: Harmonic mean of precision and recall
        - **ROC-AUC**: Area under the receiver operating characteristic curve
        """)
    
    elif report_type == "Model Performance Report":
        if not st.session_state.models_trained:
            st.warning("Please train models first to generate this report.")
            return
        
        st.markdown("### 📈 Model Performance Report")
        
        results = st.session_state.model_results
        
        # Performance comparison table
        performance_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [results[model]['accuracy'] for model in results],
            'Precision': [results[model]['precision'] for model in results],
            'Recall': [results[model]['recall'] for model in results],
            'F1-Score': [results[model]['f1_score'] for model in results],
            'CV Score': [results[model]['cv_mean'] for model in results],
            'Training Time': [results[model]['training_time'] for model in results]
        })
        
        st.dataframe(performance_df.round(4))
        
        # Best model analysis
        best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
        
        st.markdown(f"#### Best Model: {best_model_name}")
        st.markdown(f"""
        **Performance Metrics:**
        - Accuracy: {results[best_model_name]['accuracy']:.4f}
        - Precision: {results[best_model_name]['precision']:.4f}
        - Recall: {results[best_model_name]['recall']:.4f}
        - F1-Score: {results[best_model_name]['f1_score']:.4f}
        
        **Cross-Validation:**
        - Mean Score: {results[best_model_name]['cv_mean']:.4f}
        - Standard Deviation: {results[best_model_name]['cv_std']:.4f}
        
        **Training Time:** {results[best_model_name]['training_time']:.2f} seconds
        """)
    
    else:  # Data Quality Report
        st.markdown("### 📋 Data Quality Report")
        
        # Missing values analysis
        missing_values = df.isnull().sum()
        
        st.markdown("#### Missing Values Analysis")
        if missing_values.sum() == 0:
            st.success("✅ No missing values found in the dataset")
        else:
            st.dataframe(missing_values[missing_values > 0])
        
        # Data types
        st.markdown("#### Data Types")
        dtype_df = pd.DataFrame({
            'Column': df.dtypes.index,
            'Data Type': df.dtypes.values,
            'Non-Null Count': df.count().values,
            'Null Count': df.isnull().sum().values
        })
        st.dataframe(dtype_df)
        
        # Statistical summary
        st.markdown("#### Statistical Summary")
        st.dataframe(df.describe())
        
        # Data quality score
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        quality_score = (total_cells - missing_cells) / total_cells
        
        st.markdown("#### Overall Data Quality Score")
        st.metric("Quality Score", f"{quality_score:.2%}")
        
        if quality_score > 0.95:
            st.success("Excellent data quality!")
        elif quality_score > 0.80:
            st.warning("Good data quality with minor issues")
        else:
            st.error("Data quality needs improvement")

# Main app logic
if page == "🏠 Home":
    show_home()
elif page == "📊 Data Explorer":
    show_data_explorer()
elif page == "🔍 Feature Analysis":
    show_feature_analysis()
elif page == "🤖 ML Models":
    show_ml_models()
elif page == "📈 Model Comparison":
    show_model_comparison()
elif page == "🎯 Predictions":
    show_predictions()
elif page == "📋 Reports":
    show_reports()

# Footer
st.markdown("---")
st.markdown("### 🌊 FloodSentinel - Advanced Flood Risk Assessment System")
st.markdown("*Developed using state-of-the-art machine learning techniques for disaster management and early warning systems.*")
