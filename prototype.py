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

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# Kaggle Hub for dataset download
import kagglehub
import os

# Set page config
st.set_page_config(
    page_title="FloodSentinel - Flood Risk Assessment",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1e40af;
    text-align: center;
    margin-bottom: 2rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    margin: 0.5rem 0;
}
.stAlert {
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🌊 FloodSentinel</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Flood Risk Assessment Using Machine Learning</p>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", [
    "🏠 Home",
    "📊 Data Overview", 
    "🔍 Exploratory Data Analysis",
    "🤖 Model Training",
    "📈 Model Comparison",
    "🎯 Prediction Interface",
    "📋 Feature Importance"
])

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

@st.cache_data
def load_dataset():
    """Load the flood prediction dataset from Kaggle"""
    try:
        with st.spinner("Downloading dataset from Kaggle..."):
            path = kagglehub.dataset_download("naiyakhalid/flood-prediction-dataset")
            
            # Find CSV file in the downloaded path
            csv_files = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.csv'):
                        csv_files.append(os.path.join(root, file))
            
            if not csv_files:
                st.error("No CSV files found in the dataset!")
                return None
            
            # Load the first CSV file found
            df = pd.read_csv(csv_files[0])
            st.success(f"Dataset loaded successfully! Shape: {df.shape}")
            return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

@st.cache_data
def preprocess_data(df):
    """Preprocess the dataset"""
    # Handle missing values
    df = df.fillna(df.mean())
    
    # Define feature columns (all except target)
    feature_columns = [
        'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement', 
        'Deforestation', 'Urbanization', 'ClimateChange', 'DamsQuality', 
        'Siltation', 'AgriculturalPractices', 'Encroachments', 
        'IneffectiveDisasterPreparedness', 'DrainageSystems', 
        'CoastalVulnerability', 'Landslides', 'Watersheds', 
        'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss', 
        'InadequatePlanning', 'PoliticalFactors'
    ]
    
    target_column = 'FloodProbability'
    
    # Filter columns that exist in the dataset
    available_features = [col for col in feature_columns if col in df.columns]
    
    if target_column not in df.columns:
        st.error(f"Target column '{target_column}' not found in dataset!")
        return None, None, None
    
    X = df[available_features]
    y = df[target_column]
    
    return X, y, available_features

def train_multiple_models(X_train, X_test, y_train, y_test):
    """Train multiple ML models and return their performance"""
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
        'CatBoost': CatBoostRegressor(n_estimators=100, random_state=42, verbose=False),
        'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0),
        'KNN': KNeighborsRegressor(n_neighbors=5),
        'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
    }
    
    results = {}
    trained_models = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (name, model) in enumerate(models.items()):
        status_text.text(f'Training {name}...')
        
        try:
            # Train model
            if name in ['SVR', 'Neural Network']:
                # Scale features for SVR and Neural Network
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                trained_models[name] = (model, scaler)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                trained_models[name] = (model, None)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            results[name] = {
                'MSE': mse,
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'Model': model
            }
            
        except Exception as e:
            st.warning(f"Error training {name}: {str(e)}")
            results[name] = {
                'MSE': np.inf,
                'MAE': np.inf,
                'RMSE': np.inf,
                'R2': -np.inf,
                'Model': None
            }
        
        progress_bar.progress((i + 1) / len(models))
    
    status_text.text('Model training completed!')
    return results, trained_models

# Page routing
if page == "🏠 Home":
    st.markdown("""
    ## Welcome to FloodSentinel
    
    FloodSentinel is a comprehensive flood risk assessment system that leverages machine learning to predict flood probability based on various environmental and socio-economic factors.
    
    ### 🎯 Key Features:
    - **Multi-Model Approach**: Compare 14 different machine learning algorithms
    - **Interactive Visualizations**: Explore data patterns with interactive charts
    - **Real-time Predictions**: Get instant flood risk assessments
    - **Feature Importance Analysis**: Understand which factors contribute most to flood risk
    - **Model Performance Comparison**: Compare different algorithms to find the best performer
    
    ### 📊 Supported Factors:
    - Environmental: Monsoon Intensity, Topography, Climate Change, Deforestation
    - Infrastructure: River Management, Dams Quality, Drainage Systems
    - Human Impact: Urbanization, Population Score, Agricultural Practices
    - Management: Disaster Preparedness, Planning Quality, Political Factors
    
    ### 🚀 Get Started:
    1. Navigate to **Data Overview** to load and explore the dataset
    2. Use **Exploratory Data Analysis** to understand data patterns
    3. Train models in **Model Training** section
    4. Compare performance in **Model Comparison**
    5. Make predictions using **Prediction Interface**
    """)
    
    # Load dataset button
    if st.button("🔄 Load Dataset", type="primary"):
        df = load_dataset()
        if df is not None:
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.rerun()

elif page == "📊 Data Overview":
    st.header("📊 Data Overview")
    
    if not st.session_state.data_loaded:
        st.warning("Please load the dataset first from the Home page.")
        if st.button("🔄 Load Dataset"):
            df = load_dataset()
            if df is not None:
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.rerun()
    else:
        df = st.session_state.df
        
        # Basic info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Features", len(df.columns) - 1)
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        # Display dataset
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10))
        
        # Statistical summary
        st.subheader("Statistical Summary")
        st.dataframe(df.describe())
        
        # Data types
        st.subheader("Data Types")
        st.dataframe(pd.DataFrame({'Column': df.columns, 'Type': df.dtypes, 'Non-Null Count': df.count()}))

elif page == "🔍 Exploratory Data Analysis":
    st.header("🔍 Exploratory Data Analysis")
    
    if not st.session_state.data_loaded:
        st.warning("Please load the dataset first from the Home page.")
    else:
        df = st.session_state.df
        
        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix, 
                       text_auto=True, 
                       aspect="auto",
                       title="Feature Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution plots
        st.subheader("Feature Distributions")
        
        # Select features for distribution
        selected_features = st.multiselect(
            "Select features to visualize:",
            options=numeric_cols.tolist(),
            default=numeric_cols[:4].tolist()
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
                    go.Histogram(x=df[feature], name=feature, showlegend=False),
                    row=row, col=col
                )
            
            fig.update_layout(height=600, title_text="Feature Distributions")
            st.plotly_chart(fig, use_container_width=True)
        
        # Target variable analysis
        if 'FloodProbability' in df.columns:
            st.subheader("Target Variable Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(df, x='FloodProbability', nbins=30,
                                 title="Flood Probability Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(df, y='FloodProbability',
                           title="Flood Probability Box Plot")
                st.plotly_chart(fig, use_container_width=True)
        
        # Pairwise relationships
        st.subheader("Pairwise Relationships with Target")
        if 'FloodProbability' in df.columns:
            feature_for_scatter = st.selectbox(
                "Select feature for scatter plot with FloodProbability:",
                options=[col for col in numeric_cols if col != 'FloodProbability']
            )
            
            if feature_for_scatter:
                fig = px.scatter(df, x=feature_for_scatter, y='FloodProbability',
                               title=f'{feature_for_scatter} vs Flood Probability',
                               trendline="ols")
                st.plotly_chart(fig, use_container_width=True)

elif page == "🤖 Model Training":
    st.header("🤖 Model Training")
    
    if not st.session_state.data_loaded:
        st.warning("Please load the dataset first from the Home page.")
    else:
        df = st.session_state.df
        
        # Preprocess data
        X, y, feature_names = preprocess_data(df)
        
        if X is not None and y is not None:
            st.success(f"Data preprocessed successfully! Features: {len(feature_names)}")
            
            # Train/test split parameters
            col1, col2 = st.columns(2)
            with col1:
                test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
            with col2:
                random_state = st.number_input("Random State", 0, 100, 42)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            st.info(f"Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")
            
            # Train models button
            if st.button("🚀 Train All Models", type="primary"):
                results, trained_models = train_multiple_models(X_train, X_test, y_train, y_test)
                
                # Store results in session state
                st.session_state.results = results
                st.session_state.trained_models = trained_models
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.feature_names = feature_names
                st.session_state.models_trained = True
                
                st.success("All models trained successfully!")
                
                # Display quick results
                results_df = pd.DataFrame(results).T
                results_df = results_df.round(4)
                st.dataframe(results_df[['RMSE', 'MAE', 'R2']].sort_values('R2', ascending=False))

elif page == "📈 Model Comparison":
    st.header("📈 Model Comparison")
    
    if not st.session_state.models_trained:
        st.warning("Please train models first in the Model Training page.")
    else:
        results = st.session_state.results
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(results).T
        comparison_df = comparison_df.round(4)
        
        # Sort by R2 score
        comparison_df = comparison_df.sort_values('R2', ascending=False)
        
        # Display results table
        st.subheader("Model Performance Comparison")
        st.dataframe(comparison_df[['RMSE', 'MAE', 'R2']])
        
        # Best model highlight
        best_model = comparison_df.index[0]
        st.success(f"🏆 Best Model: {best_model} (R² = {comparison_df.loc[best_model, 'R2']:.4f})")
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(comparison_df.reset_index(), 
                        x='index', y='R2',
                        title="R² Score Comparison",
                        labels={'index': 'Model', 'R2': 'R² Score'})
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(comparison_df.reset_index(), 
                        x='index', y='RMSE',
                        title="RMSE Comparison",
                        labels={'index': 'Model', 'RMSE': 'RMSE'})
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Model performance metrics
        st.subheader("Detailed Performance Metrics")
        
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("Best R² Score", f"{comparison_df['R2'].max():.4f}")
        with metric_cols[1]:
            st.metric("Best RMSE", f"{comparison_df['RMSE'].min():.4f}")
        with metric_cols[2]:
            st.metric("Best MAE", f"{comparison_df['MAE'].min():.4f}")
        with metric_cols[3]:
            st.metric("Models Trained", len(comparison_df))

elif page == "🎯 Prediction Interface":
    st.header("🎯 Flood Risk Prediction")
    
    if not st.session_state.models_trained:
        st.warning("Please train models first in the Model Training page.")
    else:
        # Model selection
        results = st.session_state.results
        trained_models = st.session_state.trained_models
        feature_names = st.session_state.feature_names
        
        # Sort models by R2 score
        model_performance = {name: results[name]['R2'] for name in results.keys()}
        sorted_models = sorted(model_performance.items(), key=lambda x: x[1], reverse=True)
        
        selected_model = st.selectbox(
            "Select Model for Prediction:",
            options=[name for name, _ in sorted_models],
            index=0
        )
        
        st.info(f"Using {selected_model} (R² = {results[selected_model]['R2']:.4f})")
        
        # Input form
        st.subheader("Enter Feature Values")
        
        # Create input fields for each feature
        input_data = {}
        
        # Organize inputs in columns
        num_cols = 3
        cols = st.columns(num_cols)
        
        for i, feature in enumerate(feature_names):
            with cols[i % num_cols]:
                input_data[feature] = st.number_input(
                    f"{feature}:",
                    value=0.0,
                    step=0.1,
                    key=f"input_{feature}"
                )
        
        # Prediction button
        if st.button("🔮 Predict Flood Risk", type="primary"):
            # Prepare input data
            input_df = pd.DataFrame([input_data])
            
            # Get model and scaler
            model_info = trained_models[selected_model]
            model = model_info[0]
            scaler = model_info[1]
            
            # Make prediction
            if scaler is not None:
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)[0]
            else:
                prediction = model.predict(input_df)[0]
            
            # Display prediction
            st.success(f"🌊 Predicted Flood Probability: {prediction:.4f}")
            
            # Risk level assessment
            if prediction < 0.3:
                risk_level = "Low"
                color = "green"
            elif prediction < 0.7:
                risk_level = "Medium"
                color = "orange"
            else:
                risk_level = "High"
                color = "red"
            
            st.markdown(f"**Risk Level:** <span style='color: {color}; font-weight: bold;'>{risk_level}</span>", 
                       unsafe_allow_html=True)
            
            # Visualization
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prediction,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Flood Risk Level"},
                gauge = {
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.3], 'color': "lightgreen"},
                        {'range': [0.3, 0.7], 'color': "yellow"},
                        {'range': [0.7, 1], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.9
                    }
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True)

elif page == "📋 Feature Importance":
    st.header("📋 Feature Importance Analysis")
    
    if not st.session_state.models_trained:
        st.warning("Please train models first in the Model Training page.")
    else:
        results = st.session_state.results
        feature_names = st.session_state.feature_names
        
        # Select model for feature importance
        tree_based_models = ['Decision Tree', 'Random Forest', 'Gradient Boosting', 
                           'XGBoost', 'LightGBM', 'CatBoost', 'AdaBoost']
        
        available_models = [model for model in tree_based_models if model in results.keys()]
        
        if available_models:
            selected_model = st.selectbox(
                "Select Model for Feature Importance:",
                options=available_models
            )
            
            model = results[selected_model]['Model']
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                # Create importance dataframe
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                # Display top features
                st.subheader(f"Feature Importance - {selected_model}")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = px.bar(importance_df, 
                               x='Importance', y='Feature',
                               orientation='h',
                               title="Feature Importance Ranking")
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.dataframe(importance_df)
                
                # Top 5 features
                st.subheader("Top 5 Most Important Features")
                top_features = importance_df.head(5)
                
                for i, (_, row) in enumerate(top_features.iterrows()):
                    st.write(f"{i+1}. **{row['Feature']}** - {row['Importance']:.4f}")
            else:
                st.warning(f"{selected_model} does not provide feature importance.")
        else:
            st.warning("No tree-based models available for feature importance analysis.")

# Footer
st.markdown("---")
st.markdown("**FloodSentinel** - Developed for comprehensive flood risk assessment using machine learning")
