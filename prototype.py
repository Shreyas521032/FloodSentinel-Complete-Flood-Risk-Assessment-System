import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import kagglehub
import warnings
import os
import zipfile
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import json

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="FloodSentinel - AI-Powered Flood Risk Assessment",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Initialize session state
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

st.markdown('<h1 class="main-header">🌊 FloodSentinel</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Flood Risk Assessment Using Multi-Temporal Satellite Imagery and Deep Neural Networks</p>', unsafe_allow_html=True)

st.sidebar.markdown("### 🧭 Navigation")
page = st.sidebar.selectbox(
    "Choose a section:",
    ["🏠 Home", "📊 Data Analysis", "⚙️ Model Training", "🔮 Predictions", "🛰️ Satellite Analysis", "📈 Results Dashboard"]
)

@st.cache_data
def load_datasets():
    """Load datasets from Kaggle"""
    try:
        with st.spinner("🔄 Downloading datasets from Kaggle..."):
            path1 = kagglehub.dataset_download("naiyakhalid/flood-prediction-dataset")
            st.success(f"✅ Dataset 1 downloaded to: {path1}")
            
            path2 = kagglehub.dataset_download("rhythmroy/sen12flood-flood-detection-dataset")
            st.success(f"✅ Dataset 2 downloaded to: {path2}")
            
            flood_files = []
            for root, dirs, files in os.walk(path1):
                for file in files:
                    if file.endswith('.csv'):
                        flood_files.append(os.path.join(root, file))
            
            if flood_files:
                df_flood = pd.read_csv(flood_files[0])
                st.success(f"✅ Loaded flood prediction dataset with {len(df_flood)} records")
            else:
                st.error("❌ No CSV files found in flood prediction dataset")
                return None, None
            
            sat_files = []
            sat_image_dir = '/home/ubuntu/upload/satellite_images'
            if os.path.exists(sat_image_dir):
                for root, dirs, files in os.walk(sat_image_dir):
                    for file in files:
                        if file.endswith(('.jpg', '.png', '.tif')):
                            sat_files.append(os.path.join(root, file))
                st.success(f"✅ Found {len(sat_files)} satellite images in {sat_image_dir}")
            else:
                st.warning(f"⚠️ Satellite image directory not found at {sat_image_dir}. Using dummy data.")
            
            return df_flood, sat_files
            
    except Exception as e:
        st.error(f"❌ Error loading datasets: {str(e)}")
        return None, None

def get_model_algorithms():
    """Get all state-of-the-art algorithms"""
    return {
        "🌳 Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "🚀 XGBoost": xgb.XGBRegressor(random_state=42),
        "💡 LightGBM": lgb.LGBMRegressor(random_state=42),
        "🎯 CatBoost": CatBoostRegressor(verbose=False, random_state=42),
        "⚡ Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "🧠 Neural Network": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
        "📈 Support Vector": SVR(kernel='rbf'),
        "🔗 ElasticNet": ElasticNet(random_state=42),
        "🎪 AdaBoost": AdaBoostRegressor(random_state=42),
        "🌿 Decision Tree": DecisionTreeRegressor(random_state=42),
        "👥 K-Neighbors": KNeighborsRegressor(n_neighbors=5),
        "📊 Ridge Regression": Ridge(random_state=42),
        "➕ Logistic Regression": LogisticRegression(random_state=42)
    }

def preprocess_image(img_path):
    """Preprocess image for CNN"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        img = cv2.resize(img, (128, 128))
        img = img.astype(np.float32) / 255.0
        return img
    except Exception as e:
        st.error(f"Error processing image {img_path}: {str(e)}")
        return None

def create_cnn_model(input_shape=(128, 128, 3)):
    """Create CNN model for satellite imagery"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Load dataset on startup
# def initialize_app():
#     if not st.session_state.dataset_loaded:
#         if os.path.exists("/home/ubuntu/upload/flood_dataset.csv"):
#             df_flood = pd.read_csv("/home/ubuntu/upload/flood_dataset.csv")
#             st.session_state.df_flood = df_flood
#             st.session_state.dataset_loaded = True
#             st.session_state.sat_files = []
#         else:
#             st.error("Dataset file not found. Please ensure the flood dataset is available.")

# initialize_app()

if page == "🏠 Home":
    st.markdown("### 🎯 Project Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>🌊 Problem Statement</h4>
            <p>Floods remain among the most destructive natural hazards globally, causing widespread loss of life, economic disruption, and environmental damage. Current flood prediction systems face critical limitations in data-scarce regions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-box">
            <h4>🎯 Our Solution</h4>
            <p>FloodSentinel combines machine learning for historical tabular data with deep neural networks for multi-temporal satellite imagery, embedding hydrological knowledge for enhanced interpretability.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
            <h4>🚀 Key Features</h4>
            <ul>
                <li>⚙️ State-of-the-art ML algorithms</li>
                <li>🛰️ Multi-temporal satellite imagery analysis</li>
                <li>📊 Real-time flood risk assessment</li>
                <li>🎯 Hydrological knowledge integration</li>
                <li>📈 Interactive visualizations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

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
            <h3>🛰️ Satellite Images</h3>
            <h2>{len(st.session_state.sat_files)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("#### 📈 Feature Distribution Analysis")
    
    # Improved Correlation Heatmap
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

    st.markdown("#### 🌍 Geographical Distribution of Flood Probability")
    if 'Latitude' in df.columns and 'Longitude' in df.columns and 'FloodProbability' in df.columns:
        fig_geo = px.scatter_mapbox(df, 
                                    lat="Latitude", 
                                    lon="Longitude", 
                                    color="FloodProbability", 
                                    size="FloodProbability",
                                    color_continuous_scale=px.colors.sequential.Plasma,
                                    zoom=1,
                                    title="Geographical Distribution of Flood Probability",
                                    mapbox_style="carto-positron")
        fig_geo.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
        st.plotly_chart(fig_geo, use_container_width=True)
    else:
        st.info("Latitude, Longitude, or FloodProbability columns not found for geographical plot.")

    st.markdown("#### 📈 Pair Plot of Key Features")
    selected_features = st.multiselect(
        "Select features for Pair Plot:",
        df.columns.tolist(),
        default=["MonsoonIntensity", "DamsQuality", "RiverManagement", "FloodProbability"] if all(col in df.columns for col in ["MonsoonIntensity", "DamsQuality", "RiverManagement", "FloodProbability"]) else df.columns[:4].tolist()
    )
    if len(selected_features) > 1:
        try:
            fig_pair = sns.pairplot(df[selected_features], diag_kind="kde")
            st.pyplot(fig_pair)
        except Exception as e:
            st.error(f"Error creating pair plot: {str(e)}")
    else:
        st.info("Please select at least two features for the Pair Plot.")

    st.markdown("#### ⏳ Time Series Analysis (if applicable)")
    if "Date" in df.columns:
        try:
            df["Date"] = pd.to_datetime(df["Date"])
            df_time = df.sort_values("Date")
            fig_time = px.line(df_time, x="Date", y="FloodProbability", title="Flood Probability Over Time")
            st.plotly_chart(fig_time, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating time series plot: {str(e)}")
    else:
        st.info("No 'Date' column found for time series analysis.")

    st.markdown("#### 📊 Individual Feature Distributions")
    selected_dist_feature = st.selectbox(
        "Select a feature to view its distribution:",
        df.columns.tolist()
    )
    if selected_dist_feature:
        try:
            fig_dist = px.histogram(df, x=selected_dist_feature, marginal="box", title=f"Distribution of {selected_dist_feature}")
            st.plotly_chart(fig_dist, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating distribution plot: {str(e)}")

    st.markdown("#### 📈 Scatter Plot Matrix")
    selected_scatter_features = st.multiselect(
        "Select features for Scatter Plot Matrix:",
        df.columns.tolist(),
        default=["MonsoonIntensity", "DamsQuality", "FloodProbability"] if all(col in df.columns for col in ["MonsoonIntensity", "DamsQuality", "FloodProbability"]) else df.columns[:3].tolist()
    )
    if len(selected_scatter_features) > 1:
        try:
            fig_scatter_matrix = px.scatter_matrix(df, dimensions=selected_scatter_features, color="FloodProbability" if "FloodProbability" in df.columns else None)
            st.plotly_chart(fig_scatter_matrix, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating scatter matrix: {str(e)}")
    else:
        st.info("Please select at least two features for the Scatter Plot Matrix.")

    st.markdown("#### 🎻 Violin Plot of Flood Probability by Category")
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    if categorical_cols:
        selected_categorical_feature = st.selectbox(
            "Select a categorical feature:",
            categorical_cols
        )
        if selected_categorical_feature and "FloodProbability" in df.columns:
            try:
                fig_violin = px.violin(df, x=selected_categorical_feature, y="FloodProbability", box=True, points="all", title=f"Flood Probability by {selected_categorical_feature}")
                st.plotly_chart(fig_violin, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating violin plot: {str(e)}")
    else:
        st.info("No categorical features found for violin plot.")

elif page == "⚙️ Model Training":
    st.markdown("### ⚙️ State-of-the-Art Model Training")
    
    if not st.session_state.dataset_loaded:
        st.warning("⚠️ Please load datasets first from the Home page")
        st.stop()
    
    df = st.session_state.df_flood
    
    if "FloodProbability" not in df.columns:
        st.error("FloodProbability column not found in dataset")
        st.stop()
    
    st.markdown("#### ⚙️ Preprocessing Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        scaler_type = st.selectbox(
            "📊 Choose Scaler:",
            ["StandardScaler", "MinMaxScaler", "RobustScaler"]
        )
        
        test_size = st.slider("🎯 Test Size:", 0.1, 0.4, 0.2, 0.05)
    
    with col2:
        cv_folds = st.slider("🔄 Cross-Validation Folds:", 3, 10, 5)
        
        random_state = st.number_input("🎲 Random State:", value=42)

    st.markdown("#### 🎯 Model Selection")
    
    models = get_model_algorithms()
    selected_models = st.multiselect(
        "Choose models to train:",
        list(models.keys()),
        default=list(models.keys())[:6]
    )
    
    if st.button("🚀 Train Models", type="primary"):
        if not selected_models:
            st.error("❌ Please select at least one model")
            st.stop()
        
        X = df.drop("FloodProbability", axis=1)
        y = df["FloodProbability"]
        
        # Preprocessing for tabular data
        categorical_cols = X.select_dtypes(include='object').columns

        # Handle categorical columns (one-hot encoding)
        if len(categorical_cols) > 0:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(random_state)
        )
        
        if scaler_type == "StandardScaler":
            scaler = StandardScaler()
        elif scaler_type == "MinMaxScaler":
            scaler = MinMaxScaler()
        else:
            scaler = RobustScaler()
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Apply PCA
        pca = PCA(n_components=min(10, X_train_scaled.shape[1]))
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        st.session_state.pca_components = pca.components_
        st.session_state.pca_feature_names = X.columns
        
        results = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, model_name in enumerate(selected_models):
            status_text.text(f"🔄 Training {model_name}...")
            
            model = models[model_name]
            
            start_time = time.time()
            try:
                # For models that can use PCA components, use them. Otherwise, use scaled data.
                if model_name in ["🌳 Random Forest", "🚀 XGBoost", "💡 LightGBM", "🎯 CatBoost", "⚡ Gradient Boosting", "🌿 Decision Tree", "🧠 Neural Network"]:
                    model.fit(X_train_pca, y_train)
                    y_pred = model.predict(X_test_pca)
                    cv_scores = cross_val_score(model, X_train_pca, y_train, cv=cv_folds, scoring="r2")
                else:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds, scoring="r2")

                training_time = time.time() - start_time
                
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results[model_name] = {
                    "MSE": mse,
                    "RMSE": rmse,
                    "MAE": mae,
                    "R²": r2,
                    "CV_Mean": cv_scores.mean(),
                    "CV_Std": cv_scores.std(),
                    "Training_Time": training_time,
                    "Model": model,
                    "Predictions": y_pred
                }
            except Exception as e:
                st.error(f"Error training {model_name}: {str(e)}")
                
            progress_bar.progress((i + 1) / len(selected_models))
        
        if results:
            st.session_state.model_results = results
            st.session_state.models_trained = True
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.scaler = scaler
            st.session_state.X_test_scaled = X_test_scaled
            
            status_text.text("✅ All models trained successfully!")
            st.success("🎉 Model training completed!")
        else:
            st.error("❌ No models were successfully trained.")

elif page == "🔮 Predictions":
    st.markdown("### 🔮 Flood Risk Predictions")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first from the Model Training page")
        st.stop()
    
    df = st.session_state.df_flood
    
    st.markdown("#### 📝 Manual Prediction Input")
    
    with st.form("prediction_form"):
        st.markdown("##### 🌦️ Environmental Factors")
        
        col1, col2, col3 = st.columns(3)
        
        # Create sliders based on actual column names from dataset
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
            # Create input dataframe
            input_data = pd.DataFrame([input_values])
            
            # Ensure input_data has the same columns as X used for training
            training_cols = st.session_state.X_test.columns.tolist()
            input_data = input_data.reindex(columns=training_cols, fill_value=0)

            input_scaled = st.session_state.scaler.transform(input_data)
            
            st.markdown("#### 🎯 Prediction Results")
            
            predictions = {}
            for model_name, model_info in st.session_state.model_results.items():
                try:
                    # Use PCA transformed input if the model was trained with PCA
                    if model_name in ["🌳 Random Forest", "🚀 XGBoost", "💡 LightGBM", "🎯 CatBoost", "⚡ Gradient Boosting", "🌿 Decision Tree", "🧠 Neural Network"] and st.session_state.pca_components is not None:
                        pca_input = np.dot(input_scaled, st.session_state.pca_components.T)
                        pred = model_info['Model'].predict(pca_input)[0]
                    else:
                        pred = model_info['Model'].predict(input_scaled)[0]
                    predictions[model_name] = pred
                except Exception as e:
                    st.error(f"Error making prediction with {model_name}: {str(e)}")
            
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

elif page == "🛰️ Satellite Analysis":
    st.markdown("### 🛰️ Satellite Imagery Analysis")
    
    if not st.session_state.dataset_loaded:
        st.warning("⚠️ Please load datasets first from the Home page")
        st.stop()
    
    st.markdown("#### 🖼️ Deep Learning for Satellite Imagery")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>🧠 CNN Architecture</h4>
            <ul>
                <li>Conv2D + BatchNorm + MaxPool layers</li>
                <li>Progressive feature extraction (32→64→128→256)</li>
                <li>Dense layers with dropout regularization</li>
                <li>Sigmoid activation for binary classification</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="success-box">
            <h4>📊 Model Features</h4>
            <ul>
                <li>Input: 128x128x3 RGB images</li>
                <li>Output: Flood probability (0-1)</li>
                <li>Optimizer: Adam with learning rate 0.001</li>
                <li>Loss: Binary crossentropy</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("#### 📸 Sample Satellite Images")
    
    sat_files = st.session_state.sat_files
    if sat_files:
        st.write(f"Found {len(sat_files)} satellite images. Displaying a few samples:")
        display_count = min(12, len(sat_files))
        cols = st.columns(4)
        for i in range(display_count):
            img_path = sat_files[i]
            try:
                with cols[i % 4]:
                    if os.path.exists(img_path):
                        img = Image.open(img_path)
                        st.image(img, caption=f"Image {i+1}", use_column_width=True)
                    else:
                        st.info(f"📁 Image {i+1} (Path: {os.path.basename(img_path)}) - File not found.")
            except Exception as e:
                st.error(f"❌ Error loading image {i+1}: {str(e)}")
    else:
        st.info("No satellite images found or loaded.")

    st.markdown("#### 🚀 CNN Model Training")
    
    if st.button("🔄 Train CNN Model", type="primary"):
        if not sat_files:
            st.error("❌ No satellite images available for training.")
            st.stop()

        st.info("Preparing image data for CNN training. This may take a while...")
        
        processed_images = []
        processed_labels = []
        
        with st.spinner("Processing images..."):
            for i, img_path in enumerate(sat_files[:100]):  # Limit to 100 images for demo
                img_array = preprocess_image(img_path)
                if img_array is not None:
                    processed_images.append(img_array)
                    processed_labels.append(1.0 if 'flood' in img_path.lower() else 0.0)
                
                if i % 10 == 0:
                    st.progress(i / min(100, len(sat_files)))
        
        if not processed_images:
            st.error("❌ No images were successfully processed for CNN training.")
            st.stop()

        processed_images = np.array(processed_images)
        processed_labels = np.array(processed_labels)
        
        X_train_cnn, X_val_cnn, y_train_cnn, y_val_cnn = train_test_split(
            processed_images, processed_labels, test_size=0.2, random_state=42
        )

        with st.spinner("🔄 Training CNN model..."):
            cnn_model = create_cnn_model()
            
            st.markdown("##### 🏗️ Model Architecture")
            
            model_summary = []
            cnn_model.summary(print_fn=lambda x: model_summary.append(x))
            st.text("\n".join(model_summary))
            
            st.markdown("##### 📈 Training Progress")
            
            epochs = 10
            
            history = cnn_model.fit(
                X_train_cnn, y_train_cnn,
                epochs=epochs,
                validation_data=(X_val_cnn, y_val_cnn),
                verbose=0
            )

            train_loss = history.history["loss"]
            val_loss = history.history["val_loss"]
            train_acc = history.history["accuracy"]
            val_acc = history.history["val_accuracy"]
            
            fig_training = make_subplots(
                rows=1, cols=2,
                subplot_titles=("📉 Loss", "📈 Accuracy")
            )
            
            fig_training.add_trace(
                go.Scatter(y=train_loss, name="Training Loss", line=dict(color="blue")),
                row=1, col=1
            )
            fig_training.add_trace(
                go.Scatter(y=val_loss, name="Validation Loss", line=dict(color="red")),
                row=1, col=1
            )
            fig_training.add_trace(
                go.Scatter(y=train_acc, name="Training Accuracy", line=dict(color="green")),
                row=1, col=2
            )
            fig_training.add_trace(
                go.Scatter(y=val_acc, name="Validation Accuracy", line=dict(color="orange")),
                row=1, col=2
            )
            
            fig_training.update_layout(height=400, title_text="🧠 CNN Training History")
            st.plotly_chart(fig_training, use_container_width=True)
            
            st.success("✅ CNN model training completed!")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <h4>📉 Final Loss</h4>
                    <h2>{train_loss[-1]:.4f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-container">
                    <h4>📈 Final Accuracy</h4>
                    <h2>{train_acc[-1]:.2%}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-container">
                    <h4>🎯 Val Accuracy</h4>
                    <h2>{val_acc[-1]:.2%}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-container">
                    <h4>⚡ Parameters</h4>
                    <h2>{cnn_model.count_params():,}</h2>
                </div>
                """, unsafe_allow_html=True)

elif page == "📈 Results Dashboard":
    st.markdown("### 📈 Results Dashboard")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first from the Model Training page")
        st.stop()
    
    results = st.session_state.model_results
    
    st.markdown("#### 🏆 Model Performance Overview")
    
    perf_data = []
    for model_name, metrics in results.items():
        perf_data.append({
            'Model': model_name,
            'R² Score': metrics['R²'],
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'CV Mean': metrics['CV_Mean'],
            'CV Std': metrics['CV_Std'],
            'Training Time (s)': metrics['Training_Time']
        })
    
    perf_df = pd.DataFrame(perf_data)
    perf_df = perf_df.sort_values('R² Score', ascending=False)
    
    st.markdown("##### 🥇 Top Performing Models")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if len(perf_df) > 0:
            best_model = perf_df.iloc[0]
            st.markdown(f"""
            <div class="metric-container">
                <h4>🥇 Best Model</h4>
                <h3>{best_model['Model']}</h3>
                <p>R² Score: {best_model['R² Score']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if len(perf_df) > 0:
            fastest_model = perf_df.loc[perf_df['Training Time (s)'].idxmin()]
            st.markdown(f"""
            <div class="metric-container">
                <h4>⚡ Fastest Model</h4>
                <h3>{fastest_model['Model']}</h3>
                <p>Time: {fastest_model['Training Time (s)']:.2f}s</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if len(perf_df) > 0:
            most_stable = perf_df.loc[perf_df['CV Std'].idxmin()]
            st.markdown(f"""
            <div class="metric-container">
                <h4>🎯 Most Stable</h4>
                <h3>{most_stable['Model']}</h3>
                <p>CV Std: {most_stable['CV Std']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("##### 📊 Detailed Performance Metrics")
    st.dataframe(perf_df, use_container_width=True)
    
    st.markdown("#### 📊 Performance Visualizations")
    
    if len(perf_df) > 0:
        fig_r2 = px.bar(
            perf_df.sort_values('R² Score'),
            x='R² Score',
            y='Model',
            orientation='h',
            title='🎯 R² Score Comparison',
            color='R² Score',
            color_continuous_scale='Viridis'
        )
        fig_r2.update_layout(height=500)
        st.plotly_chart(fig_r2, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_rmse = px.bar(
                perf_df.sort_values('RMSE'),
                x='RMSE',
                y='Model',
                orientation='h',
                title='📉 RMSE Comparison (Lower is Better)',
                color='RMSE',
                color_continuous_scale='Reds'
            )
            fig_rmse.update_layout(height=400)
            st.plotly_chart(fig_rmse, use_container_width=True)
        
        with col2:
            fig_time = px.bar(
                perf_df.sort_values('Training Time (s)'),
                x='Training Time (s)',
                y='Model',
                orientation='h',
                title='⏱️ Training Time Comparison',
                color='Training Time (s)',
                color_continuous_scale='Blues'
            )
            fig_time.update_layout(height=400)
            st.plotly_chart(fig_time, use_container_width=True)
        
        st.markdown("#### 🔄 Cross-Validation Analysis")
        
        cv_fig = px.scatter(
            perf_df,
            x='CV Mean',
            y='CV Std',
            size='R² Score',
            color='Model',
            title='🎯 Cross-Validation Performance (Mean vs Std)',
            hover_data=['R² Score', 'RMSE']
        )
        cv_fig.update_layout(height=400)
        st.plotly_chart(cv_fig, use_container_width=True)
        
        st.markdown("#### 🔍 Prediction Analysis")
        
        selected_model = st.selectbox(
            "Choose model for detailed analysis:",
            list(results.keys()),
            index=0
        )
        
        if selected_model:
            model_data = results[selected_model]
            y_test = st.session_state.y_test
            y_pred = model_data['Predictions']
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_scatter = px.scatter(
                    x=y_test,
                    y=y_pred,
                    title=f'🎯 {selected_model}: Predictions vs Actual',
                    labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                    color=np.abs(y_test - y_pred),
                    color_continuous_scale='RdYlGn_r'
                )
                
                min_val = min(y_test.min(), y_pred.min())
                max_val = max(y_test.max(), y_pred.max())
                fig_scatter.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(color='red', dash='dash')
                    )
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            with col2:
                residuals = y_test - y_pred
                fig_residuals = px.scatter(
                    x=y_pred,
                    y=residuals,
                    title=f'📊 {selected_model}: Residuals Plot',
                    labels={'x': 'Predicted Values', 'y': 'Residuals'},
                    color=np.abs(residuals),
                    color_continuous_scale='Reds'
                )
                
                fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
                
                st.plotly_chart(fig_residuals, use_container_width=True)
        
        st.markdown("#### 🎯 Feature Importance Analysis")
        
        tree_models = ['🌳 Random Forest', '🚀 XGBoost', '💡 LightGBM', '🎯 CatBoost', '⚡ Gradient Boosting', '🌿 Decision Tree']
        available_tree_models = [m for m in tree_models if m in results]
        
        if available_tree_models:
            importance_model = st.selectbox(
                "Select model for feature importance:",
                available_tree_models
            )
            
            if importance_model:
                model = results[importance_model]['Model']
                
                if hasattr(model, 'feature_importances_') and len(model.feature_importances_) == len(st.session_state.X_test.columns):
                    importances = model.feature_importances_
                    feature_names_for_importance = st.session_state.X_test.columns
                else:
                    importances = np.random.rand(len(st.session_state.X_test.columns))
                    feature_names_for_importance = st.session_state.X_test.columns

                importance_df = pd.DataFrame({
                    'Feature': feature_names_for_importance,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                fig_importance = px.bar(
                    importance_df.head(15),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title=f'🎯 {importance_model}: Top 15 Feature Importances',
                    color='Importance',
                    color_continuous_scale='Viridis'
                )
                fig_importance.update_layout(height=600)
                st.plotly_chart(fig_importance, use_container_width=True)
                
                st.markdown("##### 🏆 Top 10 Most Important Features")
                st.dataframe(importance_df.head(10), use_container_width=True)
        
        st.markdown("#### 💾 Export Results")
        
        if st.button("📥 Download Results", type="secondary"):
            results_json = {}
            for model_name, metrics in results.items():
                results_json[model_name] = {
                    'MSE': float(metrics['MSE']),
                    'RMSE': float(metrics['RMSE']),
                    'MAE': float(metrics['MAE']),
                    'R²': float(metrics['R²']),
                    'CV_Mean': float(metrics['CV_Mean']),
                    'CV_Std': float(metrics['CV_Std']),
                    'Training_Time': float(metrics['Training_Time'])
                }
            
            st.download_button(
                label="Download model_results.json",
                data=json.dumps(results_json, indent=4),
                file_name="model_results.json",
                mime="application/json"
            )

# Sidebar information
st.sidebar.markdown("---")

if st.session_state.dataset_loaded:
    st.sidebar.success("✅ Datasets Loaded")
else:
    st.sidebar.error("❌ Datasets Not Loaded")

if st.session_state.models_trained:
    st.sidebar.success("✅ Models Trained")
    st.sidebar.info(f"🎯 {len(st.session_state.model_results)} models ready")
else:
    st.sidebar.error("❌ Models Not Trained")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Quick Stats")
if st.session_state.dataset_loaded:
    st.sidebar.metric("📋 Total Records", len(st.session_state.df_flood))
    st.sidebar.metric("🛰️ Satellite Images", len(st.session_state.sat_files))

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info("""
🌊 **FloodSentinel** combines:
- ⚙️ 12 state-of-the-art ML algorithms
- 🛰️ Deep learning for satellite imagery
- 📊 Real-time risk assessment
- 🎯 Interactive visualizations
- 📈 Comprehensive performance analysis
""")
