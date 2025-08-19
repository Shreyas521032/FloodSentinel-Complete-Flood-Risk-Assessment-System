import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import kagglehub
import warnings
import os
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import json

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="FloodSentinel - AI-Powered Flood Risk Assessment",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Utilities
# -----------------------------
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

def safe_open_image(path, target_size=(256, 256)):
    """Open various image formats robustly, including 16-bit GeoTIFFs, and convert to RGB for display."""
    try:
        img = Image.open(path)
        img = img.convert("RGB")
        img = img.resize(target_size)
        return img
    except Exception:
        try:
            arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if arr is None:
                raise ValueError("cv2 failed to read")
            # If single-channel, stack to RGB; if multi-channel, take first 3
            if len(arr.shape) == 2:
                arr = np.stack([arr]*3, axis=-1)
            elif arr.shape[2] > 3:
                arr = arr[:, :, :3]
            # Normalize to 0-255 if 16-bit
            if arr.dtype != np.uint8:
                arr = arr.astype(np.float32)
                arr -= arr.min()
                if arr.max() > 0:
                    arr /= arr.max()
                arr = (arr * 255).astype(np.uint8)
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            arr = cv2.resize(arr, target_size)
            return Image.fromarray(arr)
        except Exception:
            return None

def scan_image_files(root_dir):
    """Recursively scan for image files under root_dir."""
    image_paths = []
    for r, d, fns in os.walk(root_dir):
        for f in fns:
            if f.lower().endswith(IMAGE_EXTS):
                image_paths.append(os.path.join(r, f))
    return image_paths

# -----------------------------
# Streamlit app header
# -----------------------------
st.markdown("""
<style>
    .main-header {font-size: 3rem; font-weight: bold; text-align: center; background: linear-gradient(90deg, #1e3c72, #2a5298); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 2rem;}
    .metric-container {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center; margin: 0.5rem;}
    .success-box {background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 1rem; border-radius: 10px; color: white; margin: 1rem 0;}
    .warning-box {background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); padding: 1rem; border-radius: 10px; color: white; margin: 1rem 0;}
    .info-box {background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); padding: 1rem; border-radius: 10px; color: #333; margin: 1rem 0;}
</style>
""", unsafe_allow_html=True)

if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'dataset_loaded' not in st.session_state:
    st.session_state.dataset_loaded = False
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}

if 'use_pca' not in st.session_state:
    st.session_state.use_pca = True
if 'pca_n_components' not in st.session_state:
    st.session_state.pca_n_components = 10  # default as requested

st.markdown('<h1 class="main-header">🌊 FloodSentinel</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Flood Risk Assessment Using Multi-Temporal Satellite Imagery and Deep Neural Networks</p>', unsafe_allow_html=True)

st.sidebar.markdown("### 🧭 Navigation")
page = st.sidebar.selectbox(
    "Choose a section:",
    ["🏠 Home", "📊 Data Analysis", "⚙️ Model Training", "🔮 Predictions", "🛰️ Satellite Analysis", "📈 Results Dashboard"]
)

# -----------------------------
# Data loading
# -----------------------------
@st.cache_data(show_spinner=False)
def load_datasets():
    """Load datasets via KaggleHub and index satellite imagery correctly for SEN12FLOOD."""
    try:
        with st.spinner("🔄 Downloading datasets from Kaggle (first run only)..."):
            path1 = kagglehub.dataset_download("naiyakhalid/flood-prediction-dataset")
            path2 = kagglehub.dataset_download("rhythmroy/sen12flood-flood-detection-dataset")

        # Load tabular CSV
        flood_csvs = []
        for root, _, files in os.walk(path1):
            for f in files:
                if f.lower().endswith('.csv'):
                    flood_csvs.append(os.path.join(root, f))
        if not flood_csvs:
            st.error("❌ No CSV found in flood prediction dataset")
            return None, None, None, None
        df_flood = pd.read_csv(flood_csvs[0])

        # SEN12FLOOD integration: prefer JSON lists when present for labeled training
        s1_json = os.path.join(path2, 'S1list.json')
        s2_json = os.path.join(path2, 'S2list.json')

        labeled_images = []  # [{'path':..., 'label':0/1}]
        if os.path.exists(s1_json):
            try:
                with open(s1_json, 'r') as f:
                    s1_data = json.load(f)
                for _, seq in s1_data.items():
                    folder = seq.get('folder', '')
                    series = seq.get('series', [])
                    for item in series:
                        prefix = item.get('prefix')
                        flooding = item.get('FLOODING', False)
                        # Try both VV and VH files (VV commonly used)
                        candidates = [f"{prefix}_VV.tif", f"{prefix}_VH.tif"]
                        for name in candidates:
                            p = os.path.join(path2, folder, name)
                            if os.path.exists(p):
                                labeled_images.append({'path': p, 'label': int(bool(flooding))})
                                break
            except Exception as e:
                st.warning(f"⚠️ Could not parse S1list.json properly: {e}")

        # Fallback and also global count: scan all image files
        all_image_files = scan_image_files(path2)

        return df_flood, labeled_images, all_image_files, path2

    except Exception as e:
        st.error(f"❌ Error loading datasets: {e}")
        return None, None, None, None

# -----------------------------
# Models and CNN
# -----------------------------

def get_model_algorithms():
    return {
        "🌳 Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "🚀 XGBoost": xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, subsample=0.8, colsample_bytree=0.9, random_state=42),
        "💡 LightGBM": lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42),
        "🎯 CatBoost": CatBoostRegressor(verbose=False, random_state=42),
        "⚡ Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "🧠 Neural Network": MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=800, random_state=42),
        "📈 Support Vector": SVR(kernel='rbf', C=10, gamma='scale'),
        "🔗 ElasticNet": ElasticNet(random_state=42),
        "🎪 AdaBoost": AdaBoostRegressor(random_state=42),
        "🌿 Decision Tree": DecisionTreeRegressor(max_depth=None, random_state=42),
        "👥 K-Neighbors": KNeighborsRegressor(n_neighbors=7),
        "📊 Ridge Regression": Ridge(random_state=42)
    }


def create_cnn_model(input_shape=(128, 128, 3)):
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
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# -----------------------------
# Preprocessing helpers (Impute -> Scale -> optional PCA)
# -----------------------------

def build_preprocessor(scaler_type="StandardScaler", use_pca=True, n_components=10):
    if scaler_type == "StandardScaler":
        scaler = StandardScaler()
    elif scaler_type == "MinMaxScaler":
        scaler = MinMaxScaler()
    else:
        scaler = RobustScaler()
    imputer = SimpleImputer(strategy='median')
    pca = PCA(n_components=n_components) if use_pca else None
    return imputer, scaler, pca


def fit_transform_preprocessor(X_train, X_test, scaler_type, use_pca, n_components):
    imputer, scaler, pca = build_preprocessor(scaler_type, use_pca, n_components)
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)
    if use_pca:
        X_train_proc = pca.fit_transform(X_train_scaled)
        X_test_proc = pca.transform(X_test_scaled)
    else:
        X_train_proc, X_test_proc = X_train_scaled, X_test_scaled
    return X_train_proc, X_test_proc, imputer, scaler, pca

# -----------------------------
# Pages
# -----------------------------
if page == "🏠 Home":
    st.markdown("### 🎯 Project Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>🌊 Problem Statement</h4>
            <p>Floods cause severe socio-economic impacts. We combine tabular risk indicators with satellite imagery to assess and predict flood risk.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="warning-box">
            <h4>🚀 Key Features</h4>
            <ul>
                <li>⚙️ Robust preprocessing (imputation, scaling, PCA)</li>
                <li>🛰️ SEN12FLOOD satellite dataset integration</li>
                <li>📊 PCA insights and variance explained</li>
                <li>🔮 Prediction using top-10 PCA components</li>
                <li>📈 Interactive dashboards</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🔄 Load Datasets", type="primary"):
        df_flood, labeled_images, all_images, sat_root = load_datasets()
        if df_flood is not None:
            st.session_state.df_flood = df_flood
            st.session_state.labeled_images = labeled_images or []
            st.session_state.all_sat_image_files = all_images or []
            st.session_state.sat_root = sat_root
            st.session_state.dataset_loaded = True
            st.success("✅ Datasets loaded successfully!")
            st.info(f"🛰️ Satellite dataset indexed: {len(st.session_state.all_sat_image_files):,} image files found.")
        else:
            st.error("❌ Failed to load datasets")

elif page == "📊 Data Analysis":
    st.markdown("### 📊 Exploratory Data Analysis with Preprocessing and PCA")
    if not st.session_state.dataset_loaded:
        st.warning("⚠️ Please load datasets first from the Home page")
        st.stop()
    df = st.session_state.df_flood.copy()

    # Basic metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-container"><h3>📋 Records</h3><h2>{len(df)}</h2></div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-container"><h3>📊 Features</h3><h2>{len(df.columns)-1}</h2></div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-container"><h3>🎯 Target</h3><h2>FloodProbability</h2></div>
        """, unsafe_allow_html=True)
    with c4:
        total_imgs = len(st.session_state.all_sat_image_files) if 'all_sat_image_files' in st.session_state else 0
        st.markdown(f"""
        <div class="metric-container"><h3>🛰️ Satellite Images</h3><h2>{total_imgs:,}</h2></div>
        """, unsafe_allow_html=True)

    st.markdown("#### 🧹 Preprocessing Steps")
    # Show missing values and plan
    na_counts = df.isna().sum().sort_values(ascending=False)
    st.write("Missing values per column (top 15):", na_counts.head(15))
    st.info("Imputation: median for numeric features. Scaling: selectable. PCA: optional for dimensionality reduction.")

    # Correlation heatmap
    corr_matrix = df.corr(numeric_only=True).round(2)
    fig = go.Figure(data=[go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdYlBu_r', zmin=-1, zmax=1,
        text=corr_matrix.values, texttemplate="%{text}", colorbar=dict(title="Correlation")
    )], layout=go.Layout(title="🔥 Feature Correlation Heatmap", height=600))
    st.plotly_chart(fig, use_container_width=True)

    # PCA Exploration on full data (after impute+scale) for insights only
    st.markdown("#### 🧠 PCA Insights (Exploratory)")
    scaler_choice = st.selectbox("Scaler for EDA (does not affect training until next page):", ["StandardScaler", "MinMaxScaler", "RobustScaler"], index=["StandardScaler","MinMaxScaler","RobustScaler"].index("StandardScaler"))
    pca_k = st.slider("Number of PCA components to visualize", min_value=2, max_value=min(30, max(2, df.shape[1]-1)), value=10)

    feature_cols = [c for c in df.columns if c != 'FloodProbability' and np.issubdtype(df[c].dtype, np.number)]
    X_full = df[feature_cols]
    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X_full)
    if scaler_choice == "StandardScaler":
        sc = StandardScaler()
    elif scaler_choice == "MinMaxScaler":
        sc = MinMaxScaler()
    else:
        sc = RobustScaler()
    X_scaled = sc.fit_transform(X_imp)
    pca_eda = PCA(n_components=pca_k).fit(X_scaled)
    explained = pca_eda.explained_variance_ratio_

    ev_fig = px.bar(x=[f"PC{i+1}" for i in range(len(explained))], y=explained, title="Variance Explained by Components")
    ev_fig.add_trace(go.Scatter(x=[f"PC{i+1}" for i in range(len(explained))], y=np.cumsum(explained), mode='lines+markers', name='Cumulative'))
    st.plotly_chart(ev_fig, use_container_width=True)

    # 2D scatter of first two PCs colored by target
    pcs = pca_eda.transform(X_scaled)
    pca_df = pd.DataFrame({"PC1": pcs[:,0], "PC2": pcs[:,1], "FloodProbability": df['FloodProbability'].values})
    st.plotly_chart(px.scatter(pca_df, x='PC1', y='PC2', color='FloodProbability', title="PC1 vs PC2 colored by FloodProbability", color_continuous_scale='Viridis'), use_container_width=True)

    # Feature correlations with target
    st.markdown("#### 🎯 Top Features by Correlation with Target")
    corrs = df[feature_cols + ['FloodProbability']].corr()['FloodProbability'].drop('FloodProbability').abs().sort_values(ascending=False)
    st.plotly_chart(px.bar(x=corrs.values[:20], y=corrs.index[:20], orientation='h', title="Top 20 Features"), use_container_width=True)

elif page == "⚙️ Model Training":
    st.markdown("### ⚙️ Train Models with Robust Preprocessing and Optional PCA")
    if not st.session_state.dataset_loaded:
        st.warning("⚠️ Please load datasets first from the Home page")
        st.stop()

    df = st.session_state.df_flood.copy()
    feature_cols = [c for c in df.columns if c != 'FloodProbability' and np.issubdtype(df[c].dtype, np.number)]
    X = df[feature_cols]
    y = df['FloodProbability']

    c1, c2 = st.columns(2)
    with c1:
        scaler_type = st.selectbox("📊 Scaler:", ["StandardScaler", "MinMaxScaler", "RobustScaler"], index=0)
        test_size = st.slider("🎯 Test Size:", 0.1, 0.4, 0.2, 0.05)
        use_pca = st.checkbox("Use PCA", value=True)
    with c2:
        cv_folds = st.slider("🔄 Cross-Validation Folds:", 3, 10, 5)
        random_state = st.number_input("🎲 Random State:", value=42)
        n_components = st.slider("# PCA Components (used in training)", min_value=2, max_value=min(50, len(feature_cols)), value=10)

    if st.button("🚀 Train Models", type="primary"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        X_train_proc, X_test_proc, imputer, scaler, pca = fit_transform_preprocessor(X_train, X_test, scaler_type, use_pca, n_components)

        models = get_model_algorithms()
        selected_models = list(models.keys())  # train all by default; UI can be added if needed
        results = {}
        progress = st.progress(0)
        status = st.empty()

        for i, name in enumerate(selected_models):
            status.text(f"🔄 Training {name}...")
            model = models[name]
            start = time.time()
            model.fit(X_train_proc, y_train)
            tr_time = time.time() - start
            y_pred = model.predict(X_test_proc)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            try:
                cv_scores = cross_val_score(model, X_train_proc, y_train, cv=cv_folds, scoring='r2')
                cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
            except Exception:
                cv_mean, cv_std = np.nan, np.nan
            results[name] = {
                'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2,
                'CV_Mean': cv_mean, 'CV_Std': cv_std, 'Training_Time': tr_time,
                'Model': model, 'Predictions': y_pred
            }
            progress.progress((i+1)/len(selected_models))

        # Save artifacts for prediction stage
        st.session_state.model_results = results
        st.session_state.models_trained = True
        st.session_state.X_columns = feature_cols
        st.session_state.imputer = imputer
        st.session_state.scaler = scaler
        st.session_state.pca = pca
        st.session_state.use_pca = use_pca
        st.session_state.pca_n_components = n_components
        st.session_state.y_test = y_test
        st.session_state.X_test_proc = X_test_proc

        status.text("✅ Training complete")
        st.success("🎉 Models trained with preprocessing and PCA configuration saved for predictions.")

elif page == "🔮 Predictions":
    st.markdown("### 🔮 Flood Risk Predictions using Top PCA Components")
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first from the Model Training page")
        st.stop()

    # Show configuration
    st.info(f"Using PCA: {st.session_state.use_pca} | Components used: {st.session_state.pca_n_components}")

    with st.form("prediction_form"):
        st.markdown("##### 🌦️ Environmental Factors (enter values between 0 and 1)")
        # Build inputs dynamically from training columns if they match known names
        inputs = {}
        cols = st.columns(3)
        for i, col in enumerate(st.session_state.X_columns):
            default_val = 0.5
            try:
                minv, maxv = float(np.nanmin(st.session_state.df_flood[col])), float(np.nanmax(st.session_state.df_flood[col]))
                default_val = float(np.clip(np.nanmean(st.session_state.df_flood[col]), 0.0, 1.0)) if np.isfinite(minv) and np.isfinite(maxv) else 0.5
            except Exception:
                pass
            with cols[i % 3]:
                inputs[col] = st.slider(col, 0.0, 1.0, default_val)
        submitted = st.form_submit_button("⚡ Predict Flood Risk")

    if submitted:
        input_df = pd.DataFrame([{k: v for k, v in inputs.items()}])
        # Apply same preprocessing
        X_imp = st.session_state.imputer.transform(input_df)
        X_scaled = st.session_state.scaler.transform(X_imp)
        if st.session_state.use_pca and st.session_state.pca is not None:
            X_proc = st.session_state.pca.transform(X_scaled)
        else:
            X_proc = X_scaled

        predictions = {}
        for name, info in st.session_state.model_results.items():
            pred = float(info['Model'].predict(X_proc)[0])
            predictions[name] = pred

        col1, col2 = st.columns(2)
        with col1:
            ensemble_pred = float(np.mean(list(predictions.values())))
            risk_level = "🟢 Low Risk" if ensemble_pred < 0.3 else ("🟡 Medium Risk" if ensemble_pred < 0.6 else "🔴 High Risk")
            st.markdown(f"""
            <div class=\"metric-container\"> <h3>🎯 Ensemble Prediction</h3> <h1>{ensemble_pred:.2%}</h1> <h4>{risk_level}</h4> </div>
            """, unsafe_allow_html=True)
        with col2:
            pred_df = pd.DataFrame({
                'Model': list(predictions.keys()),
                'Prediction': [f"{p:.2%}" for p in predictions.values()],
                'Risk_Level': ["🟢 Low" if p < 0.3 else "🟡 Medium" if p < 0.6 else "🔴 High" for p in predictions.values()]
            })
            st.dataframe(pred_df, use_container_width=True)
        st.plotly_chart(px.bar(x=list(predictions.keys()), y=list(predictions.values()), title="📊 Model Predictions Comparison", color=list(predictions.values()), color_continuous_scale="RdYlGn_r"), use_container_width=True)

elif page == "🛰️ Satellite Analysis":
    st.markdown("### 🛰️ Satellite Imagery Analysis (SEN12FLOOD)")
    if not st.session_state.dataset_loaded:
        st.warning("⚠️ Please load datasets first from the Home page")
        st.stop()

    # Dataset linkage summary
    total_found = len(st.session_state.all_sat_image_files)
    st.success(f"✅ Linked SEN12FLOOD dataset. {total_found:,} image files found.")
    if total_found == 36107:
        st.info("This matches the expected 36,107 images.")

    labeled = st.session_state.labeled_images
    st.markdown(f"Labeled Sentinel-1 images available: {len(labeled):,}")

    # Filters
    colf1, colf2, colf3 = st.columns(3)
    with colf1:
        show_only_labeled = st.checkbox("Show only labeled S1 images", value=True)
    with colf2:
        label_filter = st.selectbox("Label filter", options=["All", "Flooded (1)", "Not Flooded (0)"], index=0)
    with colf3:
        page_size = st.selectbox("Images per page", options=[12, 24, 48], index=0)

    # Build gallery list
    if show_only_labeled and labeled:
        gallery = labeled
        if label_filter != "All":
            want = 1 if "Flooded" in label_filter else 0
            gallery = [g for g in gallery if g['label'] == want]
        gallery_paths = [g['path'] for g in gallery]
    else:
        gallery_paths = st.session_state.all_sat_image_files

    total_gallery = len(gallery_paths)
    st.markdown(f"Showing {min(page_size, total_gallery)} of {total_gallery:,} images on this page.")

    # Pagination
    num_pages = max(1, int(np.ceil(total_gallery / page_size)))
    page_idx = st.number_input("Page", min_value=1, max_value=num_pages, value=1)
    start = (page_idx - 1) * page_size
    end = min(start + page_size, total_gallery)

    cols = st.columns(4)
    for i, p in enumerate(gallery_paths[start:end]):
        img = safe_open_image(p, target_size=(256, 256))
        with cols[i % 4]:
            if img is not None:
                cap = os.path.basename(p)
                if show_only_labeled and labeled:
                    # find label
                    lab = next((li['label'] for li in labeled if li['path'] == p), None)
                    cap += f" | Flood: {lab}"
                st.image(img, caption=cap, use_column_width=True)
            else:
                st.info(f"Preview not supported: {os.path.basename(p)}")

    st.markdown("#### 🚀 Train a CNN on Labeled Sentinel-1 Thumbnails")
    if st.button("🔄 Train CNN Model", type="primary"):
        if not labeled:
            st.error("No labeled Sentinel-1 images parsed from S1list.json. Cannot train CNN.")
        else:
            with st.spinner("Training CNN on labeled images (using flow_from_dataframe)..."):
                cnn_model = create_cnn_model()
                # Prepare dataframe
                df_imgs = pd.DataFrame(labeled)
                df_imgs['label'] = df_imgs['label'].astype(str)
                train_df, val_df = train_test_split(df_imgs, test_size=0.2, random_state=42, stratify=df_imgs['label'])
                train_gen = ImageDataGenerator(rescale=1./255, shear_range=0.1, zoom_range=0.1, horizontal_flip=True)
                val_gen = ImageDataGenerator(rescale=1./255)
                train_flow = train_gen.flow_from_dataframe(train_df, x_col='path', y_col='label', target_size=(128,128), batch_size=32, class_mode='binary')
                val_flow = val_gen.flow_from_dataframe(val_df, x_col='path', y_col='label', target_size=(128,128), batch_size=32, class_mode='binary')
                history = cnn_model.fit(train_flow, steps_per_epoch=max(1, train_flow.n // train_flow.batch_size), epochs=5, validation_data=val_flow, validation_steps=max(1, val_flow.n // val_flow.batch_size))
                # Plot history
                fig_training = make_subplots(rows=1, cols=2, subplot_titles=('📉 Loss', '📈 Accuracy'))
                fig_training.add_trace(go.Scatter(y=history.history['loss'], name='Train Loss', line=dict(color='blue')), row=1, col=1)
                fig_training.add_trace(go.Scatter(y=history.history['val_loss'], name='Val Loss', line=dict(color='red')), row=1, col=1)
                fig_training.add_trace(go.Scatter(y=history.history['accuracy'], name='Train Acc', line=dict(color='green')), row=1, col=2)
                fig_training.add_trace(go.Scatter(y=history.history['val_accuracy'], name='Val Acc', line=dict(color='orange')), row=1, col=2)
                fig_training.update_layout(height=400, title_text="🧠 CNN Training History")
                st.plotly_chart(fig_training, use_container_width=True)
                st.success("✅ CNN model training completed!")

elif page == "📈 Results Dashboard":
    st.markdown("### 📈 Model Results and Diagnostics")
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first from the Model Training page")
        st.stop()

    results = st.session_state.model_results
    perf_data = []
    for name, m in results.items():
        perf_data.append({'Model': name, 'R² Score': m['R²'], 'RMSE': m['RMSE'], 'MAE': m['MAE'], 'CV Mean': m['CV_Mean'], 'CV Std': m['CV_Std'], 'Training Time (s)': m['Training_Time']})
    perf_df = pd.DataFrame(perf_data).sort_values('R² Score', ascending=False)

    # Highlights
    c1, c2, c3 = st.columns(3)
    with c1:
        best_row = perf_df.iloc[0]
        st.markdown(f"""<div class='metric-container'><h4>🥇 Best Model</h4><h3>{best_row['Model']}</h3><p>R²: {best_row['R² Score']:.3f}</p></div>""", unsafe_allow_html=True)
    with c2:
        fast_row = perf_df.loc[perf_df['Training Time (s)'].idxmin()]
        st.markdown(f"""<div class='metric-container'><h4>⚡ Fastest</h4><h3>{fast_row['Model']}</h3><p>{fast_row['Training Time (s)']:.2f}s</p></div>""", unsafe_allow_html=True)
    with c3:
        stable_row = perf_df.loc[perf_df['CV Std'].idxmin()]
        st.markdown(f"""<div class='metric-container'><h4>🎯 Most Stable</h4><h3>{stable_row['Model']}</h3><p>CV Std: {stable_row['CV Std']:.3f}</p></div>""", unsafe_allow_html=True)

    st.dataframe(perf_df, use_container_width=True)

    st.plotly_chart(px.bar(perf_df.sort_values('R² Score'), x='R² Score', y='Model', orientation='h', title='🎯 R² Score', color='R² Score', color_continuous_scale='Viridis'), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.bar(perf_df.sort_values('RMSE'), x='RMSE', y='Model', orientation='h', title='📉 RMSE', color='RMSE', color_continuous_scale='Reds'), use_container_width=True)
    with c2:
        st.plotly_chart(px.bar(perf_df.sort_values('Training Time (s)'), x='Training Time (s)', y='Model', orientation='h', title='⏱️ Training Time', color='Training Time (s)', color_continuous_scale='Blues'), use_container_width=True)

    st.markdown("#### 🔍 Prediction Diagnostics")
    sel_model = st.selectbox("Model for detailed plots", list(results.keys()))
    if sel_model:
        y_test = st.session_state.y_test
        y_pred = st.session_state.model_results[sel_model]['Predictions']
        fig_scatter = px.scatter(x=y_test, y=y_pred, title=f'{sel_model}: Predictions vs Actual', labels={'x':'Actual', 'y':'Predicted'}, color=np.abs(y_test - y_pred), color_continuous_scale='RdYlGn_r')
        min_val, max_val = float(min(y_test.min(), y_pred.min())), float(max(y_test.max(), y_pred.max()))
        fig_scatter.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Perfect', line=dict(color='red', dash='dash')))
        st.plotly_chart(fig_scatter, use_container_width=True)
        residuals = y_test - y_pred
        fig_res = px.scatter(x=y_pred, y=residuals, title=f'{sel_model}: Residuals', labels={'x':'Predicted', 'y':'Residuals'}, color=np.abs(residuals), color_continuous_scale='Reds')
        fig_res.add_hline(y=0, line_dash='dash', line_color='red')
        st.plotly_chart(fig_res, use_container_width=True)

    st.markdown("#### 🎯 Feature Importance (Tree Models)")
    tree_like = ['🌳 Random Forest', '🚀 XGBoost', '💡 LightGBM', '🎯 CatBoost', '⚡ Gradient Boosting']
    avail_trees = [m for m in tree_like if m in results]
    if avail_trees:
        imp_model = st.selectbox("Select model for importance", avail_trees)
        if imp_model:
            model = results[imp_model]['Model']
            if hasattr(model, 'feature_importances_'):
                # If PCA used, feature importance is on PCs; map to PC names
                if st.session_state.use_pca and st.session_state.pca is not None:
                    feat_names = [f"PC{i+1}" for i in range(st.session_state.pca_n_components)]
                else:
                    feat_names = st.session_state.X_columns
                importances = model.feature_importances_
                df_imp = pd.DataFrame({'Feature': feat_names[:len(importances)], 'Importance': importances}).sort_values('Importance', ascending=False)
                st.plotly_chart(px.bar(df_imp.head(15), x='Importance', y='Feature', orientation='h', title=f'{imp_model}: Top Importances'), use_container_width=True)
                st.dataframe(df_imp.head(10), use_container_width=True)


