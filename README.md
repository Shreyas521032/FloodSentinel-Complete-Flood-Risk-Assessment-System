# 🌊 FloodSentinel: Complete Flood-Risk Assessment System
[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/Shreyas521032/FloodSentinel-Complete-Flood-Risk-Assessment-System)
![Streamlit App](https://img.shields.io/badge/Streamlit-Deployed-ff4b4b?logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow-orange)
![Multi-Modal](https://img.shields.io/badge/Feature-Multi--Modal%20Analysis-purple)

FloodSentinel is a state-of-the-art flood risk assessment system built with Streamlit. It leverages machine learning and deep learning to analyze numerical data and satellite imagery, providing a comprehensive platform for flood prediction, model evaluation, and risk visualization. The system integrates directly with KaggleHub to fetch datasets automatically.

## ✨ Key Features

- **Multi-Modal Data Analysis**: Processes both numerical/tabular data (weather, terrain, hydrological factors) and satellite imagery for a holistic risk assessment.
- **Extensive Model Library**: Implements a wide range of machine learning models including Random Forest, XGBoost, LightGBM, CatBoost, SVM, and more for tabular data.
- **Advanced Deep Learning**: Utilizes Deep Neural Networks (DNNs), Convolutional Neural Networks (CNNs), and hybrid CNN-LSTM models for complex pattern recognition in both tabular and image data.
- **Interactive Dashboard**: A user-friendly web application built with Streamlit for data exploration, model training, real-time prediction, and results visualization.
- **Automated Data Ingestion**: Seamlessly downloads and loads required datasets from KaggleHub with a single click.
- **Comprehensive Model Evaluation**: Provides detailed performance metrics (Accuracy, R², RMSE, AUC, ROC Curves), model comparisons, feature importance analysis, and residual plots.
- **Real-Time Risk Prediction**: Allows users to input environmental parameters to receive an immediate flood risk probability score from an ensemble of trained models.
- **Simulated Satellite Analysis**: Includes a module for training CNNs on satellite images and visualizing simulated flood extent maps and temporal water level changes.

## 🛠️ Technology Stack

- **Dashboard**: Streamlit
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM, CatBoost
- **Deep Learning**: TensorFlow, Keras
- **Data Manipulation**: Pandas, NumPy
- **Data Visualization**: Plotly, Matplotlib, Seaborn
- **Image Processing**: OpenCV, Pillow
- **Data Source**: KaggleHub

## 📊 Datasets Used

The application automatically downloads the following datasets from Kaggle:
1.  **Flood Prediction Dataset**: A numerical dataset containing various environmental factors for predicting flood probability.
    -   [naiyakhalid/flood-prediction-dataset](https://www.kaggle.com/datasets/naiyakhalid/flood-prediction-dataset)
2.  **SEN12FLOOD Flood Detection Dataset**: A dataset containing satellite imagery for flood detection tasks.
    -   [rhythmroy/sen12flood-flood-detection-dataset](https://www.kaggle.com/datasets/rhythmroy/sen12flood-flood-detection-dataset)

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- [Kaggle API Token](https://www.kaggle.com/docs/api): You need to have your `kaggle.json` file set up in your home directory (e.g., `~/.kaggle/kaggle.json`) for `kagglehub` to work.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/shreyas521032/FloodSentinel-Complete-Flood-Risk-Assessment-System.git
    cd FloodSentinel-Complete-Flood-Risk-Assessment-System
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

This repository contains multiple Streamlit applications (`app.py`, `main.py`, `prototype.py`) representing different development stages. To run the most comprehensive version:

```bash
streamlit run app.py
```

The application will open in your default web browser.

## 🕹️ How to Use

1.  **Navigate to the Home page**: Click the "Load Datasets" button to automatically download and load the necessary data from Kaggle.
2.  **Explore the Data**: Go to the "Data Overview" / "Data Analysis" section to view dataset statistics, distributions, and correlations.
3.  **Train Models**:
    -   In the "ML Models" / "Model Training" section, select the models you want to train.
    -   Click the "Train Models" button to start the training process on the numerical dataset.
    -   Similarly, you can train Deep Learning models in their respective sections.
4.  **Analyze Results**:
    -   View detailed performance metrics, comparison charts, and ROC curves in the "Model Comparison" or "Results Dashboard" pages.
    -   Analyze feature importances to understand which factors contribute most to flood risk.
5.  **Make Predictions**:
    -   Go to the "Predictions" page.
    -   Adjust the input sliders for various environmental factors.
    -   Click "Make Prediction" to get a real-time flood risk probability from the trained models.
6.  **Assess Risk**: The "Risk Assessment" module provides a simulated, high-level risk analysis based on selected regional and seasonal parameters, complete with a risk gauge and mitigation recommendations.
