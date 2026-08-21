# 🌱 Curbing Carbon on the Road: Vehicle CO₂ Emission Detection & ML Benchmark

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **Academic Project Report & Interactive Machine Learning System**  
> **Institution:** KIIT Deemed to be University, School of Computer Engineering  
> **Department:** Information Technology  
> **Guided by:** Prof. Himanshu Das  
> **Authors:** Krishanu Roy, Souvik Basak, Suvankar Panigrahi, Tangudu Vijay Sankar, Ayush Kumar Rana  

---

## 📌 Executive Summary

Transportation is one of the leading contributors to global greenhouse gas emissions. Managing heavy vehicle logistics and reducing carbon footprints requires precise predictive modeling of **CO₂ emissions ($\text{g/km}$)** and **fuel efficiency ($\text{L/100 km}$)** based on physical vehicle characteristics.

This repository features an end-to-end Machine Learning pipeline and an interactive **Streamlit Web Application** that benchmarks **5 machine learning algorithms** trained on vehicle specification datasets:
1. **Support Vector Regression (SVR)**
2. **Linear Regression**
3. **Stochastic Gradient Descent (SGD) Regressor**
4. **Random Forest Regressor**
5. **Decision Tree Regressor (CART/ID3)**

---

## 🏆 Model Performance Benchmark Leaderboard

All models were trained on 80% train split and evaluated on 20% unseen test split:

| Algorithm | $R^2$ Score | RMSE ($\text{g/km}$) | MAE ($\text{g/km}$) | Key Strength |
| :--- | :---: | :---: | :---: | :--- |
| **Support Vector Regression (SVR)** | **0.9882** | **7.95** | 5.75 | Non-linear RBF kernel mapping |
| **Linear Regression** | **0.9797** | 10.41 | 7.54 | Fast, interpretable linear baseline |
| **Stochastic Gradient Descent (SGD)** | **0.9794** | 10.48 | 7.91 | Scalable iterative gradient optimization |
| **Random Forest Regressor** | **0.9790** | 10.57 | **5.66** | Robust ensemble decision tree voting |
| **Decision Tree Regressor** | **0.9785** | 10.70 | 5.86 | Clear hierarchical split rules |

---

## 🚀 Interactive Streamlit Showcase Features

The interactive dashboard (`app.py`) includes:
- **📋 Executive Summary & KPI Cards**: Quick overview of dataset metrics, top emitter brands, and lead model score.
- **📊 Exploratory Data Analysis (EDA)**:
  - Vehicle counts by Manufacturer & Vehicle Class.
  - Distribution of Engine Sizes (L) and Cylinder counts.
  - CO₂ emission comparison by Fuel Type (Regular, Premium, Ethanol E85, Diesel, Natural Gas).
  - Feature correlation heatmap.
- **🔮 Live CO₂ Emission Predictor**:
  - Interactive slider & dropdown controls for vehicle parameters.
  - Instant CO₂ emission prediction ($\text{g/km}$), annual footprint calculation ($\text{kg/year}$), and **Eco-Rating Badges** (Low, Moderate, High).
- **🏆 Model Leaderboard**: Interactive performance comparison charts.
- **📁 Dataset Viewer**: Interactive dataframe with CSV download support.

---

## 📁 Repository Structure

```
Carbon Emission Detection/
├── data/
│   └── sample_co2_emissions.csv     # Sample vehicle emissions dataset (1,200 records)
├── models/
│   └── trained_models.pkl            # Serialized ML pipelines & scaling objects
├── src/
│   ├── data_generator.py             # Synthetic realistic vehicle data generator
│   └── train_models.py               # Multi-model training & evaluation pipeline
├── app.py                            # Streamlit Web Showcase Application
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## 🛠️ Quickstart Guide

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/Krishanu-Roy/python.git
cd "python"
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Data & Train ML Models
```bash
python src/data_generator.py
python src/train_models.py
```

### 4. Launch Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🌐 Free Live Hosting Instructions

You can host this project live for free to share with recruiters:

### Option A: Streamlit Community Cloud (Recommended)
1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Connect your GitHub account and select repository `python`.
4. Set main file path to `app.py` and click **Deploy**!

### Option B: Hugging Face Spaces
1. Create a new Space on [Hugging Face](https://huggingface.co/spaces) selecting **Streamlit** SDK.
2. Upload `app.py`, `requirements.txt`, `src/`, `data/`, `models/`.
3. Your app will automatically build and launch!

---

## 📜 References & Acknowledgments
- Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (A. Géron)
- KIIT Deemed to be University, School of Computer Engineering
- Canada Open Data - Vehicle Fuel Consumption Ratings
