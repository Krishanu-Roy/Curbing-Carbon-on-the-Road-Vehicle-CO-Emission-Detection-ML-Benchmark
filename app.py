import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Carbon Emission Detection ML Showcase",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 20px;
    }
    .kpi-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #10B981;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .kpi-title {
        font-size: 0.85rem;
        color: #6B7280;
        text-transform: uppercase;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 1.8rem;
        color: #111827;
        font-weight: 700;
    }
    .badge-eco-green {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-eco-yellow {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-eco-red {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Load data and trained models
@st.cache_data
def load_data():
    data_path = "data/sample_co2_emissions.csv"
    if not os.path.exists(data_path):
        from src.data_generator import generate_sample_dataset
        return generate_sample_dataset(output_path=data_path)
    return pd.read_csv(data_path)

@st.cache_resource
def load_artifacts():
    model_path = "models/trained_models.pkl"
    if not os.path.exists(model_path):
        from src.train_models import train_and_evaluate_models
        return train_and_evaluate_models(model_output_path=model_path)
    with open(model_path, 'rb') as f:
        return pickle.load(f)

df = load_data()
artifacts = load_artifacts()

pipelines = artifacts['pipelines']
results = artifacts['results']

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/co2.png", width=70)
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Go to section:",
    [
        "📋 Executive Summary",
        "📊 Data Visualizations (EDA)",
        "🔮 Live Emission Predictor",
        "🏆 Model Leaderboard",
        "📁 Dataset & Batch Test"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎓 Academic Project Context")
st.sidebar.info(
    "**Institution:** KIIT Deemed to be University  \n"
    "**Department:** Information Technology  \n"
    "**Guide:** Prof. Himanshu Das  \n"
    "**Authors:** Krishanu Roy, Souvik Basak, Suvankar Panigrahi, Tangudu Vijay Sankar, Ayush Kumar Rana"
)

# SECTION 1: EXECUTIVE SUMMARY
if menu == "📋 Executive Summary":
    st.markdown('<p class="main-title">Curbing Carbon on the Road</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Machine Learning as a Tool for Greener Heavy Vehicle Operations</p>', unsafe_allow_html=True)

    # Top KPI summary row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="kpi-card"><div class="kpi-title">Dataset Size</div><div class="kpi-value">{:,}</div></div>'.format(len(df)), unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="kpi-card"><div class="kpi-title">Avg CO2 Emission</div><div class="kpi-value">{:.1f} g/km</div></div>'.format(df['CO2 Emissions(g/km)'].mean()), unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="kpi-card"><div class="kpi-title">Best Model R² Score</div><div class="kpi-value">{:.4f}</div></div>'.format(max(r['R2'] for r in results.values())), unsafe_allow_html=True)
    with c4:
        best_model_name = max(results, key=lambda k: results[k]['R2'])
        st.markdown('<div class="kpi-card"><div class="kpi-title">Top ML Model</div><div class="kpi-value" style="font-size:1.3rem;">{}</div></div>'.format(best_model_name), unsafe_allow_html=True)

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("📌 Project Motivation & Abstract")
        st.write("""
        Transportation is a major global contributor to greenhouse gas emissions. Regulating and predicting vehicle emissions—especially 
        heavy vehicle fleets—is an urgent imperative for sustainable logistics and green transportation policies.

        This project evaluates and benchmarks five supervised machine learning algorithms to accurately estimate vehicle **CO2 emissions (g/km)** 
        and **fuel consumption (L/100 km)** based on engine capacity, cylinder count, vehicle class, fuel type, and transmission specifications.
        """)

        st.subheader("🎯 Key Objectives")
        st.markdown("""
        1. **Model Development**: Implement 5 core algorithms: *Linear Regression, Decision Trees (CART/ID3), Support Vector Regression (SVR), Stochastic Gradient Descent (SGD), and Random Forest*.
        2. **Accuracy Assessment**: Evaluate predictions using $R^2$ Score, Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).
        3. **Optimization & Insights**: Identify dominant drivers of vehicle carbon footprint to enable eco-routing and fleet optimization strategies.
        4. **Interactive Showcase**: Provide stakeholders with data-driven decision tools for fleet compliance and environmental benchmarking.
        """)

    with col_right:
        st.subheader("⚙️ Technical Architecture")
        st.markdown("""
        - **Language**: Python 3.12
        - **ML Library**: `scikit-learn`
        - **Data Processing**: `pandas`, `numpy`
        - **Data Visualization**: `matplotlib`, `seaborn`
        - **UI Framework**: `streamlit`
        """)
        st.success("✅ Models and preprocessors trained and serialized in `models/trained_models.pkl`")

        # Quick preview graph
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.histplot(df['CO2 Emissions(g/km)'], kde=True, color='#10B981', ax=ax)
        ax.set_title("CO2 Emissions Distribution (g/km)", fontsize=10)
        st.pyplot(fig)

# SECTION 2: DATA VISUALIZATIONS (EDA)
elif menu == "📊 Data Visualizations (EDA)":
    st.markdown('<p class="main-title">Exploratory Data Analysis (EDA)</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Visualizing relationships between engine specs, fuel consumption, and carbon output</p>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["🏎️ Brands & Classes", "⚡ Engine Specs", "🔥 Fuel & Emissions", "🧮 Correlations"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Vehicle Count by Manufacturer (Make)")
            make_counts = df['Make'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=make_counts.index, y=make_counts.values, palette="crest", ax=ax)
            plt.xticks(rotation=60, ha='right')
            ax.set_ylabel("Number of Cars")
            st.pyplot(fig)
        
        with c2:
            st.subheader("Vehicle Count by Class")
            class_counts = df['Vehicle Class'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis", ax=ax)
            plt.xticks(rotation=60, ha='right')
            ax.set_ylabel("Number of Cars")
            st.pyplot(fig)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Distribution by Engine Size (L)")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df['Engine Size(L)'], bins=20, kde=True, color="#3B82F6", ax=ax)
            ax.set_xlabel("Engine Size (Liters)")
            st.pyplot(fig)

        with c2:
            st.subheader("Distribution by Cylinder Count")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.countplot(x='Cylinders', data=df, palette="magma", ax=ax)
            ax.set_ylabel("Number of Vehicles")
            st.pyplot(fig)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Average CO2 Emissions by Car Brand (g/km)")
            brand_co2 = df.groupby('Make')['CO2 Emissions(g/km)'].mean().sort_values()
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x=brand_co2.values, y=brand_co2.index, palette="mako", ax=ax)
            ax.set_xlabel("Mean CO2 Emissions (g/km)")
            st.pyplot(fig)

        with c2:
            st.subheader("Fuel Type Breakdown vs CO2 Emissions")
            fuel_map = {'X': 'Regular Gasoline', 'Z': 'Premium Gasoline', 'E': 'Ethanol (E85)', 'D': 'Diesel', 'N': 'Natural Gas'}
            df_fuel = df.copy()
            df_fuel['Fuel_Name'] = df_fuel['Fuel Type'].map(fuel_map)
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(x='Fuel_Name', y='CO2 Emissions(g/km)', data=df_fuel, palette="Set2", ax=ax)
            plt.xticks(rotation=20)
            st.pyplot(fig)

    with tab4:
        st.subheader("Feature Correlation Heatmap")
        num_df = df[['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)', 'CO2 Emissions(g/km)']]
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.heatmap(num_df.corr(), annot=True, cmap="YlGnBu", fmt=".2f", ax=ax, linewidths=0.5)
        st.pyplot(fig)

# SECTION 3: LIVE EMISSION PREDICTOR
elif menu == "🔮 Live Emission Predictor":
    st.markdown('<p class="main-title">Interactive Vehicle CO2 Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Enter custom vehicle parameters to predict carbon output and eco-efficiency badge</p>', unsafe_allow_html=True)

    col_input, col_output = st.columns([1, 1])

    with col_input:
        st.subheader("🎛️ Input Vehicle Parameters")
        
        selected_model_name = st.selectbox(
            "Select Machine Learning Algorithm:",
            options=list(pipelines.keys()),
            index=4 # Default to Random Forest
        )

        v_class = st.selectbox("Vehicle Class:", artifacts['unique_classes'], index=0)
        engine_size = st.slider("Engine Size (Liters):", min_value=0.9, max_value=8.4, value=2.5, step=0.1)
        cylinders = st.selectbox("Number of Cylinders:", [3, 4, 5, 6, 8, 10, 12, 16], index=1)
        
        fuel_type_map = {
            'X': 'Regular Gasoline (X)',
            'Z': 'Premium Gasoline (Z)',
            'E': 'Ethanol E85 (E)',
            'D': 'Diesel (D)',
            'N': 'Natural Gas (N)'
        }
        fuel_type_display = st.selectbox("Fuel Type:", list(fuel_type_map.values()), index=0)
        fuel_type = [k for k, v in fuel_type_map.items() if v == fuel_type_display][0]

        transmission = st.selectbox("Transmission:", artifacts['unique_transmissions'], index=0)

        # Estimate fuel consumption recommendation
        est_comb = round(3.5 + (engine_size * 1.8) + (cylinders * 0.45), 1)
        fuel_comb = st.number_input("Fuel Consumption Comb (L/100 km):", min_value=3.0, max_value=35.0, value=est_comb, step=0.1)

    with col_output:
        st.subheader("📊 Prediction Results")

        input_df = pd.DataFrame([{
            'Engine Size(L)': engine_size,
            'Cylinders': cylinders,
            'Fuel Consumption Comb (L/100 km)': fuel_comb,
            'Vehicle Class': v_class,
            'Fuel Type': fuel_type,
            'Transmission': transmission
        }])

        model_pipeline = pipelines[selected_model_name]
        predicted_co2 = model_pipeline.predict(input_df)[0]
        annual_co2_kg = (predicted_co2 * 15000) / 1000.0 # Based on 15,000 km annual drive

        st.markdown(f"### Predicted CO2 Emission: **{predicted_co2:.1f} g/km**")
        st.progress(min(1.0, max(0.0, float(predicted_co2) / 500.0)))

        # Eco Badge logic
        if predicted_co2 < 180:
            badge_html = '<span class="badge-eco-green">🌱 LOW EMISSIONS (Eco-Friendly)</span>'
            rating_text = "Extremely clean profile matching modern low-emission / compact standards."
        elif predicted_co2 < 280:
            badge_html = '<span class="badge-eco-yellow">⚠️ MODERATE EMISSIONS (Standard)</span>'
            rating_text = "Typical mid-size vehicle emission profile."
        else:
            badge_html = '<span class="badge-eco-red">🚨 HIGH EMISSIONS (Heavy Output)</span>'
            rating_text = "High carbon output profile associated with heavy engines or sports trucks."

        st.markdown(badge_html, unsafe_allow_html=True)
        st.caption(rating_text)

        st.markdown("---")
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Estimated Annual CO2", f"{annual_co2_kg:.1f} kg / year", help="Assuming 15,000 km driven annually")
        with m2:
            st.metric("Algorithm Used", selected_model_name, f"R² = {results[selected_model_name]['R2']}")

# SECTION 4: MODEL LEADERBOARD
elif menu == "🏆 Model Leaderboard":
    st.markdown('<p class="main-title">Machine Learning Model Benchmark</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Comparing 5 Machine Learning algorithms evaluated on test split</p>', unsafe_allow_html=True)

    results_df = pd.DataFrame(results).T.sort_values(by='R2', ascending=False)
    st.dataframe(results_df.style.highlight_max(axis=0, subset=['R2'], color='#D1FAE5'), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("R² Score Comparison (Higher is Better)")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(x=results_df['R2'], y=results_df.index, palette="crest", ax=ax)
        ax.set_xlim(0.9, 1.0)
        st.pyplot(fig)

    with c2:
        st.subheader("Root Mean Squared Error (Lower is Better)")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(x=results_df['RMSE'], y=results_df.index, palette="flare", ax=ax)
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("🧠 Algorithm Insights from College Project Report")
    st.markdown("""
    - **Random Forest & Support Vector Regression (SVR)** achieved top performance ($R^2 > 0.97$) due to their non-linear mapping capabilities and kernel functions.
    - **Linear Regression & SGD** provided baseline linear benchmarks, highlighting strong direct scaling between fuel consumption volume and $\text{CO}_2$ emissions.
    - **Decision Trees (CART/ID3)** offered readable decision rules and fast training times.
    """)

# SECTION 5: DATASET VIEWER
elif menu == "📁 Dataset & Batch Test":
    st.markdown('<p class="main-title">Dataset Explorer & Batch Predictions</p>', unsafe_allow_html=True)
    
    st.dataframe(df, use_container_width=True)

    st.download_button(
        label="📥 Download Sample Dataset CSV",
        data=df.to_csv(index=False),
        file_name="sample_co2_emissions.csv",
        mime="text/csv"
    )
