import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

from data_generator import generate_sample_dataset

def train_and_evaluate_models(data_path="data/sample_co2_emissions.csv", model_output_path="models/trained_models.pkl"):
    if not os.path.exists(data_path):
        print(f"Data path '{data_path}' not found. Generating sample dataset...")
        df = generate_sample_dataset(output_path=data_path)
    else:
        df = pd.read_csv(data_path)

    # Feature and Target Selection
    features = ['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)', 'Vehicle Class', 'Fuel Type', 'Transmission']
    target = 'CO2 Emissions(g/km)'

    X = df[features]
    y = df[target]

    num_cols = ['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)']
    cat_cols = ['Vehicle Class', 'Fuel Type', 'Transmission']

    # Preprocessor definition
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Models dictionary
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
        'Support Vector Regression (SVR)': SVR(kernel='rbf', C=100, epsilon=0.1),
        'Stochastic Gradient Descent (SGD)': SGDRegressor(max_iter=1000, tol=1e-3, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    results = {}
    pipelines = {}

    print("\n" + "="*60)
    print(f"{'Model Name':<35} | {'R2 Score':<8} | {'RMSE':<8} | {'MAE':<8}")
    print("="*60)

    for name, model in models.items():
        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
        pipe.fit(X_train, y_train)
        
        preds = pipe.predict(X_test)
        
        r2 = r2_score(y_test, preds)
        rmse = root_mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        
        results[name] = {
            'R2': round(r2, 4),
            'RMSE': round(rmse, 2),
            'MAE': round(mae, 2)
        }
        pipelines[name] = pipe
        print(f"{name:<35} | {r2:<8.4f} | {rmse:<8.2f} | {mae:<8.2f}")

    print("="*60 + "\n")

    # Save artifacts
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    artifacts = {
        'pipelines': pipelines,
        'results': results,
        'features': features,
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'unique_classes': sorted(df['Vehicle Class'].unique().tolist()),
        'unique_fuel_types': sorted(df['Fuel Type'].unique().tolist()),
        'unique_transmissions': sorted(df['Transmission'].unique().tolist())
    }

    with open(model_output_path, 'wb') as f:
        pickle.dump(artifacts, f)

    print(f"All models and preprocessors saved to '{model_output_path}'.")
    return artifacts

if __name__ == "__main__":
    train_and_evaluate_models()
