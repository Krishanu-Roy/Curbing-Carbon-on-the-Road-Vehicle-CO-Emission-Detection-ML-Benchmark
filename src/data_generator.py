import pandas as pd
import numpy as np
import os

def generate_sample_dataset(num_samples=1200, output_path="data/sample_co2_emissions.csv"):
    np.random.seed(42)

    makes_weights = {
        'Ford': 0.15, 'Chevrolet': 0.12, 'BMW': 0.10, 'Mercedes-Benz': 0.08, 
        'Toyota': 0.08, 'Audi': 0.06, 'Nissan': 0.06, 'Honda': 0.05,
        'Volkswagen': 0.05, 'Hyundai': 0.04, 'Jeep': 0.04, 'GMC': 0.03,
        'Dodge': 0.03, 'Lexus': 0.03, 'Subaru': 0.02, 'Porsche': 0.02,
        'Ram': 0.02, 'Bugatti': 0.005, 'Smart': 0.015
    }
    makes = list(makes_weights.keys())
    weights = np.array(list(makes_weights.values()))
    weights = weights / weights.sum()

    make_list = np.random.choice(makes, size=num_samples, p=weights)

    vehicle_classes = [
        'SUV - SMALL', 'MID-SIZE', 'COMPACT', 'SUV - STANDARD', 
        'SUBCOMPACT', 'FULL-SIZE', 'PICKUP TRUCK - STANDARD', 
        'TWO-SEATER', 'MINICOMPACT', 'STATION WAGON - SMALL', 
        'PICKUP TRUCK - SMALL', 'VAN - PASSENGER', 'MINIVAN', 'VAN - CARGO'
    ]
    
    fuel_types = ['X', 'Z', 'E', 'D', 'N'] # X: Regular, Z: Premium, E: E85, D: Diesel, N: Natural Gas
    fuel_weights = [0.45, 0.40, 0.10, 0.04, 0.01]
    
    transmissions = ['AS6', 'A8', 'M6', 'AS8', 'AM7', 'AV', 'A6', 'AS10', 'A9', 'M5']

    data = []

    for i in range(num_samples):
        mk = make_list[i]
        
        if mk in ['Bugatti', 'Lamborghini', 'Ferrari', 'Rolls-Royce', 'Aston Martin']:
            engine_size = np.round(np.random.choice([6.0, 6.2, 6.5, 8.0]), 1)
            cylinders = 12 if engine_size < 8.0 else 16
            fuel_type = 'Z'
            v_class = np.random.choice(['TWO-SEATER', 'COMPACT'])
        elif mk in ['Ford', 'GMC', 'Chevrolet', 'Ram'] and np.random.rand() > 0.4:
            engine_size = np.round(np.random.choice([2.7, 3.3, 3.5, 5.0, 5.3, 6.2]), 1)
            cylinders = 6 if engine_size <= 3.5 else 8
            v_class = np.random.choice(['PICKUP TRUCK - STANDARD', 'SUV - STANDARD', 'VAN - PASSENGER'])
            fuel_type = np.random.choice(['X', 'E', 'D'], p=[0.7, 0.2, 0.1])
        elif mk == 'Smart':
            engine_size = 1.0
            cylinders = 3
            v_class = 'TWO-SEATER'
            fuel_type = 'X'
        else:
            engine_size = np.round(np.random.choice([1.5, 1.8, 2.0, 2.4, 2.5, 3.0, 3.5, 3.6, 4.0, 5.0]), 1)
            if engine_size <= 2.0:
                cylinders = np.random.choice([3, 4], p=[0.1, 0.9])
            elif engine_size <= 2.5:
                cylinders = 4
            elif engine_size <= 3.6:
                cylinders = 6
            else:
                cylinders = 8
            v_class = np.random.choice(vehicle_classes)
            fuel_type = np.random.choice(fuel_types, p=fuel_weights)

        transmission = np.random.choice(transmissions)
        
        base_comb = 3.5 + (engine_size * 1.8) + (cylinders * 0.45)
        
        if fuel_type == 'E':
            base_comb *= 1.35
        elif fuel_type == 'D':
            base_comb *= 0.85
        elif fuel_type == 'Z':
            base_comb *= 1.05

        fuel_comb_l100 = np.round(np.random.normal(base_comb, 0.6), 1)
        fuel_comb_l100 = max(4.0, fuel_comb_l100)

        fuel_city_l100 = np.round(fuel_comb_l100 * np.random.uniform(1.10, 1.25), 1)
        fuel_hwy_l100 = np.round((3 * fuel_comb_l100 - 2 * fuel_city_l100), 1)
        fuel_hwy_l100 = max(3.5, fuel_hwy_l100)

        fuel_comb_mpg = int(np.round(282.48 / fuel_comb_l100))

        if fuel_type == 'E':
            co2_factor = 16.5
        elif fuel_type == 'D':
            co2_factor = 26.8
        elif fuel_type == 'N':
            co2_factor = 18.0
        else:
            co2_factor = 23.5

        co2_emissions = int(np.round(fuel_comb_l100 * co2_factor + np.random.normal(0, 4)))
        co2_emissions = max(90, co2_emissions)

        model_name = f"{mk} {v_class.split()[0]} {engine_size}L"

        data.append({
            'Make': mk,
            'Model': model_name,
            'Vehicle Class': v_class,
            'Engine Size(L)': engine_size,
            'Cylinders': int(cylinders),
            'Transmission': transmission,
            'Fuel Type': fuel_type,
            'Fuel Consumption City (L/100 km)': fuel_city_l100,
            'Fuel Consumption Hwy (L/100 km)': fuel_hwy_l100,
            'Fuel Consumption Comb (L/100 km)': fuel_comb_l100,
            'Fuel Consumption Comb (mpg)': fuel_comb_mpg,
            'CO2 Emissions(g/km)': co2_emissions
        })

    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Sample dataset successfully generated with {len(df)} rows at '{output_path}'.")
    return df

if __name__ == "__main__":
    generate_sample_dataset()
