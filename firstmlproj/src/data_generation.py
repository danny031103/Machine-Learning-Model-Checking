import os
import numpy as np
import pandas as pd

def generate_synthetic_data(seed=0):
    np.random.seed(seed)
    
    P = np.random.uniform(1000, 10000, 100)
    r = np.random.uniform(0.01, 0.15, 100)
    n = np.random.randint(1, 12, 100)
    t = np.random.uniform(1, 20, 100)
    
    A = P * (1 + r / n) ** (n * t)
    
    synthetic_data = pd.DataFrame({
        'Principal (P)': P,
        'Annual Interest Rate (r)': r,
        'Compounded per Year (n)': n,
        'Time (t)': t,
        'Amount (A)': A
    })
    
    print("Generated Synthetic Data:")
    print(synthetic_data.head())
    
    processed_dir = '../data/processed'
    os.makedirs(processed_dir, exist_ok=True)
    csv_path ="/Users/daniel.brito/Desktop/firstmlproj/data/processed/synthetic_data.csv"
    synthetic_data.to_csv(csv_path, index=False)
    
    print(f"Synthetic data saved to '{csv_path}'")

    return synthetic_data

if __name__ == "__main__":
    generate_synthetic_data()
