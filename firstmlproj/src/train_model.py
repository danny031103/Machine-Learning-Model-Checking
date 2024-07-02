import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def load_and_prepare_data(data_path):
    print(f"Loading data from: {data_path}")  # Debugging print statement
    synthetic_data = pd.read_csv(data_path)
    
    # Split into features and target
    X = synthetic_data[['Principal (P)', 'Annual Interest Rate (r)', 'Compounded per Year (n)', 'Time (t)']]  # Features
    y = synthetic_data['Amount (A)']  # Target
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    
    # Determine the absolute path to the project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))  # This gets the directory of the current script
    results_dir = os.path.join(project_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)  # Ensure the 'results' directory exists
    
    # Save evaluation results
    with open(os.path.join(results_dir, 'model_evaluation.txt'), 'w') as f:
        f.write(f'Mean Squared Error: {mse}\n')
        f.write(f'Predicted values: {y_pred}\n')  # Optionally write predicted values
    
    print(f'Model evaluation results saved to {os.path.join(results_dir, "model_evaluation.txt")}')

if __name__ == "__main__":
    # Get the absolute path to the current script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '/Users/daniel.brito/Desktop/firstmlproj/data/processed/synthetic_data.csv')
    
    X_train, X_test, y_train, y_test = load_and_prepare_data(data_path)
    train_and_evaluate_model(X_train, X_test, y_train, y_test)
