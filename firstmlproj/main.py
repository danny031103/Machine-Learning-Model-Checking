import subprocess

# Generate synthetic data
subprocess.run(['python', 'src/data_generation.py'])

# Train and evaluate the model
subprocess.run(['python', 'src/train_model.py'])