import pandas as pd
import numpy as np
import os

# Create data/raw directory if it doesn't exist
os.makedirs('data/raw', exist_ok=True)

# Generate mock train_transaction.csv
np.random.seed(42)
num_samples = 1000
transactions = pd.DataFrame({
    'TransactionID': range(1, num_samples + 1),
    'isFraud': np.random.choice([0, 1], size=num_samples, p=[0.9, 0.1]),
    'TransactionAmt': np.random.uniform(10, 1000, size=num_samples),
    'ProductCD': np.random.choice(['W', 'H', 'C', 'S', 'R'], size=num_samples),
    'card1': np.random.randint(1000, 20000, size=num_samples),
    'card4': np.random.choice(['visa', 'mastercard', 'discover', 'american express'], size=num_samples),
    'V1': np.random.normal(0, 1, size=num_samples),
    'V2': np.random.normal(0, 1, size=num_samples)
})
transactions.to_csv('data/raw/train_transaction.csv', index=False)

# Generate mock train_identity.csv (sparse join)
identity = pd.DataFrame({
    'TransactionID': range(1, int(num_samples * 0.8) + 1),
    'id_01': np.random.normal(-5, 5, size=int(num_samples * 0.8)),
    'id_12': np.random.choice(['Found', 'NotFound'], size=int(num_samples * 0.8)),
    'DeviceType': np.random.choice(['desktop', 'mobile'], size=int(num_samples * 0.8))
})
identity.to_csv('data/raw/train_identity.csv', index=False)
print("Mock datasets generated successfully in data/raw/")
