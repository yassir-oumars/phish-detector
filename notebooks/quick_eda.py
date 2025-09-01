# quick_eda.py

# Step 1: Install dependencies (run this in terminal, not in Python)
# pip install pandas matplotlib scipy

import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import arff

# ==== CONFIGURATION ====
# If you have a CSV, set this:
csv_path = "../data/phishing_data.csv"  # Change this to your CSV file path
use_arff = False  # Set to True if using ARFF dataset

# ==== Step 2: Load data ====
if use_arff:
    data, meta = arff.loadarff("../data/phishing_data.arff")
    df = pd.DataFrame(data)
else:
    df = pd.read_csv(csv_path)

print("Data loaded successfully!")
print("Shape of dataset (rows, columns):", df.shape)

# ==== Step 3: Quick Checks ====
# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# If the dataset has a label/target column
target_col = "label"  # Change this to your actual label column name

if target_col in df.columns:
    # Convert text labels to 0 and 1 if needed
    df[target_col] = df[target_col].map({"legit": 0, "phishing": 1}).fillna(df[target_col])
    
    print("\nTarget value counts (0=legit, 1=phishing):")
    print(df[target_col].value_counts())

# ==== Step 4: Quick Visualizations ====
# Simple histograms for numeric columns
df.hist(figsize=(10, 8))
plt.suptitle("Feature Distributions")
plt.show()
