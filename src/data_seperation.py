import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from feature_extractor import fingerprint_features

file_path = """/home/oussama/test_technique/data/balanced_dataset_single.csv"""

# Load the dataset
df = pd.read_csv(file_path)

# Features and labels
X = np.stack(df['smiles'].values)
y = df['P1'].values

# Split into train (80%) and test (20%)
X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
    X, y, df, test_size=0.20, random_state=42, stratify=y
)

# Save as CSV
df_train.to_csv("/home/oussama/test_technique/data/model1/train_data.csv", index=False)
df_test.to_csv("/home/oussama/test_technique/data/model1/test_data.csv", index=False)