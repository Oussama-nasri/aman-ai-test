#Importations
import pandas as pd
import numpy as np
from feature_extractor import fingerprint_features
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
import joblib
import tensorflow as tf
import argparse

def train_model1(data_path,model_path):
    """
    Trains a RandomForestClassifier on fingerprint features from SMILES strings and saves the best model.

    """
    df_train =pd.read_csv()

    X_train = [fingerprint_features(smi) for smi in df_train["smiles"]]
    X_train = [np.array(fp) for fp in X_train if fp is not None] 
    y_train = df_train["P1"].values[:len(X_train)]  


    model = RandomForestClassifier(random_state=42)
    param_dist = {
        "n_estimators": [100, 500],
        "max_depth": [None, 10],
        "min_samples_split": [2,5],
        "max_features": ["sqrt"],
        "bootstrap": [True],
    }
    search = RandomizedSearchCV(
        model, param_dist, n_iter=15, cv=3, scoring="roc_auc", n_jobs=-1, verbose=1
    )

    search.fit(X_train, y_train)


    best_model = search.best_estimator_
    print(f"Best params: {search.best_params_}")

    #Save model
    joblib.dump(best_model, model_path)




def main():
    parser = argparse.ArgumentParser(description="Train Random Forest model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training CSV data")
    parser.add_argument("--model_path", type=str, required=True, help="Path to save model")
    args = parser.parse_args()

    train_model1(args.data_path,args.model_path)

if __name__ == "__main__":
    main()