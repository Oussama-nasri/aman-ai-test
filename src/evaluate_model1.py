import joblib
import numpy as np
import pandas as pd
from feature_extractor import fingerprint_features
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_recall_curve,
    f1_score,
    confusion_matrix,
    classification_report,
)
import argparse



def evalute_random_forest(test_dataset_path,model_path):
    """
    Evaluates a trained RandomForest model on a test dataset.
    """
    df_test = pd.read_csv(test_dataset_path)

    X_test = [fingerprint_features(smi) for smi in df_test["smiles"]]
    X_test = [np.array(fp) for fp in X_test if fp is not None] 
    y_test = df_test["P1"].values[:len(X_test)] 

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for AUC

    # Metrics
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.2f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.2f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    conf_mat = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", conf_mat)


    # fold CV on best model
    cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring="roc_auc")
    print(f"Cross-validated AUC: {np.mean(cv_scores):.2f} (±{np.std(cv_scores):.2f})")



def main():
    parser = argparse.ArgumentParser(description="Evaluate Random Forest model.")
    parser.add_argument("--test_dataset_path", type=str, required=True, help="Path to test dataset CSV file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model.joblib")
    args = parser.parse_args()

    evalute_random_forest(args.test_dataset_path, args.model_path)

if __name__ == "__main__":
    main()


    