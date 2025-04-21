from feature_extractor import fingerprint_features
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from feature_extractor import fingerprint_features




def main():
    df = pd.read_csv("/home/oussama/test_technique/data/balanced_dataset_single.csv")
    new_df = pd.DataFrame(columns=['mol_id','smiles','feature_extracted', 'P1'])



    new_df['mol_id'] = df['mol_id']
    new_df['smiles'] = df['smiles']
    new_df['P1'] = df['P1']
    new_df['feature_extracted'] = df['smiles'].apply(fingerprint_features)
    return new_df



if __name__ == "__main__":
    result_df = main()
    result_df.to_csv('/home/oussama/test_technique/data/dataset_single_fe.csv', index=False)