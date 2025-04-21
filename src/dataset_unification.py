from feature_extractor import fingerprint_features
import pandas as pd






def main():
    df1 = pd.read_csv("/home/oussama/test_technique/data/dataset_single.csv")
    df2 = pd.read_csv("/home/oussama/test_technique/data/dataset_multi.csv")
    new_df = pd.DataFrame(columns=['mol_id','smiles','feature_extracted', 'P1', 'P2',
                                'P3','P4','P5','P6','P7','P8','P9'])



    new_df['mol_id'] = df1['mol_id']
    new_df['smiles'] = df1['smiles']
    new_df['feature_extracted'] = df1['smiles'].apply(fingerprint_features)


    properties = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9']
    new_df[properties] = df2[properties]

    return new_df



if __name__ == "__main__":
    result_df = main()
    result_df.to_csv('/home/oussama/test_technique/data/combined_dataset.csv', index=False)