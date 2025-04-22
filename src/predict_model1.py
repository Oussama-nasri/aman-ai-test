import joblib
import numpy as np
from src.feature_extractor import fingerprint_features
import tensorflow as tf
#from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
from tensorflow.keras.models import load_model
import pickle
import json
import numpy as np
import tensorflow as tf
import argparse


def predict_model1(smiles_string,model_path):

    model = joblib.load(model_path)
    
    # Convert SMILES to fingerprint
    fp = fingerprint_features(smiles_string)
    fp_array = np.array([np.array(fp)])  # Reshape for prediction
    
    # Make prediction
    prediction = model.predict(fp_array)
    probability = model.predict_proba(fp_array)[0] if hasattr(model, "predict_proba") else None
    
    return prediction[0], probability


def main():
    parser = argparse.ArgumentParser(description="Predict using trained model1.")
    parser.add_argument("--smiles", type=str, required=True, help="SMILES string to predict")
    parser.add_argument("--model_path", type=str, required=True, help="model path")
    args = parser.parse_args()

    prediction, probability = predict_model1(args.smiles,args.model_path)
    print("Prediction:", prediction)
    print("Probability:", probability)

if __name__ == "__main__":
    main()


    