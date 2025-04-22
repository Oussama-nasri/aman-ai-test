
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import joblib
import numpy as np
from src.feature_extractor import fingerprint_features
import tensorflow as tf
print("TensorFlow running on:", "GPU" if tf.config.list_physical_devices('GPU') else "CPU")
#from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import pickle
import json
import numpy as np

import argparse




def predict_model2(smiles_string,model_path,tokenizer_path):
    """
    Predicts binary outcome from a SMILES string using a pre-trained Transformer model.

    """

    tf.keras.backend.clear_session()


    # Load tokenizer
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tokenizer = tokenizer_from_json(f.read())

    # Load model
    model = tf.keras.models.load_model(model_path)

    # Preprocess SMILES
    seq = tokenizer.texts_to_sequences([smiles_string])
    padded = pad_sequences(seq, maxlen=model.input_shape[1], padding='post')

    # Predict
    probability = model.predict(padded)[0][0]
    prediction = int(probability >= 0.5)
    confidence = probability if prediction == 1 else 1 - probability

    return prediction, confidence



def main():
    parser = argparse.ArgumentParser(description="Predict using trained model2.")
    parser.add_argument("--smiles", type=str, required=True, help="SMILES string to predict")
    parser.add_argument("--model_path", type=str, required=True, help="load model path")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="load tokenizer location")
    args = parser.parse_args()

    prediction, probability = predict_model2(args.smiles,args.model_path,args.tokenizer_path)
    print("Prediction:", prediction)
    print("Probability:", probability)

if __name__ == "__main__":
    main()