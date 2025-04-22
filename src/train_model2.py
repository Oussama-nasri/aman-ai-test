
#Importations
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GlobalMaxPooling1D, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pickle
import argparse


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print("TensorFlow running on:", "GPU" if tf.config.list_physical_devices('GPU') else "CPU")

def train_model2(data_path):
    """
    Trains an LSTM model to predict 'P1' from SMILES strings.
    - Returns trained model, tokenizer, and input sequence length
    """

    # Load and preprocess data
    df = pd.read_csv(data_path)
    smiles = df['smiles'].values
    p1 = df['P1'].values

    # Tokenize smile
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(smiles)
    vocab_size = len(tokenizer.word_index) + 1

    # Convert to sequences
    sequences = tokenizer.texts_to_sequences(smiles)
    max_len = max(len(s) for s in sequences)
    X = pad_sequences(sequences, maxlen=max_len, padding='post')

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, p1, test_size=0.2, random_state=42)

    # Build LSTM model
    #Input layer : 128 - Hidden layer : 64 - Output layer : 1
    inputs = Input(shape=(max_len,))
    x = Embedding(vocab_size, 128, input_length=max_len)(inputs)
    x = LSTM(64, return_sequences=True)(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    # Train
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)


    return model, tokenizer, max_len


def predict_p1(model, tokenizer, max_len, smiles_string):
    """
    Predicts the 'P1' value for a given SMILES string using the trained model.
    
    - Tokenizes and pads input string
    - Returns model prediction
    """
    seq = tokenizer.texts_to_sequences([smiles_string])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    return model.predict(padded)[0][0]




def main():
    parser = argparse.ArgumentParser(description="Train an LSTM model on SMILES data and optionally predict P1 for a given SMILES string.")
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data CSV file')
    parser.add_argument('--save_model', type=str, default="smiles_transformer.h5", help='Path to save the trained model')
    parser.add_argument('--save_tokenizer', type=str, default="tokenizer.pkl", help='Path to save the tokenizer pickle file')

    args = parser.parse_args()

    model, tokenizer, max_len = train_model2(args.data_path)
    model.save(args.save_model)

    with open(args.save_tokenizer, "wb") as f:
        pickle.dump(tokenizer, f)

    if args.predict:
        predicted_p1 = predict_p1(model, tokenizer, max_len, args.predict)
        print(f"Predicted P1 for SMILES '{args.predict}': {predicted_p1:.4f}")

