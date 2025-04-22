import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU if needed
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import r2_score, mean_squared_error
import pickle
import argparse

print("TensorFlow running on:", "GPU" if tf.config.list_physical_devices('GPU') else "CPU")

def load_resources(model_path, tokenizer_path):
    """Load the trained model and tokenizer"""
    model = tf.keras.models.load_model(model_path)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

def evaluate_model(model, tokenizer, test_data_path):
    """Evaluate the model on test data"""
    # Load and preprocess test data
    df = pd.read_csv(test_data_path)
    smiles = df['smiles'].values
    p1 = df['P1'].values
    
    # Convert SMILES to sequences
    sequences = tokenizer.texts_to_sequences(smiles)
    max_len = model.input_shape[1]  
    X_test = pad_sequences(sequences, maxlen=max_len, padding='post')
    y_test = p1
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)  
    
    print("\nEvaluation Results:")
    print("="*40)
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print("="*40)
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained LSTM model on SMILES data.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model H5 file')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to tokenizer pickle file')
    parser.add_argument('--test_data_path', type=str, required=True, help='Path to test data CSV file')
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_resources(args.model_path, args.tokenizer_path)
    
    # Evaluate on test data
    evaluate_model(model, tokenizer, args.test_data_path)

if __name__ == "__main__":
    main()