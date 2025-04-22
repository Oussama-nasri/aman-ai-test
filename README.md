
```markdown
# Project Name
Drug Molecule Property Prediction :
This project implements deep learning models to predict molecular properties from SMILES strings, aiming to assist in drug design by optimizing molecule selection.

## 🚀 Features

- Dual-Model Architecture
- Production-Ready Deployment


## Models

Model1: Feature-Based Prediction
Input: Extracted molecular features (using RDKit fingerprints)

Architecture: Deep neural network

Usage:
aman-predict_model2=src.predict_model1:main

Model2: SMILES String Prediction
Input: Raw SMILES string (character-level processing)

Architecture: LSTM-based network


## 🛠️ Tech Stack

- Language: Python
- Framework: Flask, TensorFLow, RDKit
- Other tools: Docker, 

## 📦 Installation

Clone the repo:

```bash
git clone https://github.com/Oussama-nasri/aman-ai-test.git
cd aman-ai-test

```

Install dependencies:

```bash
conda create -n aman python=3.6
conda activate aman
conda install -c conda-forge rdkit

pip install -r src/requirements.txt
pip install -e .
```

## 🧪 Usage

Train models localy

```bash
aman-train_model1=src.train_model1:main --data_path

aman-train_model2=src.train_model1:main --data_path
```

Predict P1 with models localy

```bash
aman-predict_model1= --smiles
aman-predict_model2= --smiles
```

Evaluate Models
```bash
aman-predict_model1=src.predict_model1:main --test_dataset_path= --model_path=
aman-predict_model1=src.predict_model2:main --test_dataset_path= --model_path=  --tokenizer_path=
```


Running API
```bash
python src/app.py
```
Visit `http://127.0.0.1:5000/docs/#/default/post_predict` in your browser.

Fill the model with a valid molecule smile and the id of the model you want to use:

{
  "smiles": "COC(=O)c1ccc(Oc2nc(NC(C)C)nc(SC)n2)cc1",
  "model": 2
}

Docker Depolyment
```bash
docker build -t aman -f src/Dockerfile .
docker run -p 5000:5000 aman
```


## Project Structure
```bash

TEST_TECHNIQUE/
├── aman.egg-info/          # Python package metadata
├── api/                    # API implementation
│   ├── __pycache__/
│   ├── __init__.py
│   └── routes.py           # API endpoint definitions
├── data/                   # Data files
│   ├── model1/             # Model1 specific data
│   │   ├── test_data.csv
│   │   └── train_data.csv
│   ├── balanced_dataset_single.csv
│   ├── dataset_multi.csv
│   └── dataset_single.csv
├── models/                 # Trained models and assets
│   ├── model1.joblib
│   ├── smiles_transformer.h5
│   ├── tokenizer.json
│   └── tokenizer.pkl
├── notebooks/              # Jupyter notebooks (if any)
└── src/                    # Main source code
    ├── __pycache__/
    ├── data_separation.py  # Data splitting utilities
    ├── evaluate.py         # Model evaluation
    ├── feature_extractor.py # Molecule feature extraction
    ├── predict_model1.py   # Model1 prediction
    ├── predict_model2.py   # Model2 prediction
    ├── train_model1.py     # Model1 training
    ├── train_model2.py     # Model2 training
    ├── swagger/            # API documentation
    │   └── swagger.yml
    ├── tests/              # Test cases
    ├── .gitignore
    ├── app.py              # Flask application
    ├── Dockerfile          # Container configuration
    ├── README.md           # Documentation
    ├── requirements.txt    # Dependencies
    └── setup.py            # Package configuration
```
## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


```