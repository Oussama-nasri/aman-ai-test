from flask import Blueprint, request, jsonify
from src.predict_model1 import predict_model1
from src.predict_model2 import predict_model2

api_bp = Blueprint('api', __name__)

@api_bp.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'smiles' not in data or 'model' not in data:
            return jsonify({'error': 'Missing "smiles" or "model" in request body'}), 400

        smiles = data['smiles']
        model = data['model']

        # Choose model
        if model == 1:
            prediction, probability = predict_model1(smiles)
        elif model == 2:
            prediction, probability = predict_model2(smiles)
        else:
            return jsonify({'error': 'Invalid model specified. Use 1 or 2.'}), 400

        response = {'prediction': int(prediction) if prediction is not None else None}
        if probability is not None:
            response['probabilities'] = probability.tolist()

        return jsonify(response), 200

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500