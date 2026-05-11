"""Flask API 路由定义"""
from flask import Blueprint, request, jsonify
from models.inference import ModelInference

api = Blueprint('api', __name__)
_inference: ModelInference = None


def init_inference(inference: ModelInference):
    global _inference
    _inference = inference


@api.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': _inference is not None})


@api.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'params' not in data:
            return jsonify({'status': 'error', 'message': 'Missing "params" field'}), 400
        pred = _inference.predict(data['params'])
        return jsonify({'prediction': pred, 'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@api.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        data = request.get_json()
        params_list = data.get('params_list', [])
        if not params_list:
            return jsonify({'status': 'error', 'message': 'Empty params_list'}), 400
        preds = _inference.batch_predict(params_list)
        return jsonify({'predictions': preds, 'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@api.route('/buildings', methods=['GET'])
def buildings():
    try:
        building_list = _inference.get_building_list()
        return jsonify({'buildings': building_list, 'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
