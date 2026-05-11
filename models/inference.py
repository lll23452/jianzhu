"""ONNX Runtime 推理封装"""
import os
import numpy as np
import joblib
import onnxruntime as ort
from config import ONNX_MODEL_PATH, SCALER_PATH, ENCODER_PATH
from data_pipeline.feature_engineer import FeatureEngineer


class ModelInference:
    def __init__(self):
        if not os.path.exists(ONNX_MODEL_PATH):
            raise FileNotFoundError(f"ONNX model not found: {ONNX_MODEL_PATH}")
        self.session = ort.InferenceSession(ONNX_MODEL_PATH)
        self.input_name = self.session.get_inputs()[0].name
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")
        self.scaler = joblib.load(SCALER_PATH)
        self.feature_engineer = FeatureEngineer()
        if os.path.exists(ENCODER_PATH):
            encoders = joblib.load(ENCODER_PATH)
            self.feature_engineer.building_encoder = encoders.get('building_id')
            self.feature_engineer.meter_encoder = encoders.get('meter')
            self.feature_engineer._fitted = True

    def _build_features(self, params):
        raw = self.feature_engineer.build_single_vector(
            hour=params.get('hour', 0),
            month=params.get('month', 1),
            day_of_week=params.get('day_of_week', 0),
            building_id=params.get('building_id', 0),
            meter=params.get('meter', 0),
            history_values={k: params[k] for k in params if k.startswith('meter_reading_')}
        )
        return self.scaler.transform(raw)

    def predict(self, params):
        features = self._build_features(params)
        pred = self.session.run(None, {self.input_name: features})[0]
        return float(np.expm1(pred[0][0]))

    def batch_predict(self, params_list):
        if not params_list:
            return []
        feature_matrix = np.vstack([self._build_features(p) for p in params_list])
        preds = self.session.run(None, {self.input_name: feature_matrix})[0]
        return np.expm1(preds).flatten().tolist()

    def get_building_list(self):
        if self.feature_engineer._fitted and hasattr(self.feature_engineer.building_encoder, 'classes_'):
            return list(self.feature_engineer.building_encoder.classes_)
        return []
