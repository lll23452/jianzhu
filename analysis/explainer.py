"""SHAP 可解释性分析模块"""
import os
import numpy as np
import pandas as pd
import joblib
import shap
import xgboost as xgb
from config import XGB_MODEL_PATH, FEATURE_NAMES


class ShapExplainer:
    def __init__(self):
        self.model = None
        self.explainer = None
        self._shap_values = None
        self._X_display = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(XGB_MODEL_PATH):
            raise FileNotFoundError(f"XGBoost model not found: {XGB_MODEL_PATH}")
        self.model = xgb.Booster()
        self.model.load_model(XGB_MODEL_PATH)
        self.explainer = shap.TreeExplainer(self.model)

    def compute_global_shap(self, X, max_samples=2000):
        if len(X) > max_samples:
            X = X.sample(n=max_samples, random_state=42)
        self._X_display = X
        self._shap_values = self.explainer.shap_values(X)
        return self._shap_values

    def get_feature_importance(self):
        if self._shap_values is None:
            return pd.DataFrame()
        importance = np.abs(self._shap_values).mean(axis=0)
        result = pd.DataFrame({
            'feature': FEATURE_NAMES,
            'shap_importance': importance
        }).sort_values('shap_importance', ascending=False)
        return result

    def explain_single(self, X_row):
        shap_vals = self.explainer.shap_values(X_row)
        if shap_vals.ndim > 1:
            shap_vals = shap_vals[0]
        contributions = [
            {'feature': FEATURE_NAMES[i], 'contribution': float(shap_vals[i]),
             'value': float(X_row[0][i]) if X_row.ndim > 1 else float(X_row[i])}
            for i in range(len(FEATURE_NAMES))
        ]
        contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
        return {
            'base_value': float(self.explainer.expected_value),
            'contributions': contributions[:10]
        }

    def load_precomputed(self):
        shap_path = 'saved_models/shap_values.pkl'
        X_path = 'saved_models/X_display.pkl'
        if os.path.exists(shap_path) and os.path.exists(X_path):
            self._shap_values = joblib.load(shap_path)
            self._X_display = joblib.load(X_path)
            return True
        return False
