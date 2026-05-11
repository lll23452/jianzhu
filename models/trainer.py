"""XGBoost 训练器"""
import os
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_STATE, MODEL_DIR


def smape(y_true, y_pred):
    denominator = np.abs(y_true) + np.abs(y_pred)
    diff = np.abs(y_true - y_pred)
    mask = denominator > 0
    return float(np.mean(2.0 * diff[mask] / denominator[mask]) * 100)


def evaluate(y_true, y_pred):
    return {
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'sMAPE': smape(y_true, y_pred),
        'R²': float(r2_score(y_true, y_pred)),
    }


def train_xgboost(X, y, params=None, save=True):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(VAL_RATIO + TEST_RATIO), random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO), random_state=RANDOM_STATE)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    y_train_log = np.log1p(y_train)
    if params is None:
        params = {
            'learning_rate': 0.05, 'max_depth': 6, 'min_child_weight': 3,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'n_estimators': 200,
            'reg_alpha': 1, 'reg_lambda': 5,
            'objective': 'reg:squarederror', 'random_state': RANDOM_STATE,
        }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train_scaled, y_train_log,
              eval_set=[(X_val_scaled, np.log1p(y_val))],
              verbose=False)
    y_pred_log = model.predict(X_test_scaled)
    y_pred = np.expm1(y_pred_log)
    metrics = evaluate(y_test, y_pred)
    if save:
        os.makedirs(MODEL_DIR, exist_ok=True)
        model.save_model(os.path.join(MODEL_DIR, 'pso_xgboost_model.json'))
        joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    return model, scaler, metrics
