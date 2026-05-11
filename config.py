"""全局配置常量"""
import os

# 路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'saved_models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PLOT_DIR = os.path.join(BASE_DIR, 'plots')

# 碳排放因子 (kg CO2/kWh)
CARBON_FACTOR = 0.5

# 模型文件路径
ONNX_MODEL_PATH = os.path.join(MODEL_DIR, 'energy_model.onnx')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoders.pkl')
XGB_MODEL_PATH = os.path.join(MODEL_DIR, 'pso_xgboost_model.json')

# 特征名称（按训练顺序）
FEATURE_NAMES = [
    'hour', 'day_of_week', 'month', 'weekend',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'meter_reading_lag_1', 'meter_reading_lag_2', 'meter_reading_lag_3', 'meter_reading_lag_24',
    'meter_reading_rolling_mean_3', 'meter_reading_rolling_mean_6',
    'meter_reading_rolling_mean_12', 'meter_reading_rolling_mean_24',
    'meter_reading_rolling_std_3', 'meter_reading_rolling_std_6',
    'meter_reading_rolling_std_12', 'meter_reading_rolling_std_24',
    'building_id_encoded', 'meter_encoded'
]

# 历史特征默认值（训练集均值）
DEFAULT_HISTORY_VALUES = {
    'meter_reading_lag_1': 50000.0, 'meter_reading_lag_2': 50000.0,
    'meter_reading_lag_3': 50000.0, 'meter_reading_lag_24': 48000.0,
    'meter_reading_rolling_mean_3': 50000.0, 'meter_reading_rolling_mean_6': 49000.0,
    'meter_reading_rolling_mean_12': 48500.0, 'meter_reading_rolling_mean_24': 48000.0,
    'meter_reading_rolling_std_3': 2000.0, 'meter_reading_rolling_std_6': 2500.0,
    'meter_reading_rolling_std_12': 3000.0, 'meter_reading_rolling_std_24': 3500.0,
}

# PSO 超参数搜索边界
XGB_PARAM_BOUNDS = [
    (0.01, 0.3),   # learning_rate
    (3, 10),       # max_depth
    (1, 10),       # min_child_weight
    (0.5, 1.0),    # subsample
    (0.5, 1.0),    # colsample_bytree
    (50, 300),     # n_estimators
    (0, 10),       # reg_alpha
    (1, 10)        # reg_lambda
]

# 训练配置
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2
RANDOM_STATE = 42
