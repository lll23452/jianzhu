"""XGBoost JSON → ONNX 转换"""
import os
import xgboost as xgb
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
from config import XGB_MODEL_PATH, MODEL_DIR


def convert_to_onnx(num_features=22, output_path=None):
    if not os.path.exists(XGB_MODEL_PATH):
        raise FileNotFoundError(f"XGBoost model not found: {XGB_MODEL_PATH}")
    bst = xgb.Booster()
    bst.load_model(XGB_MODEL_PATH)
    initial_type = [('input', FloatTensorType([None, num_features]))]
    onnx_model = onnxmltools.convert_xgboost(bst, initial_types=initial_type)
    if output_path is None:
        output_path = os.path.join(MODEL_DIR, 'energy_model.onnx')
    with open(output_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    print(f"ONNX model saved to {output_path}")
    return output_path


if __name__ == '__main__':
    convert_to_onnx()
