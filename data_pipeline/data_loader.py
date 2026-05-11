"""数据加载器：从文件加载数据并完成预处理+特征工程"""
import os
import pandas as pd
from data_pipeline.preprocessor import EnergyPreprocessor
from data_pipeline.feature_engineer import FeatureEngineer
from config import DATA_DIR


class DataLoader:
    def __init__(self, preprocessor=None, feature_engineer=None):
        self.preprocessor = preprocessor or EnergyPreprocessor()
        self.feature_engineer = feature_engineer or FeatureEngineer()

    def load_from_excel(self, path, sheet_name='output', sample_ratio=1.0):
        df = pd.read_excel(path, sheet_name=sheet_name)
        required = ['meter_reading', 'building_id', 'meter']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if sample_ratio < 1.0:
            df = df.sample(frac=sample_ratio, random_state=42)
        df = df[df['meter_reading'] > 0].copy()
        return df

    def load_and_process(self, path, sheet_name='output', sample_ratio=1.0):
        df = self.load_from_excel(path, sheet_name, sample_ratio)
        df = self.preprocessor.process(df, 'meter_reading')
        df = self.feature_engineer.fit_transform(df, 'meter_reading')
        return df

    def load_test_samples(self, path=None):
        if path is None:
            path = os.path.join(DATA_DIR, 'test_samples.csv')
        if os.path.exists(path):
            return pd.read_csv(path)
        return pd.DataFrame()
