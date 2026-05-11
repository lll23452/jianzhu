"""数据预处理：线性插值、分位数截断、异常值清洗"""
import numpy as np
import pandas as pd


class EnergyPreprocessor:
    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self._lower_bound = None
        self._upper_bound = None

    def linear_interpolate(self, data, value_col='meter_reading'):
        df = data.copy()
        if 'timestamp' in df.columns:
            df = df.sort_values(['building_id', 'timestamp'])
            df[value_col] = df.groupby('building_id')[value_col].transform(
                lambda x: x.interpolate(method='linear', limit_direction='both')
            )
        else:
            mask = df[value_col].isna()
            if mask.any():
                df[value_col] = df[value_col].interpolate(method='linear', limit_direction='both')
        return df

    def fit_quantile_bounds(self, data, value_col='meter_reading'):
        values = data[value_col].dropna()
        self._lower_bound = values.quantile(self.lower_quantile)
        self._upper_bound = values.quantile(self.upper_quantile)

    def quantile_clip(self, data, value_col='meter_reading'):
        if self._lower_bound is None or self._upper_bound is None:
            self.fit_quantile_bounds(data, value_col)
        df = data.copy()
        df[value_col] = df[value_col].clip(lower=self._lower_bound, upper=self._upper_bound)
        return df

    def remove_outliers_zscore(self, data, value_col='meter_reading', threshold=3.0):
        values = data[value_col].dropna()
        z_scores = np.abs((values - values.mean()) / values.std())
        mask = z_scores <= threshold
        return data.loc[values.index[mask]]

    def process(self, data, value_col='meter_reading'):
        df = self.linear_interpolate(data, value_col)
        self.fit_quantile_bounds(df, value_col)
        df = self.quantile_clip(df, value_col)
        return df
