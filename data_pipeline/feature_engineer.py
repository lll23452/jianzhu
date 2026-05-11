"""特征工程：时间编码、滞后/滚动特征、类别编码"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class FeatureEngineer:
    def __init__(self):
        self.building_encoder = LabelEncoder()
        self.meter_encoder = LabelEncoder()
        self._fitted = False

    def add_time_features(self, data):
        df = data.copy()
        if 'timestamp' not in df.columns:
            timestamp = pd.to_datetime(
                dict(year=2016, month=df.get('month', 1), day=1, hour=df.get('hour', 0))
            )
        else:
            timestamp = pd.to_datetime(df['timestamp'])
        df['hour'] = timestamp.dt.hour
        df['day_of_week'] = timestamp.dt.dayofweek
        df['month'] = timestamp.dt.month
        df['weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        return df

    def add_lag_features(self, data, value_col='meter_reading', lags=(1, 2, 3, 24)):
        df = data.copy()
        if 'building_id' in df.columns:
            grouped = df.groupby('building_id')[value_col]
            for lag in lags:
                df[f'meter_reading_lag_{lag}'] = grouped.shift(lag)
        else:
            for lag in lags:
                df[f'meter_reading_lag_{lag}'] = df[value_col].shift(lag)
        return df

    def add_rolling_features(self, data, value_col='meter_reading', windows=(3, 6, 12, 24)):
        df = data.copy()
        if 'building_id' in df.columns:
            grouped = df.groupby('building_id')[value_col]
            for w in windows:
                df[f'meter_reading_rolling_mean_{w}'] = grouped.transform(
                    lambda x: x.rolling(w, min_periods=1).mean())
                df[f'meter_reading_rolling_std_{w}'] = grouped.transform(
                    lambda x: x.rolling(w, min_periods=1).std())
        else:
            for w in windows:
                df[f'meter_reading_rolling_mean_{w}'] = df[value_col].rolling(w, min_periods=1).mean()
                df[f'meter_reading_rolling_std_{w}'] = df[value_col].rolling(w, min_periods=1).std()
        return df

    def fit_encoders(self, data):
        self.building_encoder.fit(data['building_id'].astype(str))
        if 'meter' in data.columns:
            self.meter_encoder.fit(data['meter'].astype(str))
        self._fitted = True

    def transform_encoders(self, data):
        df = data.copy()
        df['building_id_encoded'] = self.building_encoder.transform(df['building_id'].astype(str))
        if 'meter' in df.columns:
            df['meter_encoded'] = self.meter_encoder.transform(df['meter'].astype(str))
        else:
            df['meter_encoded'] = 0
        return df

    def fit_transform(self, data, value_col='meter_reading'):
        df = self.add_time_features(data)
        df = self.add_lag_features(df, value_col)
        df = self.add_rolling_features(df, value_col)
        lag_cols = [c for c in df.columns if c.startswith('meter_reading_lag')]
        roll_cols = [c for c in df.columns if c.startswith('meter_reading_rolling')]
        for c in lag_cols:
            df[c] = df[c].fillna(df[value_col])
        for c in roll_cols:
            if 'std' in c:
                df[c] = df[c].fillna(0)
            else:
                df[c] = df[c].fillna(df[value_col])
        self.fit_encoders(df)
        df = self.transform_encoders(df)
        return df

    def build_single_vector(self, hour, month, day_of_week, building_id, meter=0, history_values=None):
        from config import DEFAULT_HISTORY_VALUES
        defaults = DEFAULT_HISTORY_VALUES.copy()
        if history_values:
            defaults.update(history_values)
        weekend = 1 if day_of_week >= 5 else 0
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        try:
            bid_enc = self.building_encoder.transform([str(building_id)])[0]
        except ValueError:
            bid_enc = 0
        try:
            mtr_enc = self.meter_encoder.transform([str(meter)])[0]
        except ValueError:
            mtr_enc = 0
        return np.array([[
            hour, day_of_week, month, weekend,
            hour_sin, hour_cos, month_sin, month_cos,
            defaults['meter_reading_lag_1'], defaults['meter_reading_lag_2'],
            defaults['meter_reading_lag_3'], defaults['meter_reading_lag_24'],
            defaults['meter_reading_rolling_mean_3'], defaults['meter_reading_rolling_mean_6'],
            defaults['meter_reading_rolling_mean_12'], defaults['meter_reading_rolling_mean_24'],
            defaults['meter_reading_rolling_std_3'], defaults['meter_reading_rolling_std_6'],
            defaults['meter_reading_rolling_std_12'], defaults['meter_reading_rolling_std_24'],
            bid_enc, mtr_enc
        ]], dtype=np.float32)
