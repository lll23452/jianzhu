"""碳排放核算模块"""
import pandas as pd
from config import CARBON_FACTOR


class CarbonCalculator:
    def __init__(self, carbon_factor=CARBON_FACTOR):
        self.carbon_factor = carbon_factor

    def compute_record_carbon(self, meter_reading):
        return meter_reading * self.carbon_factor

    def compute_summary(self, df, value_col='meter_reading'):
        total_energy = df[value_col].sum()
        total_carbon = total_energy * self.carbon_factor
        return {
            'total_energy': float(total_energy),
            'total_carbon': float(total_carbon),
            'avg_carbon_per_record': float(total_carbon / len(df)) if len(df) > 0 else 0,
            'carbon_factor': self.carbon_factor,
            'record_count': len(df)
        }

    def carbon_by_building(self, df, value_col='meter_reading', building_col='building_id'):
        result = df.groupby(building_col)[value_col].sum().reset_index()
        result['carbon_kg'] = result[value_col] * self.carbon_factor
        result = result.sort_values('carbon_kg', ascending=False)
        return result

    def carbon_by_hour(self, df, value_col='meter_reading'):
        if 'hour' not in df.columns:
            return pd.DataFrame()
        result = df.groupby('hour')[value_col].mean().reset_index()
        result['carbon_kg_avg'] = result[value_col] * self.carbon_factor
        return result
