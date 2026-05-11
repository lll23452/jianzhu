"""节能潜力评估模块"""
import pandas as pd
import numpy as np


class EnergySavingAnalyzer:
    def compute_excess(self, actual, predicted):
        return np.maximum(0, actual - predicted)

    def compute_saving_summary(self, df, actual_col='actual', predicted_col='predicted'):
        excess = self.compute_excess(df[actual_col], df[predicted_col])
        total_actual = df[actual_col].sum()
        total_predicted = df[predicted_col].sum()
        manageable_ratio = 0.7
        tot_excess = excess.sum()
        total_saving = tot_excess * manageable_ratio
        return {
            'total_actual': float(total_actual),
            'total_predicted': float(total_predicted),
            'total_excess': float(tot_excess),
            'total_saving': float(total_saving),
            'percent_saving': float(total_saving / total_actual * 100) if total_actual > 0 else 0,
            'co2_reduction': float(total_saving * 0.5),
        }

    def top_excess_buildings(self, df, building_col='building_id', actual_col='actual',
                             predicted_col='predicted', top_n=10):
        df = df.copy()
        df['excess'] = self.compute_excess(df[actual_col], df[predicted_col])
        excess_by_bld = df.groupby(building_col)['excess'].sum().reset_index()
        return excess_by_bld.sort_values('excess', ascending=False).head(top_n)

    def saving_by_hour(self, df, actual_col='actual', predicted_col='predicted'):
        if 'hour' not in df.columns:
            return pd.DataFrame()
        df = df.copy()
        df['excess'] = self.compute_excess(df[actual_col], df[predicted_col])
        result = df.groupby('hour')['excess'].mean().reset_index()
        result.columns = ['hour', 'avg_saveable_kwh']
        return result
