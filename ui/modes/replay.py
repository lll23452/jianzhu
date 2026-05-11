"""历史数据回放页面"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from config import DATA_DIR


def render_replay_page():
    st.markdown("## 📅 历史数据回放")

    test_path = os.path.join(DATA_DIR, 'test_samples.csv')
    if not os.path.exists(test_path):
        st.warning("测试样本数据未生成，请先运行训练脚本并导出 test_samples.csv")
        return

    df = pd.read_csv(test_path)
    if 'timestamp' not in df.columns:
        st.error("数据缺少 timestamp 列")
        return

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    date_min = df['timestamp'].min().date()
    date_max = df['timestamp'].max().date()
    date_range = st.date_input("选择日期范围", [date_min, date_max])

    if st.button("📅 加载数据", type="primary"):
        mask = (df['timestamp'].dt.date >= date_range[0]) & (df['timestamp'].dt.date <= date_range[-1])
        filtered = df.loc[mask]
        if filtered.empty:
            st.warning("所选日期范围内无数据")
            return

        fig = go.Figure()
        if 'actual' in filtered.columns:
            fig.add_trace(go.Scatter(x=filtered['timestamp'], y=filtered['actual'],
                                     mode='lines', name='实际值', line=dict(color='#00e676', width=2)))
        if 'predicted' in filtered.columns:
            fig.add_trace(go.Scatter(x=filtered['timestamp'], y=filtered['predicted'],
                                     mode='lines', name='预测值', line=dict(color='#ff9100', width=2)))
        fig.update_layout(
            title="历史能耗回放", xaxis_title="时间", yaxis_title="能耗 (kWh)",
            template='plotly_dark', height=450,
            paper_bgcolor='#0f1117', plot_bgcolor='#0f1117',
            font=dict(color='#e0e0e0')
        )
        st.plotly_chart(fig, width='stretch')

        if 'actual' in filtered.columns and 'predicted' in filtered.columns:
            error = filtered['actual'] - filtered['predicted']
            c1, c2, c3 = st.columns(3)
            c1.metric("MAE", f"{np.mean(np.abs(error)):,.2f} kWh")
            c2.metric("RMSE", f"{np.sqrt(np.mean(error ** 2)):,.2f} kWh")
            c3.metric("MAPE", f"{np.mean(np.abs(error) / (filtered['actual'] + 1)) * 100:.2f}%")
