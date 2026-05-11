"""碳排放仪表盘标签页"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
from config import DATA_DIR


def render_carbon_tab():
    st.markdown("## 🌍 碳排放仪表盘")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("总碳排放", "10.05 万吨 CO₂", delta="-5.96%")
    c2.metric("总用电量", "20.10 万 kWh")
    c3.metric("碳因子", "0.5 kg/kWh")
    c4.metric("平均碳强度", "79,373 kg/记录")

    cb_path = os.path.join(DATA_DIR, 'carbon_by_building.csv')
    if os.path.exists(cb_path):
        df = pd.read_csv(cb_path)
        fig = px.bar(df.head(7), x='Building ID', y='kg CO2', color='kg CO2',
                     color_continuous_scale=['#ff9100', '#ff5252'])
        fig.update_layout(template='plotly_dark', paper_bgcolor='#0f1117', plot_bgcolor='#0f1117',
                          font=dict(color='#e0e0e0'), height=400)
        st.plotly_chart(fig, width='stretch')

    ch_path = os.path.join(DATA_DIR, 'carbon_by_hour.csv')
    if os.path.exists(ch_path):
        df = pd.read_csv(ch_path)
        fig = go.Figure(go.Scatter(x=df['Hour'], y=df['kg CO2 (average)'], mode='lines+markers',
                                   line=dict(color='#ff9100', width=2),
                                   fill='tozeroy', fillcolor='rgba(255,145,0,0.08)'))
        fig.update_layout(title="小时平均碳排放", xaxis_title="小时", yaxis_title="kg CO₂",
                          template='plotly_dark', paper_bgcolor='#0f1117', plot_bgcolor='#0f1117',
                          font=dict(color='#e0e0e0'), height=350)
        st.plotly_chart(fig, width='stretch')
