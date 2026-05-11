"""模型解释标签页"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def render_explanation_tab():
    st.markdown("## 🔍 模型解释")

    shap_data = {
        '特征': ['meter_reading_rolling_mean_3', 'meter_reading_lag_2',
                 'meter_reading_lag_1', 'meter_reading_rolling_std_3',
                 'meter_reading_rolling_mean_6', 'meter_reading_lag_3',
                 'meter_reading_rolling_mean_12', 'meter_reading_rolling_mean_24'],
        'SHAP重要性': [1.553, 0.385, 0.262, 0.134, 0.082, 0.068, 0.051, 0.043]
    }
    df = pd.DataFrame(shap_data)

    fig = go.Figure(go.Bar(
        x=df['SHAP重要性'], y=df['特征'], orientation='h',
        marker=dict(color='#00e676'),
        text=df['SHAP重要性'].round(3), textposition='outside',
        textfont=dict(color='#e0e0e0')
    ))
    fig.update_layout(
        title="SHAP 特征重要性排名",
        xaxis_title="平均 |SHAP|", yaxis=dict(autorange='reversed'),
        template='plotly_dark', height=400,
        paper_bgcolor='#0f1117', plot_bgcolor='#0f1117',
        font=dict(color='#e0e0e0')
    )
    st.plotly_chart(fig, width='stretch')

    st.markdown("""
    <div style="background:#1a1d27;border-left:3px solid #00e676;padding:12px 16px;margin-top:8px;border-radius:2px;">
    <strong>特征解读</strong><br>
    近3小时滚动均值（rolling_mean_3）对预测影响最大（SHAP=1.553），滞后2小时和滞后1小时能耗次之。
    模型强依赖历史负荷模式，时间编码特征（hour_sin/cos）相对影响较小。这表示建筑能耗具有强短期惯性。
    </div>
    """, unsafe_allow_html=True)
