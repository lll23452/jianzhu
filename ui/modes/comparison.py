"""多建筑对比页面"""
import streamlit as st
import requests
import plotly.graph_objects as go
from config import DEFAULT_HISTORY_VALUES

API_BASE = "http://localhost:5000"
COMPARISON_COLORS = ['#00e676', '#ff9100', '#42a5f5']


def render_comparison_page(building_options, meter_mapping):
    st.markdown("## 📊 多建筑对比")

    selected = st.multiselect("选择建筑（最多 3 个）", building_options,
                               default=building_options[:2] if len(building_options) >= 2 else building_options)
    if len(selected) > 3:
        st.warning("最多选择 3 个建筑")
        selected = selected[:3]

    if st.button("📊 生成对比", type="primary") and selected:
        history = DEFAULT_HISTORY_VALUES.copy()
        st.subheader("24小时能耗对比")
        fig = go.Figure()
        for i, bld in enumerate(selected):
            params_list = [{
                'hour': h, 'month': 7, 'day_of_week': 3,
                'building_id': bld, 'meter': meter_mapping.get('electricity', 0),
                **history
            } for h in range(24)]
            resp = requests.post(f"{API_BASE}/batch_predict", json={"params_list": params_list})
            if resp.status_code == 200:
                preds = resp.json()['predictions']
                color = COMPARISON_COLORS[i % len(COMPARISON_COLORS)]
                fig.add_trace(go.Scatter(x=list(range(24)), y=preds, mode='lines+markers',
                                         line=dict(color=color, width=2), name=bld))
        fig.update_layout(
            title="建筑能耗对比", xaxis_title="小时", yaxis_title="能耗 (kWh)",
            template='plotly_dark', height=450,
            paper_bgcolor='#0f1117', plot_bgcolor='#0f1117',
            font=dict(color='#e0e0e0')
        )
        st.plotly_chart(fig, width='stretch')
