"""场景模拟器页面 — 通过调节历史负荷参数间接模拟不同运行场景"""
import streamlit as st
import requests
import plotly.graph_objects as go
from config import DEFAULT_HISTORY_VALUES

API_BASE = "http://localhost:5000"
SCENE_PRESETS = {"基准场景": 1.0, "节能场景": 0.75, "高峰场景": 1.25, "极端高峰": 1.5}


def render_simulator_page(building_options, meter_mapping):
    st.markdown("## 🔬 场景模拟器")
    st.caption("通过调节历史负荷参数模拟不同运行条件下的能耗响应")

    col1, col2 = st.columns(2)
    with col1:
        selected_building = st.selectbox("选择建筑", building_options)
        month = st.slider("月份", 1, 12, 7)
    with col2:
        day_of_week = st.slider("星期", 0, 6, 3)
        preset = st.selectbox("场景预设", list(SCENE_PRESETS.keys()))

    base_multiplier = SCENE_PRESETS[preset]
    multiplier = st.slider("负荷系数微调", 0.5, 1.5, base_multiplier, 0.05,
                           help="<1.0 = 节能场景，>1.0 = 高峰场景")

    if st.button("🔬 运行模拟", type="primary"):
        base_history = DEFAULT_HISTORY_VALUES.copy()
        scene_history = {k: v * multiplier for k, v in DEFAULT_HISTORY_VALUES.items()}

        base_params_list = [{
            'hour': h, 'month': month, 'day_of_week': day_of_week,
            'building_id': selected_building, 'meter': meter_mapping.get('electricity', 0),
            **base_history
        } for h in range(24)]
        scene_params_list = [{
            'hour': h, 'month': month, 'day_of_week': day_of_week,
            'building_id': selected_building, 'meter': meter_mapping.get('electricity', 0),
            **scene_history
        } for h in range(24)]

        base_resp = requests.post(f"{API_BASE}/batch_predict", json={"params_list": base_params_list})
        scene_resp = requests.post(f"{API_BASE}/batch_predict", json={"params_list": scene_params_list})

        if base_resp.status_code == 200 and scene_resp.status_code == 200:
            base_preds = base_resp.json()['predictions']
            scene_preds = scene_resp.json()['predictions']

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(24)), y=base_preds, mode='lines+markers',
                                     line=dict(color='#8892a4', width=2), name='基准'))
            fig.add_trace(go.Scatter(x=list(range(24)), y=scene_preds, mode='lines+markers',
                                     line=dict(color='#00e676', width=2.5), name=f'场景 (×{multiplier:.2f})'))
            fig.update_layout(
                title=f"场景模拟对比 — {selected_building}",
                xaxis_title="小时", yaxis_title="能耗 (kWh)",
                template='plotly_dark', height=450,
                paper_bgcolor='#0f1117', plot_bgcolor='#0f1117',
                font=dict(color='#e0e0e0')
            )
            st.plotly_chart(fig, width='stretch')

            delta_total = sum(scene_preds) - sum(base_preds)
            c1, c2, c3 = st.columns(3)
            c1.metric("基准日总能耗", f"{sum(base_preds):,.0f} kWh")
            c2.metric("场景日总能耗", f"{sum(scene_preds):,.0f} kWh", delta=f"{delta_total:+,.0f} kWh")
            c3.metric("碳排放变化", f"{delta_total * 0.5:+,.0f} kg CO₂")
