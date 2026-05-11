"""单点预测页面"""
import streamlit as st
import requests
import plotly.graph_objects as go
from config import DEFAULT_HISTORY_VALUES

API_BASE = "http://localhost:5000"


def render_prediction_page(building_options, meter_mapping):
    st.markdown("## 📈 单点预测")

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_building = st.selectbox("建筑", building_options)
    with col2:
        hour = st.slider("小时", 0, 23, 14)
    with col3:
        month = st.slider("月份", 1, 12, 7)

    day_of_week = st.slider("星期 (0=周一)", 0, 6, 3)

    with st.expander("📊 历史负荷参数（高级）"):
        history = {}
        cols = st.columns(4)
        keys = list(DEFAULT_HISTORY_VALUES.keys())
        for i, k in enumerate(keys):
            history[k] = cols[i % 4].number_input(
                k.replace('meter_reading_', ''), value=DEFAULT_HISTORY_VALUES[k],
                step=1000.0 if 'std' not in k else 100.0
            )

    if st.button("🚀 执行预测", type="primary"):
        params = {
            'hour': hour, 'month': month, 'day_of_week': day_of_week,
            'building_id': selected_building, 'meter': meter_mapping.get('electricity', 0),
            **history
        }
        resp = requests.post(f"{API_BASE}/predict", json={"params": params})
        if resp.status_code == 200:
            pred = resp.json()['prediction']
            c1, c2, c3 = st.columns(3)
            c1.metric("预测能耗", f"{pred:,.0f} kWh")
            c2.metric("预估碳排放", f"{pred * 0.5:,.0f} kg CO₂")
            c3.metric("碳因子", "0.5 kg/kWh")

            params_list = [{
                'hour': h, 'month': month, 'day_of_week': day_of_week,
                'building_id': selected_building, 'meter': 0, **history
            } for h in range(24)]
            batch_resp = requests.post(f"{API_BASE}/batch_predict", json={"params_list": params_list})
            if batch_resp.status_code == 200:
                preds = batch_resp.json()['predictions']
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(range(24)), y=preds, mode='lines+markers',
                                         line=dict(color='#00e676', width=2),
                                         fill='tozeroy', fillcolor='rgba(0,230,118,0.08)',
                                         name='预测能耗'))
                fig.update_layout(
                    title="24小时能耗预测曲线", xaxis_title="小时", yaxis_title="能耗 (kWh)",
                    template='plotly_dark', height=400,
                    paper_bgcolor='#0f1117', plot_bgcolor='#0f1117',
                    font=dict(color='#e0e0e0')
                )
                st.plotly_chart(fig, width='stretch')
                st.session_state['last_prediction'] = {
                    'building': selected_building, 'hour': hour, 'month': month,
                    'prediction': pred, 'curve': preds
                }
        else:
            st.error(f"预测失败: {resp.text}")
