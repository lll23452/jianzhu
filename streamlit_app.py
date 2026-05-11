"""streamlit_app.py — 独立版能耗预测平台（直接内嵌ONNX推理，无需Flask）"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os
import base64
from datetime import datetime

# ---- 模型推理（内嵌） ----
from models.inference import ModelInference


@st.cache_resource
def load_model():
    return ModelInference()


model = load_model()

# ---- 配置 ----
from config import DEFAULT_HISTORY_VALUES, DATA_DIR

st.set_page_config(page_title="建筑能耗预测与诊断平台", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

# ---- CSS ----
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Teko:wght@400;500;600&family=Work+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
:root {
    --bg-root: #0f1117; --bg-surface: #1a1d27; --bg-elevated: #242836;
    --accent-energy: #00e676; --accent-carbon: #ff9100; --accent-danger: #ff5252;
    --text-primary: #cccccc; --text-secondary: #999999; --border-subtle: #2a2e3a;
    --font-display: 'Teko', sans-serif; --font-body: 'Work Sans', sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
}
.stApp { background-color: var(--bg-root); }
[data-testid="stSidebar"] { background-color: var(--bg-surface); border-right: 1px solid var(--border-subtle); }
[data-baseweb="select"] [role="listbox"] li, [data-baseweb="menu"] [role="option"], ul[role="listbox"] li, [data-baseweb="select"] [role="option"] { color: #000000 !important; }
h1 { font-family: var(--font-display) !important; font-size: 2.5rem !important; font-weight: 600 !important; letter-spacing: 0.04em !important; color: #00e676 !important; text-transform: uppercase; }
h2, h3 { font-family: var(--font-display) !important; font-weight: 500 !important; letter-spacing: 0.02em !important; color: #ffffff !important; }
[data-testid="stMetric"] { background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 4px; padding: 12px 16px; }
[data-testid="stMetricValue"] { font-family: var(--font-mono) !important; font-size: 1.6rem !important; color: #00e676 !important; }
.stButton > button { font-family: var(--font-display) !important; letter-spacing: 0.05em !important; text-transform: uppercase; background: var(--accent-energy); color: #0f1117; border: none; border-radius: 2px; padding: 8px 24px; transition: all 0.2s; }
.stButton > button:hover { box-shadow: 0 0 20px rgba(0, 230, 118, 0.3); }
[data-testid="stDataFrame"] { background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 4px; }
.stTabs [aria-selected="true"] { color: #00e676 !important; }
[data-testid="stExpander"] { background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ---- 建筑列表 ----
BUILDING_OPTIONS = model.get_building_list()
if not BUILDING_OPTIONS:
    BUILDING_OPTIONS = ['NDRC Building', "People's Hall", 'Benxi Central Hospital']
METER_MAPPING = {'electricity': 0}
COMPARISON_COLORS = ['#00e676', '#ff9100', '#42a5f5']
SCENE_PRESETS = {"基准场景": 1.0, "节能场景": 0.75, "高峰场景": 1.25, "极端高峰": 1.5}

# ---- 侧边栏 ----
with st.sidebar:
    st.markdown("## ⚡ 控制台")
    st.markdown('<span style="color:#00e676;font-family:JetBrains Mono,monospace;">● 模型已就绪</span>', unsafe_allow_html=True)
    st.markdown("---")
    mode = st.radio("选择模式", ["单点预测", "多建筑对比", "历史数据回放", "场景模拟器"])

# ---- 顶部标题和KPI ----
st.title("⚡ 城市公共建筑能耗预测与诊断平台")

st.markdown("""
<div style="display:flex;gap:16px;margin-bottom:24px;flex-wrap:wrap;">
<div style="flex:1;min-width:200px;background:#1a1d27;border:1px solid #2a2e3a;border-radius:4px;padding:16px;border-left:3px solid #00e676;">
    <div style="color:#999999;font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;">总用电量</div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:1.5rem;color:#00e676;">20.10 万 kWh</div>
</div>
<div style="flex:1;min-width:200px;background:#1a1d27;border:1px solid #2a2e3a;border-radius:4px;padding:16px;border-left:3px solid #ff9100;">
    <div style="color:#999999;font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;">总碳排放</div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:1.5rem;color:#ff9100;">10.05 万吨 CO₂</div>
</div>
<div style="flex:1;min-width:200px;background:#1a1d27;border:1px solid #2a2e3a;border-radius:4px;padding:16px;border-left:3px solid #00e676;">
    <div style="color:#999999;font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;">模型 R²</div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:1.5rem;color:#00e676;">0.9360</div>
</div>
<div style="flex:1;min-width:200px;background:#1a1d27;border:1px solid #2a2e3a;border-radius:4px;padding:16px;border-left:3px solid #ff5252;">
    <div style="color:#999999;font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;">可节约</div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:1.5rem;color:#ff5252;">5.96%</div>
</div>
</div>
""", unsafe_allow_html=True)

# ====== 标签页 ======
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 预测分析", "🌍 碳排放看板", "💡 节能评估", "🔍 模型解释", "📋 报告导出"])

# ====== Tab 1: 预测分析 ======
with tab1:
    # ---- 单点预测 ----
    if mode == "单点预测":
        st.markdown("## 📈 单点预测")
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_building = st.selectbox("建筑", BUILDING_OPTIONS)
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
                    step=1000.0 if 'std' not in k else 100.0, key=f"pred_{k}")

        if st.button("🚀 执行预测", type="primary"):
            params = {'hour': hour, 'month': month, 'day_of_week': day_of_week,
                      'building_id': selected_building, 'meter': 0, **history}
            pred = model.predict(params)
            c1, c2, c3 = st.columns(3)
            c1.metric("预测能耗", f"{pred:,.0f} kWh")
            c2.metric("预估碳排放", f"{pred * 0.5:,.0f} kg CO₂")
            c3.metric("碳因子", "0.5 kg/kWh")

            params_list = [{'hour': h, 'month': month, 'day_of_week': day_of_week,
                            'building_id': selected_building, 'meter': 0, **history} for h in range(24)]
            preds = model.batch_predict(params_list)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(24)), y=preds, mode='lines+markers',
                                     line=dict(color='#00e676', width=2),
                                     fill='tozeroy', fillcolor='rgba(0,230,118,0.08)', name='预测能耗'))
            fig.update_layout(title="24小时能耗预测曲线", xaxis_title="小时", yaxis_title="能耗 (kWh)",
                              template='plotly_dark', height=400, paper_bgcolor='#0f1117',
                              plot_bgcolor='#0f1117', font=dict(color='#e0e0e0'))
            st.plotly_chart(fig, width='stretch')
            st.session_state['last_prediction'] = {
                'building': selected_building, 'hour': hour, 'month': month,
                'prediction': pred, 'curve': preds
            }

    # ---- 多建筑对比 ----
    elif mode == "多建筑对比":
        st.markdown("## 📊 多建筑对比")
        selected = st.multiselect("选择建筑（最多 3 个）", BUILDING_OPTIONS,
                                   default=BUILDING_OPTIONS[:2] if len(BUILDING_OPTIONS) >= 2 else BUILDING_OPTIONS)
        if len(selected) > 3:
            st.warning("最多选择 3 个建筑")
            selected = selected[:3]

        if st.button("📊 生成对比", type="primary") and selected:
            st.subheader("24小时能耗对比")
            fig = go.Figure()
            history = DEFAULT_HISTORY_VALUES.copy()
            for i, bld in enumerate(selected):
                params_list = [{'hour': h, 'month': 7, 'day_of_week': 3,
                                'building_id': bld, 'meter': 0, **history} for h in range(24)]
                preds = model.batch_predict(params_list)
                color = COMPARISON_COLORS[i % len(COMPARISON_COLORS)]
                fig.add_trace(go.Scatter(x=list(range(24)), y=preds, mode='lines+markers',
                                         line=dict(color=color, width=2), name=str(bld)))
            fig.update_layout(title="建筑能耗对比", xaxis_title="小时", yaxis_title="能耗 (kWh)",
                              template='plotly_dark', height=450, paper_bgcolor='#0f1117',
                              plot_bgcolor='#0f1117', font=dict(color='#e0e0e0'))
            st.plotly_chart(fig, width='stretch')

    # ---- 历史数据回放 ----
    elif mode == "历史数据回放":
        st.markdown("## 📅 历史数据回放")
        test_path = os.path.join(DATA_DIR, 'test_samples.csv')
        if not os.path.exists(test_path):
            st.warning("测试样本数据未生成，请先运行训练脚本并导出 test_samples.csv")
        else:
            df = pd.read_csv(test_path)
            if 'timestamp' not in df.columns:
                st.error("数据缺少 timestamp 列")
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                date_min = df['timestamp'].min().date()
                date_max = df['timestamp'].max().date()
                date_range = st.date_input("选择日期范围", [date_min, date_max])
                if st.button("📅 加载数据", type="primary"):
                    mask = (df['timestamp'].dt.date >= date_range[0]) & (df['timestamp'].dt.date <= date_range[-1])
                    filtered = df.loc[mask]
                    if filtered.empty:
                        st.warning("所选日期范围内无数据")
                    else:
                        fig = go.Figure()
                        if 'actual' in filtered.columns:
                            fig.add_trace(go.Scatter(x=filtered['timestamp'], y=filtered['actual'],
                                                     mode='lines', name='实际值', line=dict(color='#00e676', width=2)))
                        if 'predicted' in filtered.columns:
                            fig.add_trace(go.Scatter(x=filtered['timestamp'], y=filtered['predicted'],
                                                     mode='lines', name='预测值', line=dict(color='#ff9100', width=2)))
                        fig.update_layout(title="历史能耗回放", xaxis_title="时间", yaxis_title="能耗 (kWh)",
                                          template='plotly_dark', height=450, paper_bgcolor='#0f1117',
                                          plot_bgcolor='#0f1117', font=dict(color='#e0e0e0'))
                        st.plotly_chart(fig, width='stretch')
                        if 'actual' in filtered.columns and 'predicted' in filtered.columns:
                            error = filtered['actual'] - filtered['predicted']
                            c1, c2, c3 = st.columns(3)
                            c1.metric("MAE", f"{np.mean(np.abs(error)):,.2f} kWh")
                            c2.metric("RMSE", f"{np.sqrt(np.mean(error ** 2)):,.2f} kWh")
                            c3.metric("MAPE", f"{np.mean(np.abs(error) / (filtered['actual'] + 1)) * 100:.2f}%")

    # ---- 场景模拟器 ----
    elif mode == "场景模拟器":
        st.markdown("## 🔬 场景模拟器")
        st.caption("通过调节历史负荷参数模拟不同运行条件下的能耗响应")
        col1, col2 = st.columns(2)
        with col1:
            sim_building = st.selectbox("选择建筑", BUILDING_OPTIONS, key="sim_bld")
            sim_month = st.slider("月份", 1, 12, 7, key="sim_month")
        with col2:
            sim_dow = st.slider("星期", 0, 6, 3, key="sim_dow")
            preset = st.selectbox("场景预设", list(SCENE_PRESETS.keys()))

        base_multiplier = SCENE_PRESETS[preset]
        multiplier = st.slider("负荷系数微调", 0.5, 1.5, base_multiplier, 0.05,
                               help="<1.0 = 节能场景，>1.0 = 高峰场景")

        if st.button("🔬 运行模拟", type="primary"):
            base_history = DEFAULT_HISTORY_VALUES.copy()
            scene_history = {k: v * multiplier for k, v in DEFAULT_HISTORY_VALUES.items()}

            base_params_list = [{'hour': h, 'month': sim_month, 'day_of_week': sim_dow,
                                 'building_id': sim_building, 'meter': 0, **base_history} for h in range(24)]
            scene_params_list = [{'hour': h, 'month': sim_month, 'day_of_week': sim_dow,
                                  'building_id': sim_building, 'meter': 0, **scene_history} for h in range(24)]

            base_preds = model.batch_predict(base_params_list)
            scene_preds = model.batch_predict(scene_params_list)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(24)), y=base_preds, mode='lines+markers',
                                     line=dict(color='#8892a4', width=2), name='基准'))
            fig.add_trace(go.Scatter(x=list(range(24)), y=scene_preds, mode='lines+markers',
                                     line=dict(color='#00e676', width=2.5), name=f'场景 (×{multiplier:.2f})'))
            fig.update_layout(title=f"场景模拟对比 — {sim_building}", xaxis_title="小时", yaxis_title="能耗 (kWh)",
                              template='plotly_dark', height=450, paper_bgcolor='#0f1117',
                              plot_bgcolor='#0f1117', font=dict(color='#e0e0e0'))
            st.plotly_chart(fig, width='stretch')

            delta_total = sum(scene_preds) - sum(base_preds)
            c1, c2, c3 = st.columns(3)
            c1.metric("基准日总能耗", f"{sum(base_preds):,.0f} kWh")
            c2.metric("场景日总能耗", f"{sum(scene_preds):,.0f} kWh", delta=f"{delta_total:+,.0f} kWh")
            c3.metric("碳排放变化", f"{delta_total * 0.5:+,.0f} kg CO₂")

# ====== Tab 2: 碳排放看板 ======
with tab2:
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

# ====== Tab 3: 节能评估 ======
with tab3:
    st.markdown("## 💡 节能潜力评估")
    c1, c2, c3 = st.columns(3)
    c1.metric("可节约电量", "1,197 万 kWh", delta="-5.96%")
    c2.metric("CO₂ 减排量", "5,985 吨")
    c3.metric("超标建筑", "阜新清禾门区政府")

    st.markdown("#### 🏢 能耗超标 Top 10 建筑")
    top_data = {
        '建筑名称': [
            'Fuxin Qinghemen District Government', 'Benxi Central Hospital',
            'Shenyang Agricultural University Library', 'Building 7',
            "Benxi Municipal People's Government Office Building",
            'Dalian Jiaotong University Library', 'Dalian Transportation Bureau Pikou Port Building',
            'No.3 Box Transformer', "Changtu County People's Court Office Building",
            'Provincial Government No.2 Courtyard Complex'
        ],
        '超标电量 (kWh)': [1314178, 1202736, 1155951, 711219, 321365, 304324, 289588, 182692, 154541, 151254]
    }
    st.dataframe(pd.DataFrame(top_data), width='stretch', hide_index=True)

    sh_path = os.path.join(DATA_DIR, 'saving_by_hour.csv')
    if os.path.exists(sh_path):
        df = pd.read_csv(sh_path)
        fig = px.bar(df, x='Hour', y='Average Saveable Energy by Hour of Day (kWh)',
                     color='Average Saveable Energy by Hour of Day (kWh)',
                     color_continuous_scale=['#00e676', '#1b5e20'])
        fig.update_layout(template='plotly_dark', paper_bgcolor='#0f1117', plot_bgcolor='#0f1117',
                          font=dict(color='#e0e0e0'), height=350)
        st.plotly_chart(fig, width='stretch')

    st.success("高峰时段（10-11时、14-16时）建议优化空调设定与照明控制，重点关注阜新清禾门区政府、本溪中心医院等超标严重建筑。")

# ====== Tab 4: 模型解释 ======
with tab4:
    st.markdown("## 🔍 模型解释")
    shap_data = {
        '特征': ['meter_reading_rolling_mean_3', 'meter_reading_lag_2',
                 'meter_reading_lag_1', 'meter_reading_rolling_std_3',
                 'meter_reading_rolling_mean_6', 'meter_reading_lag_3',
                 'meter_reading_rolling_mean_12', 'meter_reading_rolling_mean_24'],
        'SHAP重要性': [1.553, 0.385, 0.262, 0.134, 0.082, 0.068, 0.051, 0.043]
    }
    df_shap = pd.DataFrame(shap_data)
    fig = go.Figure(go.Bar(x=df_shap['SHAP重要性'], y=df_shap['特征'], orientation='h',
                           marker=dict(color='#00e676'),
                           text=df_shap['SHAP重要性'].round(3), textposition='outside',
                           textfont=dict(color='#e0e0e0')))
    fig.update_layout(title="SHAP 特征重要性排名", xaxis_title="平均 |SHAP|",
                      yaxis=dict(autorange='reversed'), template='plotly_dark', height=400,
                      paper_bgcolor='#0f1117', plot_bgcolor='#0f1117', font=dict(color='#e0e0e0'))
    st.plotly_chart(fig, width='stretch')
    st.markdown("""
    <div style="background:#1a1d27;border-left:3px solid #00e676;padding:12px 16px;margin-top:8px;border-radius:2px;">
    <strong style="color:#ffffff;">特征解读</strong><br>
    <span style="color:#cccccc;">近3小时滚动均值（rolling_mean_3）对预测影响最大（SHAP=1.553），滞后2小时和滞后1小时能耗次之。模型强依赖历史负荷模式。</span>
    </div>
    """, unsafe_allow_html=True)

# ====== Tab 5: 报告导出 ======
with tab5:
    st.markdown("## 📋 报告导出")
    if 'last_prediction' in st.session_state:
        pred_data = st.session_state['last_prediction']
        report_df = pd.DataFrame({'小时': range(24), '预测能耗 (kWh)': pred_data['curve']})
        st.dataframe(report_df, width='stretch', hide_index=True)
        csv = report_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{pred_data["building"]}_forecast.csv" style="color:#00e676;text-decoration:none;">⬇️ 下载CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("#### 预测摘要")
        st.json({
            'building': pred_data['building'],
            'peak_hour': int(report_df.loc[report_df['预测能耗 (kWh)'].idxmax(), '小时']),
            'peak_value': float(report_df['预测能耗 (kWh)'].max()),
            'total_daily': float(report_df['预测能耗 (kWh)'].sum()),
            'carbon_estimate': float(report_df['预测能耗 (kWh)'].sum() * 0.5)
        })
    else:
        st.info("请先进行单点预测，生成预测数据后可导出报告。")
