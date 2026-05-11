"""Streamlit 主入口 — 工业控制台 × 生态意识暗色主题"""
import streamlit as st
import requests

st.set_page_config(
    page_title="建筑能耗预测与诊断平台",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Teko:wght@400;500;600&family=Work+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">

<style>
:root {
    --bg-root: #0f1117;
    --bg-surface: #1a1d27;
    --bg-elevated: #242836;
    --accent-energy: #00e676;
    --accent-carbon: #ff9100;
    --accent-danger: #ff5252;
    --text-primary: #cccccc;
    --text-secondary: #999999;
    --border-subtle: #2a2e3a;
    --font-display: 'Teko', sans-serif;
    --font-body: 'Work Sans', sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
}

.stApp {
    background-color: var(--bg-root);
}

[data-testid="stSidebar"] {
    background-color: var(--bg-surface);
    border-right: 1px solid var(--border-subtle);
}

/* 下拉选择框选项 - 黑色文字 */
[data-baseweb="select"] [role="listbox"] li,
[data-baseweb="menu"] [role="option"],
ul[role="listbox"] li,
[data-baseweb="select"] [role="option"] {
    color: #000000 !important;
}

h1 {
    font-family: var(--font-display) !important;
    font-size: 2.5rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    color: #00e676 !important;
    text-transform: uppercase;
}

h2, h3 {
    font-family: var(--font-display) !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
    color: #ffffff !important;
}

[data-testid="stMetric"] {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    padding: 12px 16px;
}

[data-testid="stMetricValue"] {
    font-family: var(--font-mono) !important;
    font-size: 1.6rem !important;
    color: #00e676 !important;
}

.stButton > button {
    font-family: var(--font-display) !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase;
    background: var(--accent-energy);
    color: #0f1117;
    border: none;
    border-radius: 2px;
    padding: 8px 24px;
    transition: all 0.2s;
}

.stButton > button:hover {
    box-shadow: 0 0 20px rgba(0, 230, 118, 0.3);
}

[data-testid="stDataFrame"] {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
}

.stTabs [aria-selected="true"] {
    color: #00e676 !important;
}

[data-testid="stExpander"] {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

API_BASE = "http://localhost:5000"


@st.cache_resource
def get_building_list():
    try:
        resp = requests.get(f"{API_BASE}/buildings", timeout=5)
        if resp.status_code == 200:
            return resp.json().get('buildings', [])
    except Exception:
        pass
    return ['NDRC Building', "People's Hall", 'Benxi Central Hospital']


BUILDING_OPTIONS = get_building_list()

# ---- 侧边栏控制台 ----
with st.sidebar:
    st.markdown("## ⚡ 控制台")

    try:
        hr = requests.get(f"{API_BASE}/health", timeout=3)
        if hr.status_code == 200:
            st.markdown('<span style="color:#00e676;font-family:JetBrains Mono,monospace;">● API ONLINE</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#ff5252">● API OFFLINE</span>', unsafe_allow_html=True)
    except Exception:
        st.markdown('<span style="color:#ff5252">● API OFFLINE</span>', unsafe_allow_html=True)

    st.markdown("---")
    mode = st.radio("选择模式", ["单点预测", "多建筑对比", "历史数据回放", "场景模拟器"])

# ---- 顶部 KPI 条 ----
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

# ---- 导入页面和标签页 ----
from ui.modes.prediction import render_prediction_page
from ui.modes.comparison import render_comparison_page
from ui.modes.replay import render_replay_page
from ui.modes.simulator import render_simulator_page
from ui.tabs.carbon_dashboard import render_carbon_tab
from ui.tabs.energy_saving import render_energy_saving_tab
from ui.tabs.model_explanation import render_explanation_tab
from ui.tabs.report_export import render_report_tab

METER_MAPPING = {'electricity': 0}

# ---- 标签页 ----
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 预测分析", "🌍 碳排放看板", "💡 节能评估", "🔍 模型解释", "📋 报告导出"
])

with tab1:
    if mode == "单点预测":
        render_prediction_page(BUILDING_OPTIONS, METER_MAPPING)
    elif mode == "多建筑对比":
        render_comparison_page(BUILDING_OPTIONS, METER_MAPPING)
    elif mode == "历史数据回放":
        render_replay_page()
    elif mode == "场景模拟器":
        render_simulator_page(BUILDING_OPTIONS, METER_MAPPING)

with tab2:
    render_carbon_tab()

with tab3:
    render_energy_saving_tab()

with tab4:
    render_explanation_tab()

with tab5:
    render_report_tab()
