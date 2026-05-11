"""节能评估标签页"""
import streamlit as st
import pandas as pd
import plotly.express as px
import os
from config import DATA_DIR


def render_energy_saving_tab():
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
    df_top = pd.DataFrame(top_data)
    st.dataframe(df_top, width='stretch', hide_index=True)

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
