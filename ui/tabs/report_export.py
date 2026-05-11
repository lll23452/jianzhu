"""报告导出标签页"""
import streamlit as st
import pandas as pd
import base64


def get_download_link(df, filename="report.csv", label="⬇️ 下载CSV"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="color:#00e676;text-decoration:none;">{label}</a>'
    return href


def render_report_tab():
    st.markdown("## 📋 报告导出")

    if 'last_prediction' in st.session_state:
        pred_data = st.session_state['last_prediction']
        report_df = pd.DataFrame({
            '小时': range(24),
            '预测能耗 (kWh)': pred_data['curve']
        })
        st.dataframe(report_df, width='stretch', hide_index=True)
        st.markdown(get_download_link(report_df, f"{pred_data['building']}_forecast.csv"), unsafe_allow_html=True)

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
