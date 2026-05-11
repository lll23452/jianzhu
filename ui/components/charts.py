"""Plotly 图表工厂 — 统一暗色主题"""
import plotly.graph_objects as go
import plotly.express as px


def dark_layout(fig, **kwargs):
    defaults = {
        'template': 'plotly_dark',
        'paper_bgcolor': '#0f1117',
        'plot_bgcolor': '#0f1117',
        'font': dict(color='#e0e0e0', family='Work Sans'),
    }
    defaults.update(kwargs)
    fig.update_layout(**defaults)
    return fig


def make_24h_curve(values, name='能耗', color='#00e676', fill=True):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(24)), y=values, mode='lines+markers',
        line=dict(color=color, width=2), name=name,
        **(dict(fill='tozeroy', fillcolor='rgba(0,230,118,0.08)') if fill else {})
    ))
    return dark_layout(fig, xaxis_title='小时', yaxis_title='能耗 (kWh)', height=400)


def make_comparison_curves(curves, title="对比"):
    fig = go.Figure()
    for name, (values, color) in curves.items():
        fig.add_trace(go.Scatter(
            x=list(range(24)), y=values, mode='lines+markers',
            line=dict(color=color, width=2), name=name
        ))
    return dark_layout(fig, title=title, xaxis_title='小时', yaxis_title='能耗 (kWh)', height=450)


def make_shap_bar(features, values, color='#00e676'):
    fig = go.Figure(go.Bar(
        x=values, y=features, orientation='h',
        marker=dict(color=color),
        text=[f'{v:.3f}' for v in values], textposition='outside',
        textfont=dict(color='#e0e0e0')
    ))
    return dark_layout(fig, title='SHAP 特征重要性', yaxis=dict(autorange='reversed'), height=400)


def make_carbon_bars(df, x_col, y_col, color_col=None):
    fig = px.bar(df, x=x_col, y=y_col, color=color_col or y_col,
                 color_continuous_scale=['#ff9100', '#ff5252'])
    return dark_layout(fig, height=400)
