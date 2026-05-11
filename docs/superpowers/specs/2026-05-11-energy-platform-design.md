# 公共建筑能耗预测与诊断平台 — 设计规格

## 概述

全新重写现有能耗预测平台，保持前后端分离架构（Flask API + Streamlit UI），复用已有 PSO-XGBoost 模型与编码器，补全数据预处理（线性插值/分位数截断）、增强场景模拟器、整合碳排放核算与 SHAP 可解释性分析。

## 技术栈

Python / XGBoost / PSO / SHAP / Streamlit / Flask / ONNX Runtime / Plotly / Pandas / NumPy / scikit-learn

## 项目结构

```
PythonProject9/
├── config.py                  # 全局配置
├── data_pipeline/
│   ├── preprocessor.py        # 线性插值 + 分位数截断
│   ├── feature_engineer.py    # 时间编码/滞后滚动特征/类别编码
│   └── data_loader.py         # 数据加载
├── models/
│   ├── pso_optimizer.py       # PSO 超参优化
│   ├── trainer.py             # XGBoost 训练 (RMSE/MAE/sMAPE/R²)
│   ├── converter.py           # XGBoost → ONNX
│   └── inference.py           # ONNX Runtime 推理封装
├── analysis/
│   ├── carbon.py              # 碳排放核算（0.5 kg/kWh）
│   ├── energy_saving.py       # 节能潜力评估
│   └── explainer.py           # SHAP 全局/局部可解释性
├── api/
│   ├── app.py                 # Flask 主应用
│   └── routes.py              # API 路由
├── ui/
│   ├── app.py                 # Streamlit 入口
│   ├── pages/                 # 4 种模式页面
│   ├── tabs/                  # 5 个分析标签页
│   └── components/            # 图表/控件复用
├── saved_models/              # 模型与编码器文件
├── data/                      # 数据文件
├── results/                   # 输出报告
└── requirements.txt
```

## 模块设计

### 1. config.py — 全局配置

- 路径常量（MODEL_DIR、DATA_DIR、RESULTS_DIR）
- 碳因子（0.5 kg CO₂/kWh）
- 特征默认值（历史负荷 param defaults）
- 特征名称列表
- PSO 默认参数范围

### 2. data_pipeline/preprocessor.py — 数据预处理

- **线性插值**：对缺失的 meter_reading 按时间序列插值填充
- **分位数截断**：对能耗值做 1%/99% 分位数截断，剔除极端离群值
- **异常值清洗**：IQR 或 Z-score 方法标记异常

### 3. data_pipeline/feature_engineer.py — 特征工程

- **时间周期编码**：hour_sin/cos、month_sin/cos、weekend 标识
- **滞后特征**：lag_1/2/3/24（前 n 小时能耗）
- **滚动统计**：rolling_mean/std（3/6/12/24 窗口）
- **类别编码**：building_id、meter 的 LabelEncoder

### 4. data_pipeline/data_loader.py — 数据加载

- 从 output.xlsx 或 CSV 加载原始数据
- 调用 preprocessor 和 feature_engineer 完成全部特征构建
- 返回训练就绪的 (X, y)

### 5. models/pso_optimizer.py — PSO 优化

- 粒子群算法搜索 XGBoost 8 个超参数
- 适应度函数 = 验证集 sMAPE
- 支持早停和收敛监控

### 6. models/trainer.py — 训练与评估

- 数据划分（train/val/test）
- StandardScaler 标准化
- XGBoost 训练（使用 PSO 最优参数）
- 评估指标：RMSE / MAE / sMAPE / R²
- 保存模型、scaler、encoder

### 7. models/converter.py — 模型转换

- XGBoost JSON → ONNX 格式
- 验证 ONNX 输入输出形状一致性

### 8. models/inference.py — 推理封装

- 加载 ONNX session + scaler + encoders
- build_features() 方法：从原始参数构建标准化特征向量
- predict() / batch_predict() 方法
- log1p → expm1 逆变换

### 9. analysis/carbon.py — 碳排放核算

- 逐记录碳排 = meter_reading × 0.5 kg
- 按建筑/按小时聚合统计
- 导出 carbon_by_building.csv / carbon_by_hour.csv

### 10. analysis/energy_saving.py — 节能评估

- 超标电量 = max(0, actual - predicted)
- 可节约电量 = sum(超标) × 系数
- 按建筑排名 / 按小时分布
- 节能策略推荐文本

### 11. analysis/explainer.py — SHAP 解释

- TreeExplainer 计算全局 SHAP 值
- 特征重要性排序与可视化数据
- 单个样本的局部解释（force plot 数据）

### 12. api/routes.py — API 路由

| 路由 | 方法 | 功能 |
|------|------|------|
| /health | GET | 健康检查 |
| /predict | POST | 单点预测 |
| /batch_predict | POST | 批量预测（24h 曲线） |
| /buildings | GET | 返回可用建筑列表 |
| /shap/global | GET | 全局特征重要性 |
| /shap/local | POST | 单个样本 SHAP 解释 |
| /carbon/summary | GET | 碳排放汇总 |
| /saving/summary | GET | 节能潜力汇总 |

### 13. UI 设计方向（由 frontend-design 技能驱动）

**美学定位：工业控制台 × 生态意识**

将能源监控的严肃性与绿色可持续的现代感融合。深色为主的数据密集型监控界面，辅以高亮的能效色彩。

- **字体**：标题用 `Teko`（窄体、科技感、数字显示风格），正文用 `Work Sans`（清晰、现代、高可读性），数据指标用 `JetBrains Mono`（等宽数字对齐）
- **配色**：
  - 主背景：深炭灰 `#0f1117` / 面板 `#1a1d27`
  - 主强调色：电光绿 `#00e676`（能耗/预测数据）
  - 辅强调色：琥珀 `#ff9100`（碳排放相关）
  - 警告/高耗能：珊瑚红 `#ff5252`
  - 文本：灰白 `#e0e0e0` / 次级 `#8892a4`
- **空间**：三栏仪表盘布局，顶部 KPI 条，不对称图表网格，侧边栏收窄为控制台面板
- **动效**：数值跳动计数器动画、图表加载骨架屏、标签切换淡入
- **氛围**：微妙的网格线背景、指标卡发光边框、数据流线条装饰
- **反AI风格**：不用紫色渐变、不用 Inter/Roboto 字体、不用白底圆角卡片

### 14. ui/app.py — Streamlit 入口

- 注入自定义 CSS（字体 via Google Fonts CDN、CSS 变量、暗色主题覆盖）
- 页面配置（dark mode、wide layout）
- 顶部 KPI 指标条（实时总能耗/碳排/节约量/R²）
- 侧边栏：控制台面板，4 种模式切换，API 状态指示灯
- 5 个标签页组织分析内容

### 15. ui/pages/ — 4 种模式

- **prediction.py**：单点预测，建筑/时间选择器，历史负荷参数面板，24h 预测曲线（电光绿渐变填充）
- **comparison.py**：多建筑（≤3）选择，同时间 24h 曲线叠加对比（不同强调色区分）
- **replay.py**：日期范围选择，加载 test_samples.csv，实际 vs 预测双线图，误差指标卡（MAE/RMSE/MAPE）
- **simulator.py**：场景预设按钮（节能/高峰/自定义），lag/rolling 百分比滑块，基准 vs 场景双线对比图

### 16. ui/tabs/ — 5 个标签页

- **carbon_dashboard.py**：碳排放指标卡 + 建筑碳排柱状图 + 小时碳排放热力曲线
- **energy_saving.py**：可节约电量/CO₂减排环形指标 + Top10 表格（琥珀高亮） + 小时节能分布 + 策略推荐卡片
- **model_explanation.py**：SHAP 特征重要性水平条形图 + 局部解释力图数据 + 特征解读卡片
- **report_export.py**：预测数据表 + CSV/JSON 下载按钮

### 17. ui/components/ — 复用组件

- **charts.py**：Plotly 图表工厂（暗色模板、统一强调色、网格线淡化）
- **widgets.py**：复用控件（建筑搜索下拉、时间选择器、历史参数可折叠面板、场景预设按钮组、KPI 指标卡）

## 数据流

```
output.xlsx → DataLoader → Preprocessor(插值+截断) → FeatureEngineer(编码+滞后+滚动)
  → Trainer(PSO + XGBoost) → Scaler + Encoder + Model(JSON)
  → Converter → ONNX
  → Inference.load() → Flask API
  → Streamlit UI → 用户
```

## 场景模拟器设计

由于当前模型不包含温度/人员密度特征，模拟器通过 **间接调节历史负荷参数** 实现场景模拟：

- **节能场景**：将 lag/rolling 值乘以 0.7~0.9（模拟设备关停或低负荷运行后的累积效应）
- **高峰场景**：将 lag/rolling 值乘以 1.1~1.3（模拟极端冷热天气或高人员密度导致的高负荷）
- **自定义场景**：用户手动调节各历史参数
- 输出：场景能耗 24h 曲线 vs 基准 24h 曲线对比

## 关键决策

- **log1p 变换**：训练时对 y 做 log1p，推理时 expm1 还原（已在现有模型中使用）
- **碳因子 0.5**：沿用 0.5 kg CO₂/kWh（中国电网平均值）
- **PSO 参数**：沿用现有 8 参数搜索空间
- **ONNX 推理**：不做改动，直接加载现有 ONNX 模型

## 不包含的内容

- 模型重训练（使用现有 saved_models）
- 新特征（温度/湿度/人员密度）加入
- 用户登录/权限系统
- 实时数据接入
