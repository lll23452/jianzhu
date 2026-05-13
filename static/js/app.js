// ====== 全局状态 ======
let currentTab = 'analysis';
let currentMode = 'predict';
let lastPrediction = null;
const DH = {'meter_reading_lag_1':50000,'meter_reading_lag_2':50000,'meter_reading_lag_3':50000,'meter_reading_lag_24':48000,'meter_reading_rolling_mean_3':50000,'meter_reading_rolling_mean_6':49000,'meter_reading_rolling_mean_12':48500,'meter_reading_rolling_mean_24':48000,'meter_reading_rolling_std_3':2000,'meter_reading_rolling_std_6':2500,'meter_reading_rolling_std_12':3000,'meter_reading_rolling_std_24':3500};

// ====== 初始化 ======
document.addEventListener('DOMContentLoaded', () => {
  setupTabs();
  setupModes();
  setupDashboard();
  loadBuildingList();
});

// ====== API 辅助 ======
async function api(url, opts={}) {
  const resp = await fetch(url, opts);
  return resp.json();
}
const plotLayout = (extra={}) => ({
  template:'plotly_dark', paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
  font:{color:'#8899b4'}, margin:{t:20,r:10,b:40,l:50}, height:400,
  xaxis:{gridcolor:'rgba(255,255,255,0.04)'}, yaxis:{gridcolor:'rgba(255,255,255,0.04)'},
  ...extra
});

// ====== Tab 切换 ======
function setupTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentTab = btn.dataset.tab;
      renderTab(currentTab);
    });
  });
}

// ====== 模式切换 ======
function setupModes() {
  const setMode = (mode) => {
    currentMode = mode;
    document.querySelectorAll('.mode-btn, .bn-btn').forEach(b => b.classList.toggle('active', b.dataset.mode === mode));
    if (currentTab === 'analysis') renderPanel();
  };
  document.querySelectorAll('.mode-btn').forEach(btn => btn.addEventListener('click', () => setMode(btn.dataset.mode)));
  document.querySelectorAll('.bn-btn').forEach(btn => btn.addEventListener('click', () => setMode(btn.dataset.mode)));
}

// ====== 渲染 ======
function renderPanel() {
  const p = document.getElementById('panelContent');
  p.innerHTML = '<div class="loading-spinner" style="text-align:center;padding:2rem;color:var(--text-secondary);">加载中...</div>';
  switch(currentMode) {
    case 'predict': renderPredict(p); break;
    case 'compare': renderComparison(p); break;
    case 'replay': renderReplay(p); break;
    case 'simulate': renderSimulator(p); break;
    case 'import': renderImport(p); break;
  }
}

function renderTab(tab) {
  currentTab = tab;
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
  if (tab === 'analysis') renderPanel();
  else { document.getElementById('kpiRow').classList.add('hidden'); renderStaticTab(tab); }
}

function showKPI(show) { document.getElementById('kpiRow').classList.toggle('hidden', !show); }

// ====== 加载建筑列表 ======
async function loadBuildingList() {
  try {
    const data = await api('/api/buildings');
    window._buildings = data.buildings;
  } catch(e) { window._buildings = ['NDRC Building']; }
  if (currentMode === 'predict') renderPanel();
}

function fillBuildingSelect(id, multi=false) {
  const sel = document.getElementById(id);
  if (!sel || !window._buildings) return;
  sel.innerHTML = '';
  window._buildings.forEach((b,i) => {
    const o = document.createElement('option'); o.value = b; o.textContent = b;
    if (multi && i < 2) o.selected = true;
    sel.appendChild(o);
  });
}

// =============================================
//  1. 单点预测
// =============================================
function renderPredict(p) {
  showKPI(true);
  p.innerHTML = `
    <h2 style="font-family:var(--font-display);color:var(--glow-green);margin-bottom:1rem;">📈 单点预测</h2>
    <div class="form-row">
      <div class="form-group"><label>建筑</label><select id="predBuilding"></select></div>
      <div class="form-group"><label>小时</label><input type="number" id="predHour" value="14" min="0" max="23"></div>
      <div class="form-group"><label>月份</label><input type="number" id="predMonth" value="7" min="1" max="12"></div>
      <div class="form-group"><label>星期(0-6)</label><input type="number" id="predDow" value="3" min="0" max="6"></div>
    </div>
    <button class="btn btn-primary" id="predBtn">🚀 执行预测</button>
    <div id="predResult" style="margin-top:1rem;"></div>
    <div id="predChart" class="chart-container"></div>
  `;
  fillBuildingSelect('predBuilding');
  document.getElementById('predBtn').addEventListener('click', doPredict);
}

async function doPredict() {
  const params = {
    hour: +document.getElementById('predHour').value,
    month: +document.getElementById('predMonth').value,
    day_of_week: +document.getElementById('predDow').value,
    building_id: document.getElementById('predBuilding').value, meter: 0, ...DH
  };
  const data = await api('/api/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({params})});
  document.getElementById('predResult').innerHTML = `
    <div class="kpi-row" style="margin-top:1rem">
      <div class="kpi-card"><span class="kpi-label">预测能耗</span><span class="kpi-value" style="--glow:#00f5a0">${data.prediction.toLocaleString(undefined,{maximumFractionDigits:0})} kWh</span></div>
      <div class="kpi-card"><span class="kpi-label">碳排放</span><span class="kpi-value" style="--glow:#b388ff">${(data.prediction*0.5).toLocaleString(undefined,{maximumFractionDigits:0})} kg</span></div>
    </div>`;
  const pl = Array.from({length:24},(_,h)=>({...params, hour:h}));
  const batch = await api('/api/batch_predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({params_list:pl})});
  Plotly.newPlot('predChart', [{x:[...Array(24).keys()], y:batch.predictions, type:'scatter', mode:'lines+markers', line:{color:'#00f5a0',width:2}, fill:'tozeroy', fillcolor:'rgba(0,245,160,0.08)', name:'预测'}], plotLayout({title:'24小时能耗预测'}), {responsive:true});
  lastPrediction = {building:params.building_id, curve:batch.predictions, prediction:data.prediction};
}

// =============================================
//  2. 多建筑对比
// =============================================
function renderComparison(p) {
  showKPI(true);
  p.innerHTML = `
    <h2 style="font-family:var(--font-display);color:var(--glow-green);margin-bottom:1rem;">📊 多建筑对比</h2>
    <div class="form-group"><label>选择建筑（Ctrl+多选，最多3个）</label><select id="cmpBuildings" multiple style="height:120px;"></select></div>
    <button class="btn btn-primary" id="cmpBtn" style="margin-top:0.75rem">📊 生成对比</button>
    <div id="cmpChart" class="chart-container" style="margin-top:1rem"></div>
  `;
  fillBuildingSelect('cmpBuildings', true);
  document.getElementById('cmpBtn').addEventListener('click', doComparison);
}

async function doComparison() {
  const sel = document.getElementById('cmpBuildings');
  const selected = Array.from(sel.selectedOptions).slice(0,3).map(o=>o.value);
  const colors = ['#00f5a0','#b388ff','#ff5370'];
  const traces = [];
  for (let i=0; i<selected.length; i++) {
    const pl = Array.from({length:24},(_,h)=>({hour:h,month:7,day_of_week:3,building_id:selected[i],meter:0,...DH}));
    const data = await api('/api/batch_predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({params_list:pl})});
    traces.push({x:[...Array(24).keys()], y:data.predictions, type:'scatter', mode:'lines+markers', line:{color:colors[i],width:2}, name:selected[i]});
  }
  Plotly.newPlot('cmpChart', traces, plotLayout({title:'24h建筑能耗对比'}), {responsive:true});
}

// =============================================
//  3. 历史回放
// =============================================
function renderReplay(p) {
  showKPI(true);
  p.innerHTML = `
    <h2 style="font-family:var(--font-display);color:var(--glow-green);margin-bottom:1rem;">📅 历史数据回放</h2>
    <p class="text-secondary">从测试样本中加载历史预测与实际值进行对比</p>
    <button class="btn btn-primary" id="replayBtn">📅 加载回放数据</button>
    <div id="replayChart" class="chart-container" style="margin-top:1rem"></div>
    <div id="replayMetrics" class="kpi-row"></div>
  `;
  document.getElementById('replayBtn').addEventListener('click', doReplay);
}

async function doReplay() {
  try {
    const resp = await fetch('/api/samples');
    const data = await resp.json();
    if (!data.timestamps || data.timestamps.length === 0) {
      document.getElementById('replayChart').innerHTML = '<p class="text-secondary text-center" style="padding:2rem;">暂无回放数据</p>';
      return;
    }
    Plotly.newPlot('replayChart', [
      {x:data.timestamps, y:data.actuals, type:'scatter', mode:'lines', line:{color:'#00f5a0',width:2}, name:'实际值'},
      {x:data.timestamps, y:data.predictions, type:'scatter', mode:'lines', line:{color:'#ff9100',width:2}, name:'预测值'}
    ], plotLayout({title:'历史能耗回放', xaxis:{title:'时间'}, yaxis:{title:'kWh'}}), {responsive:true});
    const err = data.actuals.map((a,i) => a - data.predictions[i]);
    const mae = err.reduce((s,v) => s+Math.abs(v),0)/err.length;
    const rmse = Math.sqrt(err.reduce((s,v)=>s+v*v,0)/err.length);
    document.getElementById('replayMetrics').innerHTML = `
      <div class="kpi-card"><span class="kpi-label">MAE</span><span class="kpi-value" style="--glow:#00f5a0">${mae.toLocaleString(undefined,{maximumFractionDigits:0})} kWh</span></div>
      <div class="kpi-card"><span class="kpi-label">RMSE</span><span class="kpi-value" style="--glow:#b388ff">${rmse.toLocaleString(undefined,{maximumFractionDigits:0})} kWh</span></div>
      <div class="kpi-card"><span class="kpi-label">记录数</span><span class="kpi-value" style="--glow:#ff9100">${err.length}</span></div>
    `;
  } catch(e) {
    document.getElementById('replayChart').innerHTML = '<p class="text-secondary text-center" style="padding:2rem;">加载失败，请检查数据文件</p>';
  }
}

// =============================================
//  4. 场景模拟
// =============================================
function renderSimulator(p) {
  showKPI(true);
  p.innerHTML = `
    <h2 style="font-family:var(--font-display);color:var(--glow-green);margin-bottom:1rem;">🔬 场景模拟器</h2>
    <p class="text-secondary" style="margin-bottom:1rem;">通过调节历史负荷参数模拟不同运行条件</p>
    <div class="form-row">
      <div class="form-group"><label>建筑</label><select id="simBuilding"></select></div>
      <div class="form-group"><label>月份</label><input type="number" id="simMonth" value="7" min="1" max="12"></div>
      <div class="form-group"><label>星期(0-6)</label><input type="number" id="simDow" value="3" min="0" max="6"></div>
      <div class="form-group"><label>场景预设</label><select id="simPreset"><option value="1.0">基准场景</option><option value="0.75">节能场景</option><option value="1.25">高峰场景</option><option value="1.5">极端高峰</option></select></div>
    </div>
    <div class="form-group"><label>负荷系数: <span id="simMultLabel">1.00</span></label><input type="range" id="simMult" min="0.5" max="1.5" step="0.05" value="1.0" style="width:100%;accent-color:var(--glow-green);"></div>
    <button class="btn btn-primary" id="simBtn">🔬 运行模拟</button>
    <div id="simChart" class="chart-container" style="margin-top:1rem"></div>
    <div id="simMetrics" class="kpi-row"></div>
  `;
  fillBuildingSelect('simBuilding');
  document.getElementById('simPreset').addEventListener('change', e => { document.getElementById('simMult').value=e.target.value; document.getElementById('simMultLabel').textContent=parseFloat(e.target.value).toFixed(2); });
  document.getElementById('simMult').addEventListener('input', e => { document.getElementById('simMultLabel').textContent=parseFloat(e.target.value).toFixed(2); });
  document.getElementById('simBtn').addEventListener('click', doSimulation);
}

async function doSimulation() {
  const bld = document.getElementById('simBuilding').value;
  const month = +document.getElementById('simMonth').value;
  const dow = +document.getElementById('simDow').value;
  const mult = +document.getElementById('simMult').value;

  const baseList = Array.from({length:24},(_,h)=>({hour:h,month,day_of_week:dow,building_id:bld,meter:0,...DH}));
  const sceneList = Array.from({length:24},(_,h)=>({hour:h,month,day_of_week:dow,building_id:bld,meter:0,...Object.fromEntries(Object.entries(DH).map(([k,v])=>[k,v*mult]))}));

  const [base, scene] = await Promise.all([
    api('/api/batch_predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({params_list:baseList})}),
    api('/api/batch_predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({params_list:sceneList})})
  ]);

  Plotly.newPlot('simChart', [
    {x:[...Array(24).keys()], y:base.predictions, type:'scatter', mode:'lines', line:{color:'#8899b4',width:2}, name:'基准'},
    {x:[...Array(24).keys()], y:scene.predictions, type:'scatter', mode:'lines', line:{color:'#00f5a0',width:2.5}, name:`场景 ×${mult}`}
  ], plotLayout({title:`场景模拟 — ${bld}`}), {responsive:true});

  const tb = base.predictions.reduce((a,b)=>a+b,0);
  const ts = scene.predictions.reduce((a,b)=>a+b,0);
  document.getElementById('simMetrics').innerHTML = `
    <div class="kpi-card"><span class="kpi-label">基准日总能耗</span><span class="kpi-value" style="--glow:#8899b4">${tb.toLocaleString(undefined,{maximumFractionDigits:0})} kWh</span></div>
    <div class="kpi-card"><span class="kpi-label">场景日总能耗</span><span class="kpi-value" style="--glow:#00f5a0">${ts.toLocaleString(undefined,{maximumFractionDigits:0})} kWh</span></div>
    <div class="kpi-card"><span class="kpi-label">碳排放变化</span><span class="kpi-value" style="--glow:#b388ff">${((ts-tb)*0.5).toLocaleString(undefined,{maximumFractionDigits:0})} kg</span></div>
  `;
}

// =============================================
//  5. 在线导入
// =============================================
function renderImport(p) {
  showKPI(true);
  p.innerHTML = `
    <h2 style="font-family:var(--font-display);color:var(--glow-green);margin-bottom:1rem;">📂 在线导入预测</h2>
    <div class="upload-zone" id="uploadZone">
      <p style="color:var(--text-secondary);">点击或拖拽上传 CSV/Excel 文件</p>
      <p style="font-size:0.7rem;color:var(--text-secondary);margin-top:0.25rem;">需包含 building_id 列</p>
      <input type="file" id="fileInput" accept=".csv,.xlsx,.xls" style="display:none">
    </div>
    <div id="uploadStatus"></div>
    <div id="importResults" class="hidden"></div>
  `;
  const zone = document.getElementById('uploadZone');
  const input = document.getElementById('fileInput');
  zone.addEventListener('click', () => input.click());
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => { e.preventDefault(); zone.classList.remove('dragover'); handleFile(e.dataTransfer.files[0]); });
  input.addEventListener('change', e => { if (e.target.files[0]) handleFile(e.target.files[0]); });
}

async function handleFile(file) {
  document.getElementById('uploadStatus').innerHTML = '<div class="alert alert-success">上传中...</div>';
  const formData = new FormData();
  formData.append('file', file);
  try {
    const resp = await fetch('/api/upload', {method:'POST', body:formData});
    const data = await resp.json();
    if (data.error) { document.getElementById('uploadStatus').innerHTML = `<div class="alert alert-warning">${data.error}</div>`; return; }
    document.getElementById('uploadStatus').innerHTML = `<div class="alert alert-success">✅ 预测完成！${data.count} 条记录</div>`;
    showImportResults(data);
  } catch(e) {
    document.getElementById('uploadStatus').innerHTML = '<div class="alert alert-warning">上传失败</div>';
  }
}

function showImportResults(data) {
  const div = document.getElementById('importResults');
  div.classList.remove('hidden');
  div.innerHTML = `
    <div class="kpi-row">
      <div class="kpi-card"><span class="kpi-label">总能耗</span><span class="kpi-value" style="--glow:#00f5a0">${data.total_kwh.toLocaleString(undefined,{maximumFractionDigits:0})} kWh</span></div>
      <div class="kpi-card"><span class="kpi-label">总碳排放</span><span class="kpi-value" style="--glow:#b388ff">${data.total_carbon.toLocaleString(undefined,{maximumFractionDigits:0})} kg</span></div>
      <div class="kpi-card"><span class="kpi-label">平均值</span><span class="kpi-value" style="--glow:#00f5a0">${data.avg_kwh.toLocaleString(undefined,{maximumFractionDigits:0})} kWh</span></div>
      <div class="kpi-card"><span class="kpi-label">峰值</span><span class="kpi-value" style="--glow:#ff5370">${data.peak_kwh.toLocaleString(undefined,{maximumFractionDigits:0})} kWh</span></div>
    </div>
    <h3 class="section-title">🏢 按建筑汇总</h3>
    <div style="overflow-x:auto"><table><thead><tr><th>建筑</th><th>记录数</th><th>总能耗(kWh)</th><th>平均能耗(kWh)</th><th>总碳排放(kg)</th></tr></thead><tbody>
      ${data.by_building.map(b => `<tr><td>${b.building_id}</td><td>${b.count}</td><td>${b.total_kwh.toLocaleString(undefined,{maximumFractionDigits:0})}</td><td>${b.avg_kwh.toLocaleString(undefined,{maximumFractionDigits:0})}</td><td>${b.total_carbon.toLocaleString(undefined,{maximumFractionDigits:0})}</td></tr>`).join('')}
    </tbody></table></div>
    <div id="importBldChart" class="chart-container" style="margin-top:1rem"></div>
    ${data.by_hour ? `<div id="importHourChart" class="chart-container"></div>` : ''}
    <button class="btn btn-primary" onclick="downloadImportCSV()" style="margin-top:1rem">⬇️ 下载完整结果CSV</button>
  `;

  Plotly.newPlot('importBldChart', [{
    x: data.by_building.slice(0,10).map(b=>b.building_id),
    y: data.by_building.slice(0,10).map(b=>b.total_kwh),
    type:'bar', marker:{color:'#00f5a0'}
  }], plotLayout({title:'各建筑预测能耗 Top 10', yaxis:{title:'kWh'}}), {responsive:true});

  if (data.by_hour) {
    Plotly.newPlot('importHourChart', [{
      x: data.by_hour.map(h=>h.hour),
      y: data.by_hour.map(h=>h.predicted_kwh || h.avg_saveable_kwh || 0),
      type:'bar', marker:{color:'#b388ff'}
    }], plotLayout({title:'小时能耗分布', xaxis:{title:'小时'}, yaxis:{title:'kWh'}}), {responsive:true});
  }
  window._importData = data;
}

function downloadImportCSV() {
  if (!window._importData) return;
  let csv = 'building_id,hour,month,day_of_week,predicted_kwh,carbon_kg\n';
  window._importData.records.forEach(r => { csv += `${r.building_id},${r.hour||''},${r.month||''},${r.day_of_week||''},${r.predicted_kwh},${r.carbon_kg}\n`; });
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([csv],{type:'text/csv'})); a.download = 'prediction_results.csv'; a.click();
}

// =============================================
//  静态标签页
// =============================================
function renderStaticTab(tab) {
  const p = document.getElementById('panelContent');
  switch(tab) {
    case 'carbon': renderCarbon(p); break;
    case 'saving': renderSaving(p); break;
    case 'explain': renderExplain(p); break;
    case 'export': renderExport(p); break;
  }
}

// === 碳排放 ===
async function renderCarbon(p) {
  showKPI(false);
  p.innerHTML = '<div class="loading-spinner" style="text-align:center;padding:2rem;">加载中...</div>';
  const data = await api('/api/carbon');
  p.innerHTML = `
    <h2 style="font-family:var(--font-display);color:var(--glow-green);margin-bottom:1rem;">🌍 碳排放仪表盘</h2>
    <div class="kpi-row">
      <div class="kpi-card"><span class="kpi-label">总碳排放</span><span class="kpi-value" style="--glow:#b388ff">${(data.total_carbon/1e6).toFixed(2)} 万吨</span></div>
      <div class="kpi-card"><span class="kpi-label">总用电量</span><span class="kpi-value" style="--glow:#00f5a0">${(data.total_energy/1e6).toFixed(2)} 万 kWh</span></div>
      <div class="kpi-card"><span class="kpi-label">碳因子</span><span class="kpi-value" style="--glow:#ff9100">${data.carbon_factor} kg/kWh</span></div>
    </div>
    <div id="carbonBldChart" class="chart-container"></div>
    <div id="carbonHourChart" class="chart-container"></div>
  `;
  if (data.by_building.length) {
    Plotly.newPlot('carbonBldChart', [{
      x: data.by_building.slice(0,7).map(b=>b['Building ID']||b.building_id||'N/A'),
      y: data.by_building.slice(0,7).map(b=>b['kg CO2']||b.carbon_kg||0),
      type:'bar', marker:{color:'#b388ff'}
    }], plotLayout({title:'各建筑碳排放'}));
  }
  if (data.by_hour.length) {
    Plotly.newPlot('carbonHourChart', [{
      x: data.by_hour.map(h=>h['Hour']||h.hour),
      y: data.by_hour.map(h=>h['kg CO2 (average)']||h.carbon_kg_avg||0),
      type:'scatter', mode:'lines+markers', line:{color:'#ff9100',width:2}, fill:'tozeroy', fillcolor:'rgba(255,145,0,0.08)'
    }], plotLayout({title:'小时平均碳排放'}));
  }
}

// === 节能评估 ===
async function renderSaving(p) {
  showKPI(false);
  p.innerHTML = '<div class="loading-spinner" style="text-align:center;padding:2rem;">加载中...</div>';
  const data = await api('/api/saving');
  p.innerHTML = `
    <h2 style="font-family:var(--font-display);color:var(--glow-green);margin-bottom:1rem;">💡 节能潜力评估</h2>
    <div class="kpi-row">
      <div class="kpi-card"><span class="kpi-label">可节约电量</span><span class="kpi-value" style="--glow:#00f5a0">${(data.total_saving/1e4).toFixed(0)} 万 kWh</span></div>
      <div class="kpi-card"><span class="kpi-label">CO₂ 减排量</span><span class="kpi-value" style="--glow:#b388ff">${(data.co2_reduction/1e3).toFixed(0)} 吨</span></div>
      <div class="kpi-card"><span class="kpi-label">节约比例</span><span class="kpi-value" style="--glow:#ff5370">${data.percent_saving}%</span></div>
    </div>
    <h3 class="section-title">🏢 超标建筑 Top 10</h3>
    <table><thead><tr><th>#</th><th>建筑</th><th>超标电量(kWh)</th></tr></thead><tbody>
      ${data.top_buildings.map((b,i) => `<tr><td>${i+1}</td><td>${b.name}</td><td>${b.excess_kwh.toLocaleString()}</td></tr>`).join('')}
    </tbody></table>
    <div id="savingHourChart" class="chart-container" style="margin-top:1rem"></div>
    <div class="alert alert-success" style="margin-top:1rem">${data.strategies.map(s=>`<div>• ${s}</div>`).join('')}</div>
  `;
  if (data.by_hour.length) {
    Plotly.newPlot('savingHourChart', [{
      x: data.by_hour.map(h=>h['Hour']||h.hour),
      y: data.by_hour.map(h=>h['Average Saveable Energy by Hour of Day (kWh)']||h.avg_saveable_kwh||0),
      type:'bar', marker:{color:'#00f5a0'}
    }], plotLayout({title:'小时可节约电量分布'}));
  }
}

// === 模型解释 ===
async function renderExplain(p) {
  showKPI(false);
  p.innerHTML = '<div class="loading-spinner" style="text-align:center;padding:2rem;">加载中...</div>';
  const data = await api('/api/shap');
  p.innerHTML = `
    <h2 style="font-family:var(--font-display);color:var(--glow-green);margin-bottom:1rem;">🔍 模型解释</h2>
    <div id="shapChart" class="chart-container"></div>
    <div style="background:var(--bg-panel);backdrop-filter:blur(16px);border-left:3px solid var(--glow-green);padding:12px 16px;margin-top:1rem;border-radius:4px;">
      <strong style="color:var(--text-primary);">特征解读</strong><br>
      <span style="color:var(--text-secondary);">${data.interpretation}</span>
    </div>
  `;
  Plotly.newPlot('shapChart', [{
    x: data.features.map(f=>f.importance),
    y: data.features.map(f=>f.cn),
    type:'bar', orientation:'h', marker:{color:'#00f5a0'},
    text: data.features.map(f=>f.importance.toFixed(3)), textposition:'outside', textfont:{color:'#8899b4'}
  }], plotLayout({title:'SHAP 特征重要性', yaxis:{autorange:'reversed'}, height:400}), {responsive:true});
}

// === 报告导出 ===
function renderExport(p) {
  showKPI(false);
  if (!lastPrediction) {
    p.innerHTML = '<p class="text-secondary text-center" style="padding:3rem;">请先在预测分析中进行单点预测</p>';
    return;
  }
  const total = lastPrediction.curve.reduce((a,b)=>a+b,0);
  const peakH = lastPrediction.curve.indexOf(Math.max(...lastPrediction.curve));
  p.innerHTML = `
    <h2 style="font-family:var(--font-display);color:var(--glow-green);margin-bottom:1rem;">📋 报告导出</h2>
    <div class="kpi-row">
      <div class="kpi-card"><span class="kpi-label">建筑</span><span class="kpi-value" style="--glow:#00f5a0;font-size:1.1rem;">${lastPrediction.building}</span></div>
      <div class="kpi-card"><span class="kpi-label">峰值小时</span><span class="kpi-value" style="--glow:#ff9100">${peakH}:00</span></div>
      <div class="kpi-card"><span class="kpi-label">日总能耗</span><span class="kpi-value" style="--glow:#00f5a0">${total.toLocaleString(undefined,{maximumFractionDigits:0})} kWh</span></div>
      <div class="kpi-card"><span class="kpi-label">碳排放</span><span class="kpi-value" style="--glow:#b388ff">${(total*0.5).toLocaleString(undefined,{maximumFractionDigits:0})} kg</span></div>
    </div>
    <h3 class="section-title">24h 预测数据</h3>
    <table><thead><tr><th>小时</th><th>预测能耗(kWh)</th><th>碳排放(kg)</th></tr></thead><tbody>
      ${lastPrediction.curve.map((v,i)=>`<tr><td>${i}:00</td><td>${v.toLocaleString(undefined,{maximumFractionDigits:0})}</td><td>${(v*0.5).toLocaleString(undefined,{maximumFractionDigits:0})}</td></tr>`).join('')}
    </tbody></table>
    <button class="btn btn-primary" onclick="downloadReportCSV()" style="margin-top:1rem">⬇️ 下载CSV</button>
  `;
}

function downloadReportCSV() {
  if (!lastPrediction) return;
  let csv = '小时,预测能耗(kWh),碳排放(kg)\n';
  lastPrediction.curve.forEach((v,i) => { csv += `${i}:00,${v},${v*0.5}\n`; });
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([csv],{type:'text/csv'})); a.download = `${lastPrediction.building}_forecast.csv`; a.click();
}

// =============================================
//  数据大屏
// =============================================
function setupDashboard() {
  document.getElementById('dashBtn').addEventListener('click', openDashboard);
  document.getElementById('dashExit').addEventListener('click', closeDashboard);
}

async function openDashboard() {
  document.getElementById('dashOverlay').style.display = 'block';
  document.body.style.overflow = 'hidden';
  await refreshDashboard();
  window._dashTimer = setInterval(refreshDashboard, 30000);
}

function closeDashboard() {
  document.getElementById('dashOverlay').style.display = 'none';
  document.body.style.overflow = '';
  if (window._dashTimer) { clearInterval(window._dashTimer); window._dashTimer = null; }
}

async function refreshDashboard() {
  const grid = document.getElementById('dashGrid');
  const data = await api('/api/dashboard');
  grid.innerHTML = `
    <div class="dash-kpi"><span class="kpi-label">建筑数量</span><span class="kpi-value" style="--glow:#00f5a0">${data.buildings_count}</span></div>
    <div class="dash-kpi"><span class="kpi-label">模型 R²</span><span class="kpi-value" style="--glow:#00f5a0">${data.r2.toFixed(4)}</span></div>
    <div class="dash-kpi"><span class="kpi-label">sMAPE</span><span class="kpi-value" style="--glow:#ff9100">${data.smape}%</span></div>
    <div class="dash-kpi"><span class="kpi-label">节能量</span><span class="kpi-value" style="--glow:#ff5370">${data.saving_pct}%</span></div>
    <div class="dash-chart" id="dashChart1"></div>
    <div class="dash-chart" id="dashChart2"></div>
    <div class="dash-chart" id="dashChart3" style="grid-column:span 4"></div>
    <div class="dash-refresh">刷新: ${new Date().toLocaleTimeString()}</div>
  `;
  Plotly.newPlot('dashChart1', [{x:[...Array(24).keys()], y:await get24hPredictions(), type:'scatter', mode:'lines', line:{color:'#00f5a0',width:2}, fill:'tozeroy', fillcolor:'rgba(0,245,160,0.08)'}], plotLayout({title:'24h能耗曲线', height:300}), {responsive:true});
  Plotly.newPlot('dashChart2', [{x:[...Array(24).keys()], y:await get24hPredictions(1.2), type:'scatter', mode:'lines', line:{color:'#b388ff',width:2}, fill:'tozeroy', fillcolor:'rgba(179,136,255,0.08)'}], plotLayout({title:'高峰场景', height:300}), {responsive:true});
  const sh = await api('/api/shap');
  Plotly.newPlot('dashChart3', [{x:sh.features.map(f=>f.importance), y:sh.features.map(f=>f.cn), type:'bar', orientation:'h', marker:{color:'#00f5a0'}}], plotLayout({title:'SHAP 特征重要性', height:300, yaxis:{autorange:'reversed'}}), {responsive:true});
}

async function get24hPredictions(mult=1.0) {
  const bld = (window._buildings && window._buildings[0]) || 'NDRC Building';
  const pl = Array.from({length:24},(_,h)=>({hour:h,month:7,day_of_week:3,building_id:bld,meter:0,...Object.fromEntries(Object.entries(DH).map(([k,v])=>[k,v*mult]))}));
  const data = await api('/api/batch_predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({params_list:pl})});
  return data.predictions;
}
