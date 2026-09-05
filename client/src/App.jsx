import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  getBenchmarks, getHealth, getManifold, getMetadata, getSearchHistory,
  getTopAnomalies, getRetrainOptions, retrain, scoreTelemetry,
} from './services/api'
import './styles.css'

function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—'
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits })
}

function StateBox({ status, error, emptyText = 'No data returned by the API.' }) {
  if (status === 'loading') return <div className="state">Loading live API data…</div>
  if (status === 'error') return <div className="state error"><strong>API error</strong><span>{error}</span></div>
  if (status === 'empty') return <div className="state">{emptyText}</div>
  return null
}

function Section({ id, title, kicker, children }) {
  return <section id={id} className="panel"><div className="section-head"><div><p className="kicker">{kicker}</p><h2>{title}</h2></div></div>{children}</section>
}

function Metric({ label, value, suffix = '' }) {
  return <div className="metric"><span>{label}</span><strong>{value}{suffix}</strong></div>
}

function HealthBanner({ health }) {
  if (!health) return <div className="banner">Connecting to FastAPI…</div>
  if (!health.ml_ready) return <div className="banner danger"><strong>ML artifacts are not ready.</strong><span>{health.ml_reason}</span><span>Use Retrain controls below to build a fresh validated artifact set.</span></div>
  return <div className="banner success"><strong>Production model online</strong><span>{health.model_type} · artifact schema {health.artifact_schema_version}</span><span>Selected params: {JSON.stringify(health.selected_hyperparameters)}</span></div>
}

function ThreatScorer({ metadata, health }) {
  const features = metadata?.feature_metadata || []
  const [values, setValues] = useState({})
  const [state, setState] = useState({ status: 'idle', result: null, error: null })
  useEffect(() => {
    if (features.length) setValues(Object.fromEntries(features.map((f) => [f.feature, f.default])))
  }, [retrainMeta?.default_rows, retrainMeta?.default_seed, retrainMeta?.default_contamination, retrainMeta?.default_validation_fraction])
  const submit = async (event) => {
    event.preventDefault()
    setState({ status: 'loading', result: null, error: null })
    try {
      const payload = Object.fromEntries(features.map((f) => [f.api_field, Number(values[f.feature])]))
      const result = await scoreTelemetry(payload)
      setState({ status: 'ready', result, error: null })
    } catch (error) { setState({ status: 'error', result: null, error: error.message }) }
  }
  if (!health?.ml_ready) return <StateBox status="empty" emptyText={health?.ml_reason || 'ML model is not ready.'} />
  if (!features.length) return <StateBox status="empty" emptyText="No feature metadata returned by the API." />
  return <>
    <form onSubmit={submit} className="scorer-grid">
      {features.map((f) => <label className="control" key={f.feature}>
        <span><b>{f.feature}</b><small>{f.description} · {f.unit} · normal {f.normal_range}</small></span>
        <input type="range" min={f.min} max={f.max} step={f.step} value={values[f.feature] ?? f.default} onChange={(e) => setValues((old) => ({ ...old, [f.feature]: e.target.value }))} />
        <input type="number" min={f.min} max={f.max} step={f.step} value={values[f.feature] ?? f.default} onChange={(e) => setValues((old) => ({ ...old, [f.feature]: e.target.value }))} />
      </label>)}
      <button className="primary" disabled={state.status === 'loading'}>{state.status === 'loading' ? 'Scoring with production model…' : 'Score telemetry'}</button>
    </form>
    {state.status === 'error' && <StateBox status="error" error={state.error} />}
    {state.result && <div className="score-layout">
      <div className="score-card">
        <p className="kicker">Model output</p><div className="threat-number">{fmt(state.result.threat_index, 2)}</div><div className="muted">Calibrated threat index</div>
        <div className={`verdict ${state.result.is_anomaly ? 'bad' : 'good'}`}>{state.result.verdict.replace('_', ' ')}</div>
        <Metric label="Raw anomaly score" value={fmt(state.result.raw_anomaly_score, 6)} />
        <Metric label="Threat threshold" value={fmt(state.result.threat_threshold, 2)} />
        <p className="note">Verdict is produced by the persisted {state.result.model_type} after the persisted RobustScaler and score calibration.</p>
      </div>
      <div className="attribution-card">
        <p className="kicker">Why it looks unusual · separate explanation</p>
        <p className="note">These robust-IQR deviations explain unusual raw dimensions. They do not produce or override the model verdict.</p>
        <div className="bars">{state.result.attribution.map((a) => <div className="bar-row" key={a.feature}><span>{a.feature}</span><div><i style={{ width: `${Math.min(100, a.iqr_deviation * 18)}%` }} /></div><b>{fmt(a.iqr_deviation, 2)}× IQR</b></div>)}</div>
      </div>
    </div>}
  </>
}

function Manifold({ data, status, error }) {
  if (status !== 'ready') return <StateBox status={status} error={error} />
  const points = data?.points || []
  if (!points.length) return <StateBox status="empty" />
  const xs = points.map((p) => p.pc1), ys = points.map((p) => p.pc2)
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys)
  const sx = (x) => 20 + ((x - minX) / (maxX - minX || 1)) * 760
  const sy = (y) => 380 - ((y - minY) / (maxY - minY || 1)) * 360
  return <>
    <div className="metric-row"><Metric label="PCA variance · PC1" value={`${fmt(data.pca_explained_variance_ratio?.[0] * 100, 1)}%`} /><Metric label="PCA variance · PC2" value={`${fmt(data.pca_explained_variance_ratio?.[1] * 100, 1)}%`} /><Metric label="Points" value={fmt(points.length, 0)} /></div>
    <div className="chart-shell"><svg viewBox="0 0 800 400" role="img" aria-label="PCA manifold scatterplot">{points.map((p) => <circle key={p.row_id} cx={sx(p.pc1)} cy={sy(p.pc2)} r={p.is_anomaly ? 3.2 : 1.6} className={p.is_anomaly ? 'dot anomaly' : 'dot normal'}><title>{`${p.archetype} · threat ${fmt(p.threat_index, 2)}`}</title></circle>)}</svg></div>
    <p className="note">Coordinates and threat indices are loaded from <code>/api/manifold</code>; labels are evaluation/visualization ground truth only.</p>
  </>
}

function ModelTournament({ data, status, error }) {
  if (status !== 'ready') return <StateBox status={status} error={error} />
  const rows = data?.results || []; if (!rows.length) return <StateBox status="empty" />
  return <div className="table-wrap"><table><thead><tr><th>Detector</th><th>ROC-AUC</th><th>PR-AUC</th><th>Precision</th><th>Recall</th><th>F1</th><th>Fit sec</th><th>ms / row</th></tr></thead><tbody>{rows.map((r) => <tr key={r.model_name}><td>{r.model_name}</td><td>{fmt(r.roc_auc, 4)}</td><td>{fmt(r.pr_auc, 4)}</td><td>{fmt(r.precision, 3)}</td><td>{fmt(r.recall, 3)}</td><td>{fmt(r.f1, 3)}</td><td>{fmt(r.fit_seconds, 3)}</td><td>{fmt(r.inference_ms_per_row, 4)}</td></tr>)}</tbody></table></div>
}

function AutoResearch({ data, status, error }) {
  if (status !== 'ready') return <StateBox status={status} error={error} />
  const rows = data?.runs || []; if (!rows.length) return <StateBox status="empty" />
  return <><div className="banner measured"><strong>Real measured search runs</strong><span>{data.selection_reason}</span></div><div className="table-wrap"><table><thead><tr><th>Run</th><th>Contamination</th><th>Trees</th><th>Max samples</th><th>PR-AUC</th><th>F1</th><th>ROC-AUC</th><th>Measured</th></tr></thead><tbody>{rows.map((r) => <tr key={r.run_id} className={JSON.stringify(r.params) === JSON.stringify(data.selected_params) ? 'selected' : ''}><td>{r.run_id}</td><td>{fmt(r.params.contamination, 3)}</td><td>{r.params.n_estimators}</td><td>{String(r.params.max_samples)}</td><td>{fmt(r.pr_auc, 4)}</td><td>{fmt(r.f1, 4)}</td><td>{fmt(r.roc_auc, 4)}</td><td>{String(r.measured)}</td></tr>)}</tbody></table></div></>
}

function AnomalyExplorer({ data, status, error }) {
  if (status !== 'ready') return <StateBox status={status} error={error} />
  const rows = data?.rows || []; if (!rows.length) return <StateBox status="empty" />
  return <div className="table-wrap"><table><thead><tr><th>Rank</th><th>Row</th><th>Threat</th><th>Raw score</th><th>Ground-truth archetype</th></tr></thead><tbody>{rows.map((r) => <tr key={r.rank}><td>{r.rank}</td><td>{r.row_id}</td><td>{fmt(r.threat_index, 2)}</td><td>{fmt(r.raw_anomaly_score, 5)}</td><td>{r.archetype}</td></tr>)}</tbody></table><p className="note">Showing all {rows.length.toLocaleString()} stored top-anomaly rows returned by the API.</p></div>
}

function CrispReport({ health, metadata, benchmarks, search }) {
  if (!health?.ml_ready) return <StateBox status="empty" emptyText={health?.ml_reason || 'ML model is not ready.'} />
  if (!metadata || !benchmarks || !search) return <StateBox status="loading" />
  const best = [...(benchmarks.results || [])].sort((a,b) => b.pr_auc - a.pr_auc)[0]
  return <div className="report-grid">
    <article><h3>Business understanding</h3><p>Detect unusual infrastructure telemetry early enough for an analyst to investigate operational or security threats.</p></article>
    <article><h3>Data understanding</h3><p>{metadata.dataset_rows.toLocaleString()} locally synthesized rows across {metadata.feature_order.length.toLocaleString()} telemetry dimensions. Ground truth exists only for evaluation and visualization.</p></article>
    <article><h3>Data preparation</h3><p>The active RobustScaler was fit on {metadata.training_rows.toLocaleString()} training rows only; validation rows were excluded from learned preprocessing.</p></article>
    <article><h3>Modeling</h3><p>{benchmarks.results.length.toLocaleString()} detector families are represented in the stored benchmark. The live production scorer is {metadata.production_model} using parameters {JSON.stringify(metadata.selected_params)}.</p></article>
    <article><h3>Evaluation</h3><p>The strongest stored benchmark PR-AUC is {fmt(best?.pr_auc, 4)} from {best?.model_name || 'the stored comparison'}. These high benchmark values are measured on deliberately separable synthetic anomaly archetypes and should not be read as real-world production accuracy.</p></article>
    <article><h3>Deployment</h3><p>Live scoring loads the same persisted scaler, detector, and calibration evaluated offline. AutoResearch shows {search.runs.length.toLocaleString()} real measured configurations rather than a fabricated leaderboard.</p></article>
  </div>
}

function RetrainControls({ onDone, options }) {
  const retrainMeta = options
  const [form, setForm] = useState(null)
  useEffect(() => {
    if (retrainMeta) setForm({ rows: retrainMeta.default_rows, contamination: retrainMeta.default_contamination, seed: retrainMeta.default_seed, validation_fraction: retrainMeta.default_validation_fraction })
  }, [retrainMeta?.default_rows, retrainMeta?.default_seed, retrainMeta?.default_contamination, retrainMeta?.default_validation_fraction])
  const [state, setState] = useState({ status: 'idle', result: null, error: null })
  const run = async (e) => {
    e.preventDefault(); setState({ status: 'loading', result: null, error: null })
    try { const result = await retrain(form); setState({ status: 'ready', result, error: null }); await onDone() }
    catch (error) { setState({ status: 'error', result: null, error: error.status === 409 ? `Retrain already running: ${error.message}` : error.message }) }
  }
  if (!form || !retrainMeta) return <StateBox status="loading" />
  const constraints = {
    rows: [retrainMeta.min_rows, retrainMeta.max_rows, retrainMeta.rows_step],
    contamination: [retrainMeta.min_contamination, retrainMeta.max_contamination, retrainMeta.contamination_step],
    seed: [undefined, undefined, 1],
    validation_fraction: [retrainMeta.min_validation_fraction, retrainMeta.max_validation_fraction, retrainMeta.validation_step],
  }
  return <><form className="retrain" onSubmit={run}>
    {Object.entries(form).map(([key,value]) => <label key={key}><span>{key.replaceAll('_',' ')}</span><input type="number" value={value} min={constraints[key][0]} max={constraints[key][1]} step={constraints[key][2]} onChange={(e)=>setForm((old)=>({...old,[key]:Number(e.target.value)}))}/></label>)}
    <button className="primary" disabled={state.status === 'loading'}>{state.status === 'loading' ? 'Running real search and atomic retrain…' : 'Retrain production model'}</button>
  </form>
  {state.status === 'loading' && <div className="progress"><i /></div>}
  {state.status === 'error' && <StateBox status="error" error={state.error} />}
  {state.result && <div className="banner success"><strong>Retrain complete</strong><span>{state.result.message}</span><span>Rows used: {state.result.rows_used.toLocaleString()} · search runs: {state.result.search_runs.toLocaleString()}</span></div>}</>
}

export default function App() {
  const [healthState, setHealthState] = useState({ status:'loading', data:null, error:null })
  const [resources, setResources] = useState({})
  const nav = useMemo(() => [['scorer','Threat Scorer'],['manifold','PCA Manifold'],['models','Model Tournament'],['research','AutoResearch'],['anomalies','Anomalies'],['report','CRISP-DM'],['retrain','Retrain']],[])
  const refresh = useCallback(async () => {
    setHealthState({status:'loading',data:null,error:null})
    try {
      const [health, retrainOptions] = await Promise.all([getHealth(), getRetrainOptions()])
      setHealthState({status:'ready',data:health,error:null})
      setResources({ retrainOptions: { status:'ready', data:retrainOptions, error:null } })
      if (!health.ml_ready) return
      const calls = { metadata:getMetadata, benchmarks:getBenchmarks, manifold:getManifold, search:getSearchHistory, anomalies:getTopAnomalies }
      setResources((old) => ({ ...old, ...Object.fromEntries(Object.keys(calls).map((k)=>[k,{status:'loading',data:null,error:null}])) }))
      await Promise.all(Object.entries(calls).map(async ([key,fn]) => {
        try { const data=await fn(); setResources((old)=>({...old,[key]:{status:Array.isArray(data)?(data.length?'ready':'empty'):'ready',data,error:null}})) }
        catch(error){ setResources((old)=>({...old,[key]:{status:'error',data:null,error:error.message}})) }
      }))
    } catch (error) { setHealthState({status:'error',data:null,error:error.message}) }
  },[])
  useEffect(()=>{ refresh() },[refresh])
  const health=healthState.data
  return <main>
    <header className="topbar"><div><p className="kicker">Assignment One · Part Two · Analyst Console</p><h1>Anomaly Detection Studio</h1><p>Live production scoring, measured model evidence, and reproducible synthetic telemetry.</p></div><nav>{nav.map(([id,label])=><a href={`#${id}`} key={id}>{label}</a>)}</nav></header>
    <div className="shell">
      {healthState.status==='error' ? <StateBox status="error" error={healthState.error}/> : <HealthBanner health={health}/>} 
      <Section id="scorer" title="Live Threat Scorer" kicker="Production pipeline"><ThreatScorer metadata={resources.metadata?.data} health={health}/></Section>
      <Section id="manifold" title="PCA Manifold" kicker="Stored visualization">{health?.ml_ready ? <Manifold {...(resources.manifold || {status:'loading'})}/> : <StateBox status="empty" emptyText={health?.ml_reason || "ML model is not ready."}/>}</Section>
      <Section id="models" title="Model Tournament" kicker="Stored validation benchmark">{health?.ml_ready ? <ModelTournament {...(resources.benchmarks || {status:'loading'})}/> : <StateBox status="empty" emptyText={health?.ml_reason || "ML model is not ready."}/>}</Section>
      <Section id="research" title="AutoResearch Leaderboard" kicker="Measured hyperparameter search">{health?.ml_ready ? <AutoResearch data={resources.search?.data} status={resources.search?.status || 'loading'} error={resources.search?.error}/> : <StateBox status="empty" emptyText={health?.ml_reason || "ML model is not ready."}/>}</Section>
      <Section id="anomalies" title="Anomaly Explorer" kicker="Stored top anomalies">{health?.ml_ready ? <AnomalyExplorer data={resources.anomalies?.data} status={resources.anomalies?.status || 'loading'} error={resources.anomalies?.error}/> : <StateBox status="empty" emptyText={health?.ml_reason || "ML model is not ready."}/>}</Section>
      <Section id="report" title="CRISP-DM Report" kicker="Honest analysis narrative"><CrispReport health={health} metadata={resources.metadata?.data} benchmarks={resources.benchmarks?.data} search={resources.search?.data}/></Section>
      <Section id="retrain" title="Retrain Controls" kicker="Real bounded search"><RetrainControls onDone={refresh} options={resources.retrainOptions?.data}/></Section>
    </div>
  </main>
}
