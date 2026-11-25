import React, { useEffect, useMemo, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { BrowserRouter, Link, NavLink, Route, Routes } from 'react-router-dom'
import {
  Activity, BarChart3, Boxes, Database, FlaskConical, LayoutDashboard,
  Play, Plus, Search, Settings, Trash2, UploadCloud, BrainCircuit, MessageCircle, Send, Sparkles,
} from 'lucide-react'
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api } from './services/api'
import type { ChatMessage, DashboardSummary, Dataset, DatasetPreview, LatestPrediction, TrainedModel } from './services/api'
import './App.css'
import './Chat.css'
import './ChatModels.css'

const navGroups = [
  {
    label: 'Workspace',
    items: [
      { to: '/', label: 'Overview', icon: LayoutDashboard },
      { to: '/datasets', label: 'Datasets', icon: Database },
      { to: '/training', label: 'Training studio', icon: FlaskConical },
      { to: '/experiments', label: 'Experiments', icon: Activity },
      { to: '/models', label: 'Models', icon: Boxes },
      { to: '/predict', label: 'Prediction playground', icon: BrainCircuit },
      { to: '/chat', label: 'Chat with Data', icon: MessageCircle },
    ],
  },
  {
    label: 'Integrations',
    items: [
      { to: '/hf-chat', label: 'Hugging Face', icon: Sparkles },
    ],
  },
]

function Shell() {
  const [mobileOpen, setMobileOpen] = useState(false)
  return (
    <div className="app-shell">
      <aside className={mobileOpen ? 'sidebar open' : 'sidebar'}>
        <div className="brand">
          <span className="brand-mark"><BrainCircuit size={18} /></span>
          <span>Neural<span>Lab</span></span>
        </div>
        <nav>
          {navGroups.map((group) => (
            <div className="nav-group" key={group.label}>
              <div className="nav-caption">{group.label}</div>
              {group.items.map(({ to, label, icon: Icon }) => (
                <NavLink key={to} to={to} end={to === '/'} onClick={() => setMobileOpen(false)}>
                  <Icon size={17} /><span>{label}</span>
                </NavLink>
              ))}
            </div>
          ))}
        </nav>
        <div className="sidebar-footer">
          <div className="plan-card"><Settings size={17} /><div><strong>Local workspace</strong><span>SQLite database</span></div></div>
        </div>
      </aside>
      <main>
        <header className="topbar">
          <button className="mobile-menu" onClick={() => setMobileOpen(!mobileOpen)}>Menu</button>
          <div className="breadcrumbs">NeuralLab <span>/</span> <strong>Workspace</strong></div>
          <div className="topbar-actions"><button className="icon-button"><Search size={16} /></button><div className="avatar">NL</div></div>
        </header>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/datasets" element={<Datasets />} />
          <Route path="/training" element={<Training />} />
          <Route path="/experiments" element={<Experiments />} />
          <Route path="/models" element={<Models />} />
          <Route path="/predict" element={<Predict />} />
          <Route path="/chat" element={<ChatWithData />} />
          <Route path="/hf-chat" element={<HuggingFacePlayground />} />
        </Routes>
      </main>
    </div>
  )
}

function PageHeading({ title, subtitle, action }: { title: string; subtitle: string; action?: React.ReactNode }) {
  return <div className="page-heading"><div><h1>{title}</h1><p>{subtitle}</p></div>{action}</div>
}

function Loading() {
  return <div className="loading"><span className="spinner" /> Loading workspace...</div>
}

function ErrorState({ error }: { error: string }) {
  return <div className="error-state"><strong>Something went wrong</strong><span>{error}</span></div>
}

function Dashboard() {
  const [data, setData] = useState<DashboardSummary | null>(null)
  const [error, setError] = useState('')
  useEffect(() => { api.dashboard().then(setData).catch((err: Error) => setError(err.message)) }, [])
  if (error) return <div className="page"><ErrorState error={error} /></div>
  if (!data) return <div className="page"><Loading /></div>
  return (
    <div className="page">
      <PageHeading title="Workspace overview" subtitle="A clear view of your machine learning work." action={<Link className="button primary" to="/training"><Plus size={16} />New training run</Link>} />
      <div className="metric-grid">
        <Metric label="Datasets" value={data.total_datasets} icon={<Database />} tone="violet" />
        <Metric label="Trained models" value={data.total_models} icon={<Boxes />} tone="blue" />
        <Metric label="Experiments" value={data.total_experiments} icon={<FlaskConical />} tone="amber" />
        <Metric label="Completed" value={data.completed_models} icon={<Activity />} tone="green" />
      </div>
      <div className="dashboard-grid">
        <section className="panel chart-panel">
          <div className="panel-heading"><div><h2>Training activity</h2><p>Models trained over the last 7 days</p></div><span className="live"><i />Live data</span></div>
          <div className="chart">
            <ResponsiveContainer width="100%" height={260}>
              <AreaChart data={data.training_history}>
                <defs><linearGradient id="area" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#8b7cff" stopOpacity={0.35}/><stop offset="95%" stopColor="#8b7cff" stopOpacity={0}/></linearGradient></defs>
                <CartesianGrid stroke="#20232c" vertical={false}/>
                <XAxis dataKey="date" stroke="#777d8a" tickLine={false} axisLine={false}/>
                <YAxis allowDecimals={false} stroke="#777d8a" tickLine={false} axisLine={false}/>
                <Tooltip contentStyle={{ background:'#15171d', border:'1px solid #292c35', borderRadius:10 }}/>
                <Area type="monotone" dataKey="models" stroke="#9b8cff" strokeWidth={2} fill="url(#area)"/>
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </section>
        <section className="panel best-panel">
          <div className="panel-heading"><div><h2>Best performer</h2><p>Highest reported validation score</p></div><BarChart3 size={20}/></div>
          {data.best_model ? <div className="best-content"><span className="score">{(data.best_model.score * 100).toFixed(1)}%</span><strong>{data.best_model.name}</strong><p>{data.best_model.model_type.replaceAll('_',' ')}</p></div> : <Empty icon={<TrophyIcon />} title="No trained models" text="Complete a training run to rank your models." />}
        </section>
      </div>
    </div>
  )
}

function TrophyIcon() { return <BarChart3 size={22} /> }

function Metric({ label, value, icon, tone }: { label: string; value: number; icon: React.ReactNode; tone: string }) {
  return <article className="metric"><div><span>{label}</span><div className={`metric-icon ${tone}`}>{icon}</div></div><strong>{value}</strong><p><i />Tracked in workspace</p></article>
}

function Datasets() {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [selected, setSelected] = useState<DatasetPreview | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [name, setName] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const refresh = () => api.datasets().then(setDatasets).catch((err: Error) => setError(err.message))
  useEffect(() => { void refresh() }, [])
  const upload = async (event: FormEvent) => {
    event.preventDefault()
    if (!file) return setError('Choose a CSV file first.')
    setBusy(true); setError('')
    try { await api.uploadDataset(file, name); setFile(null); setName(''); refresh() } catch (err) { setError((err as Error).message) } finally { setBusy(false) }
  }
  const inspect = async (id: number) => {
    try { setSelected(await api.dataset(id)); setError('') } catch (err) { setError((err as Error).message) }
  }
  const remove = async (id: number) => {
    if (!confirm('Delete this dataset and its stored CSV?')) return
    await api.deleteDataset(id); setSelected(null); refresh()
  }
  return (
    <div className="page">
      <PageHeading title="Datasets" subtitle="Upload, inspect, and prepare your training data." />
      {error && <ErrorState error={error}/>}
      <div className="dataset-layout">
        <section className="panel upload-panel">
          <div className="panel-heading"><div><h2>Upload CSV</h2><p>Maximum file size: 25 MB</p></div></div>
          <form onSubmit={upload}>
            <label className="dropzone"><UploadCloud size={24}/><strong>{file ? file.name : 'Choose a CSV file'}</strong><span>{file ? `${(file.size/1024).toFixed(1)} KB` : 'CSV format required'}</span><input type="file" accept=".csv,text/csv" onChange={(e) => setFile(e.target.files?.[0] || null)} /></label>
            <label className="field"><span>Dataset name</span><input value={name} onChange={(e) => setName(e.target.value)} placeholder="Customer churn" /></label>
            <button className="button primary" disabled={busy || !file}>{busy ? 'Uploading...' : 'Upload dataset'}</button>
          </form>
        </section>
        <section className="panel dataset-list">
          <div className="panel-heading"><div><h2>Data library</h2><p>{datasets.length} dataset{datasets.length === 1 ? '' : 's'}</p></div></div>
          {datasets.length === 0 ? <Empty title="No datasets yet" text="Upload a CSV to begin." /> : <div className="dataset-items">{datasets.map((dataset) => (
            <button key={dataset.id} onClick={() => inspect(dataset.id)}>
              <span className="dataset-icon"><Database size={18}/></span><div><strong>{dataset.name}</strong><span>{dataset.row_count.toLocaleString()} rows · {dataset.column_count} columns</span></div><span className="dataset-size">{(dataset.size_bytes/1024).toFixed(1)} KB</span>
            </button>
          ))}</div>}
        </section>
      </div>
      {selected && <section className="panel inspection">
        <div className="panel-heading"><div><h2>{selected.dataset.name}</h2><p>{selected.dataset.row_count} rows · {selected.dataset.column_count} columns</p></div><button className="icon-danger" onClick={() => remove(selected.dataset.id)}><Trash2 size={16}/></button></div>
        <div className="stats-row"><Stat label="Missing values" value={String((selected.statistics.missing_values as number) ?? 0)} /><Stat label="Duplicate rows" value={String((selected.statistics.duplicate_rows as number) ?? 0)} /><Stat label="Numeric" value={String((selected.statistics.numeric_columns as number) ?? 0)} /><Stat label="Categorical" value={String((selected.statistics.categorical_columns as number) ?? 0)} /></div>
        <div className="table-wrap"><table><thead><tr><th>Column</th><th>Type</th><th>Missing</th><th>Unique</th><th>Example</th></tr></thead><tbody>{selected.columns.map((column) => <tr key={column.name}><td><strong>{column.name}</strong></td><td>{column.dtype}</td><td>{column.missing}</td><td>{column.unique}</td><td>{column.example || '—'}</td></tr>)}</tbody></table></div>
      </section>}
    </div>
  )
}

function Stat({ label, value }: { label: string; value: string }) { return <div className="stat"><span>{label}</span><strong>{value}</strong></div> }

function Empty({ title, text, icon }: { title: string; text: string; icon?: React.ReactNode }) {
  return <div className="empty"><div className="empty-icon">{icon || <Database size={20}/>}</div><strong>{title}</strong><span>{text}</span></div>
}

function Training() {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [datasetId, setDatasetId] = useState('')
  const [preview, setPreview] = useState<DatasetPreview | null>(null)
  const [target, setTarget] = useState('')
  const [features, setFeatures] = useState<string[]>([])
  const [name, setName] = useState('')
  const [modelType, setModelType] = useState('random_forest')
  const [problemType, setProblemType] = useState<'classification' | 'regression'>('classification')
  const [testSize, setTestSize] = useState('0.2')
  const [seed, setSeed] = useState('42')
  const [missing, setMissing] = useState('median')
  const [scaling, setScaling] = useState('standard')
  const [error, setError] = useState('')
  const [result, setResult] = useState<TrainedModel | null>(null)
  const [busy, setBusy] = useState(false)
  useEffect(() => { api.datasets().then(setDatasets).catch((err: Error) => setError(err.message)) }, [])
  useEffect(() => {
    if (!datasetId) return
    api.dataset(Number(datasetId)).then(setPreview).catch((err: Error) => setError(err.message))
  }, [datasetId])
  const submit = async (event: FormEvent) => {
    event.preventDefault()
    if (!datasetId || !target || features.length === 0) return setError('Select a dataset, target, and at least one feature.')
    setBusy(true); setError(''); setResult(null)
    try {
      const model = await api.train({ dataset_id:Number(datasetId), name:name || `${modelType.replaceAll('_',' ')} run`, model_type:modelType, problem_type:problemType, target_column:target, feature_columns:features, test_size:Number(testSize), random_seed:Number(seed), missing_strategy:missing, scaling, hyperparameters: modelType === 'random_forest' ? { n_estimators:100 } : modelType === 'mlp' ? { epochs:20, learning_rate:0.001, batch_size:32 } : {} })
      setResult(model)
      const interval = window.setInterval(async () => {
        const updated = await api.models().then(models => models.find(item => item.id === model.id))
        if (updated && updated.status !== 'queued' && updated.status !== 'running') { setResult(updated); window.clearInterval(interval) }
      }, 1200)
    } catch (err) { setError((err as Error).message) } finally { setBusy(false) }
  }
  return (
    <div className="page"><PageHeading title="Training studio" subtitle="Configure a reproducible experiment and train a real model." />
      {error && <ErrorState error={error}/>}
      <form className="training-layout" onSubmit={submit}>
        <section className="panel">
          <div className="panel-heading"><div><h2>Data configuration</h2><p>Select inputs and preprocessing</p></div></div>
          <div className="form-grid">
            <label className="field"><span>Dataset</span><select value={datasetId} onChange={(e) => { setDatasetId(e.target.value); setTarget(''); setFeatures([]) }}><option value="">Choose dataset</option>{datasets.map(d => <option value={d.id} key={d.id}>{d.name}</option>)}</select></label>
            <label className="field"><span>Problem type</span><select value={problemType} onChange={(e) => setProblemType(e.target.value as 'classification'|'regression')}><option value="classification">Classification</option><option value="regression">Regression</option></select></label>
            <label className="field"><span>Target column</span><select value={target} onChange={(e) => { setTarget(e.target.value); setFeatures(preview?.columns.map(c => c.name).filter(c => c !== e.target.value) || []) }}><option value="">Choose target</option>{preview?.columns.map(c => <option key={c.name}>{c.name}</option>)}</select></label>
            <label className="field"><span>Missing values</span><select value={missing} onChange={(e) => setMissing(e.target.value)}><option value="median">Median / most frequent</option><option value="drop">Drop rows</option></select></label>
          </div>
          {preview && <div className="feature-list"><span>Feature columns</span><div>{preview.columns.filter(c => c.name !== target).map(column => <label key={column.name}><input type="checkbox" checked={features.includes(column.name)} onChange={(e) => setFeatures(e.target.checked ? [...features,column.name] : features.filter(f => f !== column.name))}/>{column.name}</label>)}</div></div>}
        </section>
        <section className="panel">
          <div className="panel-heading"><div><h2>Model configuration</h2><p>Algorithm and training settings</p></div></div>
          <div className="form-grid">
            <label className="field"><span>Experiment name</span><input value={name} onChange={(e) => setName(e.target.value)} placeholder="Random forest baseline"/></label>
            <label className="field"><span>Model type</span><select value={modelType} onChange={(e) => setModelType(e.target.value)}><option value="linear_regression">Linear regression</option><option value="logistic_regression">Logistic regression</option><option value="random_forest">Random forest</option><option value="knn">K-nearest neighbors</option><option value="decision_tree">Decision tree</option><option value="mlp">PyTorch MLP</option></select></label>
            <label className="field"><span>Test split</span><select value={testSize} onChange={(e) => setTestSize(e.target.value)}><option value="0.2">20%</option><option value="0.25">25%</option><option value="0.3">30%</option></select></label>
            <label className="field"><span>Random seed</span><input type="number" min="0" value={seed} onChange={(e) => setSeed(e.target.value)}/></label>
            <label className="field"><span>Numeric scaling</span><select value={scaling} onChange={(e) => setScaling(e.target.value)}><option value="standard">Standard</option><option value="robust">Robust</option><option value="none">None</option></select></label>
          </div>
          <button className="button primary train-button" disabled={busy}><Play size={16}/>{busy ? 'Starting training...' : 'Start training'}</button>
        </section>
      </form>
      {result && <section className="panel training-result"><div className="panel-heading"><div><h2>Training status</h2><p>{result.name}</p></div><span className={`status ${result.status}`}>{result.status}</span></div>{result.error && <ErrorState error={result.error}/>}<Metrics model={result}/></section>}
    </div>
  )
}

function Metrics({ model }: { model: TrainedModel }) {
  if (model.status !== 'completed') return <div className="empty-inline">Waiting for the training worker...</div>
  const entries = Object.entries(model.metrics).filter(([key]) => key !== 'confusion_matrix' && key !== 'labels' && key !== 'loss_history')
  const loss = (model.metrics.loss_history as { epoch:number; loss:number }[] | undefined) || []
  return <><div className="metric-row">{entries.map(([key,value]) => <div key={key}><span>{key.replaceAll('_',' ')}</span><strong>{typeof value === 'number' ? value.toFixed(4) : String(value)}</strong></div>)}</div>{loss.length > 0 && <div className="loss-chart"><ResponsiveContainer width="100%" height={220}><AreaChart data={loss}><CartesianGrid stroke="#20232c" vertical={false}/><XAxis dataKey="epoch" stroke="#777d8a"/><YAxis stroke="#777d8a"/><Tooltip contentStyle={{background:'#15171d',border:'1px solid #292c35',borderRadius:10}}/><Area type="monotone" dataKey="loss" stroke="#5ccba4" fill="#5ccba420"/></AreaChart></ResponsiveContainer></div>}</>
}

function Experiments() {
  const [experiments, setExperiments] = useState<{id:number;name:string;description:string|null;created_at:string}[]>([])
  const [name,setName]=useState(''); const [description,setDescription]=useState('')
  const refresh=()=>api.experiments().then(setExperiments)
  useEffect(()=>{refresh()},[])
  const create=async(e:FormEvent)=>{e.preventDefault();await api.createExperiment(name,description);setName('');setDescription('');refresh()}
  const remove=async(id:number)=>{if(confirm('Delete this experiment?')){await api.deleteExperiment(id);refresh()}}
  return <div className="page"><PageHeading title="Experiments" subtitle="Track and compare your saved machine learning work."/>
    <section className="panel experiment-layout"><form onSubmit={create}><h2>Create experiment</h2><label className="field"><span>Name</span><input required value={name} onChange={e=>setName(e.target.value)} placeholder="Baseline comparison"/></label><label className="field"><span>Description</span><textarea value={description} onChange={e=>setDescription(e.target.value)} placeholder="What are you testing?"/></label><button className="button primary"><Plus size={16}/>Create</button></form><div className="experiment-list">{experiments.length===0?<Empty title="No experiments" text="Create your first experiment."/>:experiments.map(exp=><article key={exp.id}><div><strong>{exp.name}</strong><p>{exp.description||'No description'}</p></div><span>{new Date(exp.created_at).toLocaleDateString()}</span><button className="icon-danger" onClick={()=>remove(exp.id)}><Trash2 size={15}/></button></article>)}</div></section>
  </div>
}

function Models() {
  const [models,setModels]=useState<TrainedModel[]>([])
  const refresh=()=>api.models().then(setModels)
  useEffect(()=>{refresh()},[])
  const remove=async(id:number)=>{if(confirm('Delete this trained model and artifact?')){await api.deleteModel(id);refresh()}}
  return <div className="page"><PageHeading title="Models" subtitle="Manage trained artifacts and export-ready model metadata."/>
    <section className="panel"><div className="table-wrap"><table><thead><tr><th>Name</th><th>Algorithm</th><th>Status</th><th>Target</th><th>Duration</th><th></th></tr></thead><tbody>{models.map(model=><tr key={model.id}><td><strong>{model.name}</strong></td><td>{model.model_type.replaceAll('_',' ')}</td><td><span className={`status ${model.status}`}>{model.status}</span></td><td>{model.target_column}</td><td>{model.duration_seconds ? `${model.duration_seconds}s`:'—'}</td><td><button className="icon-danger" onClick={()=>remove(model.id)}><Trash2 size={15}/></button></td></tr>)}</tbody></table></div>{models.length===0&&<Empty title="No trained models" text="Start a run in the training studio."/>}</section>
  </div>
}

function Predict() {
  const [models,setModels]=useState<TrainedModel[]>([])
  const [modelId,setModelId]=useState('')
  const [values,setValues]=useState<Record<string,string>>({})
  const [result,setResult]=useState<{prediction:unknown;probabilities:Record<string,number>|null}|null>(null)
  const [error,setError]=useState('')
  useEffect(()=>{api.models().then(models=>setModels(models.filter(m=>m.status==='completed')))},[])
  const selected=useMemo(()=>models.find(m=>m.id===Number(modelId)),[models,modelId])
  const submit=async(e:FormEvent)=>{e.preventDefault();if(!selected)return;try{const features=Object.fromEntries(Object.entries(values).map(([k,v])=>[k, v.trim() !== '' && !Number.isNaN(Number(v)) ? Number(v) : v]));const prediction=await api.predict(selected.id,features);setResult(prediction);const context:LatestPrediction={model_id:selected.id,model_name:selected.name,problem_type:selected.problem_type,target_column:selected.target_column,features,prediction:prediction.prediction,probabilities:prediction.probabilities};try{window.localStorage.setItem('neural-lab.latest-prediction',JSON.stringify(context));setError('')}catch{setError('Prediction succeeded, but the latest result could not be saved for Chat with Data.')}}catch(err){setError((err as Error).message)}}
  return <div className="page"><PageHeading title="Prediction playground" subtitle="Generate predictions using a trained model."/>
    {error&&<ErrorState error={error}/>}
    <div className="prediction-layout"><section className="panel"><div className="panel-heading"><div><h2>Model input</h2><p>Enter values for every feature</p></div></div><form onSubmit={submit}><label className="field"><span>Trained model</span><select value={modelId} onChange={e=>{setModelId(e.target.value);setValues({});setResult(null)}}><option value="">Choose model</option>{models.map(m=><option value={m.id} key={m.id}>{m.name}</option>)}</select></label>{selected&&<div className="prediction-fields">{selected.feature_columns.map(feature=><label className="field" key={feature}><span>{feature}</span><input required value={values[feature]||''} onChange={e=>setValues({...values,[feature]:e.target.value})}/></label>)}</div>}<button className="button primary" disabled={!selected}><BrainCircuit size={16}/>Generate prediction</button></form></section><section className="panel result-panel"><div className="panel-heading"><div><h2>Result</h2><p>Prediction output</p></div></div>{result?<div className="prediction-result"><span>Prediction</span><strong>{String(result.prediction)}</strong>{result.probabilities&&<div className="probabilities">{Object.entries(result.probabilities).map(([label,value])=><div key={label}><span>{label}</span><strong>{(value*100).toFixed(1)}%</strong><div><i style={{width:`${value*100}%`}}/></div></div>)}</div>}</div>:<Empty icon={<BrainCircuit size={22}/>} title="No prediction yet" text="Select a model and enter feature values."/>}</section></div>
  </div>
}

const starterPrompts = [
  'Summarize my dataset',
  'Which features affect churn most?',
  'Explain my latest prediction',
]

function readLatestPrediction(): LatestPrediction | null {
  try {
    const stored = window.localStorage.getItem('neural-lab.latest-prediction')
    return stored ? JSON.parse(stored) as LatestPrediction : null
  } catch {
    return null
  }
}

const savedModelsStorageKey = 'neural-lab.hugging-face-models'

function readSavedModels(): { ids: string[]; error: string } {
  try {
    const saved = window.localStorage.getItem(savedModelsStorageKey)
    if (!saved) return { ids: [], error: '' }
    const parsed: unknown = JSON.parse(saved)
    if (!Array.isArray(parsed) || !parsed.every((item) => typeof item === 'string')) {
      return { ids: [], error: 'Saved model list has an invalid format.' }
    }
    return { ids: parsed, error: '' }
  } catch {
    return { ids: [], error: 'Could not load saved Hugging Face models.' }
  }
}

function ChatWithData() {
  const [initialModels] = useState(readSavedModels)
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [datasetId, setDatasetId] = useState('')
  const [savedModelIds, setSavedModelIds] = useState(initialModels.ids)
  const [modelId, setModelId] = useState(initialModels.ids[0] ?? '')
  const [newModelId, setNewModelId] = useState('')
  const [hfToken, setHfToken] = useState('')
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: 'assistant', content: 'Ask a question about your uploaded data or your latest prediction.' },
  ])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(initialModels.error)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    api.datasets().then((items) => {
      setDatasets(items)
      if (items.length) setDatasetId(String(items[0].id))
    }).catch((err: Error) => setError(err.message))
  }, [])
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, busy])

  const addModel = (event: FormEvent) => {
    event.preventDefault()
    const repositoryId = newModelId.trim()
    if (!repositoryId) {
      setError('Enter a Hugging Face model repository ID.')
      return
    }
    const next = [...new Set([...savedModelIds, repositoryId])]
    setSavedModelIds(next)
    setModelId(repositoryId)
    setNewModelId('')
    try {
      window.localStorage.setItem(savedModelsStorageKey, JSON.stringify(next))
      setError('')
    } catch {
      setError('Model selected for this session, but could not be saved in browser storage.')
    }
  }

  const sendMessage = async (content: string) => {
    const prompt = content.trim()
    if (!prompt || busy) return
    const conversation = [...messages, { role: 'user' as const, content: prompt }]
    setMessages(conversation)
    setInput('')
    setBusy(true)
    setError('')
    try {
      const response = await api.chat(
        conversation,
        datasetId ? Number(datasetId) : undefined,
        modelId.trim(),
        hfToken,
        readLatestPrediction(),
      )
      setMessages([...conversation, { role: 'assistant', content: response.reply }])
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const submit = (event: FormEvent) => {
    event.preventDefault()
    void sendMessage(input)
  }

  return (
    <div className="page chat-page">
      <PageHeading title="Chat with Data" subtitle="Ask questions about your active dataset and latest playground prediction." />
      {error && <ErrorState error={error} />}
      <section className="panel chat-panel">
        <div className="chat-toolbar">
          <label className="field"><span>Active dataset</span><select value={datasetId} onChange={(event) => setDatasetId(event.target.value)}><option value="">No dataset</option>{datasets.map((dataset) => <option value={dataset.id} key={dataset.id}>{dataset.name}</option>)}</select></label>
          <label className="field"><span>Hugging Face model</span><select value={modelId} onChange={(event) => setModelId(event.target.value)}><option value="">Use configured default model</option>{savedModelIds.map((id) => <option value={id} key={id}>{id}</option>)}</select></label>
          <label className="field"><span>Hugging Face token (optional)</span><input type="password" autoComplete="off" value={hfToken} onChange={(event) => setHfToken(event.target.value)} placeholder="hf_..." /></label>
        </div>
        <form className="add-hf-model" onSubmit={addModel}>
          <label className="field"><span>Add a Hugging Face model</span><input value={newModelId} onChange={(event) => setNewModelId(event.target.value)} placeholder="organization/model-repo" /></label>
          <button className="button" type="submit"><Plus size={15} />Add model</button>
          <p>Enter a public Hub repository ID supported by an Inference Provider. Model IDs are saved in this browser; weights are not uploaded.</p>
        </form>
        <div className="chat-starters">
          <span><Sparkles size={14} /> Try asking</span>
          {starterPrompts.map((prompt) => <button className="starter-button" key={prompt} onClick={() => void sendMessage(prompt)} disabled={busy}>{prompt}</button>)}
        </div>
        <div className="chat-history" aria-live="polite">
          {messages.map((message, index) => <article className={`chat-message ${message.role}`} key={`${index}-${message.role}`}><span>{message.role === 'user' ? 'You' : 'NeuralLab AI'}</span><p>{message.content}</p></article>)}
          {busy && <div className="chat-pending"><span className="spinner" />Thinking with Hugging Face Inference API...</div>}
          <div ref={bottomRef} />
        </div>
        <form className="chat-compose" onSubmit={submit}>
          <textarea value={input} onChange={(event) => setInput(event.target.value)} placeholder="Ask a question about your data..." aria-label="Message" rows={2} />
          <button className="button primary" disabled={busy || !input.trim()}><Send size={15} />{busy ? 'Sending...' : 'Send'}</button>
        </form>
        <p className="chat-footnote">The model runs remotely through the Hugging Face Inference API. A token entered here stays in page memory, is sent with chat requests, and is never saved in browser storage. Data context is limited to a small sample and summary.</p>
      </section>
    </div>
  )
}

const hfStarterPrompts = [
  'Explain a machine learning concept',
  'Write a Python function',
  'Help me debug some code',
]

function HuggingFacePlayground() {
  const [initialModels] = useState(readSavedModels)
  const [savedModelIds, setSavedModelIds] = useState(initialModels.ids)
  const [modelId, setModelId] = useState(initialModels.ids[0] ?? '')
  const [newModelId, setNewModelId] = useState('')
  const [hfToken, setHfToken] = useState('')
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: 'assistant', content: 'Ask the selected Hugging Face model anything. This chat is independent of uploaded datasets and predictions.' },
  ])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(initialModels.error)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, busy])

  const addModel = (event: FormEvent) => {
    event.preventDefault()
    const repositoryId = newModelId.trim()
    if (!repositoryId) {
      setError('Enter a Hugging Face model repository ID.')
      return
    }
    const next = [...new Set([...savedModelIds, repositoryId])]
    setSavedModelIds(next)
    setModelId(repositoryId)
    setNewModelId('')
    try {
      window.localStorage.setItem(savedModelsStorageKey, JSON.stringify(next))
      setError('')
    } catch {
      setError('Model selected for this session, but could not be saved in browser storage.')
    }
  }

  const removeModel = () => {
    if (!modelId) return
    const next = savedModelIds.filter((id) => id !== modelId)
    setSavedModelIds(next)
    setModelId(next[0] ?? '')
    try {
      window.localStorage.setItem(savedModelsStorageKey, JSON.stringify(next))
      setError('')
    } catch {
      setError('Model removed for this session, but browser storage could not be updated.')
    }
  }

  const sendMessage = async (content: string) => {
    const prompt = content.trim()
    if (!prompt || busy) return
    const conversation = [...messages, { role: 'user' as const, content: prompt }]
    setMessages(conversation)
    setInput('')
    setBusy(true)
    setError('')
    try {
      const response = await api.chat(conversation, undefined, modelId.trim(), hfToken, null, 'standalone')
      setMessages([...conversation, { role: 'assistant', content: response.reply }])
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const submit = (event: FormEvent) => {
    event.preventDefault()
    void sendMessage(input)
  }

  return (
    <div className="page chat-page">
      <PageHeading title="Hugging Face playground" subtitle="Chat directly with a Hugging Face model without dataset or prediction context." />
      {error && <ErrorState error={error} />}
      <section className="panel chat-panel">
        <div className="chat-toolbar">
          <label className="field"><span>Hugging Face model</span><select value={modelId} onChange={(event) => setModelId(event.target.value)}><option value="">Use configured default model</option>{savedModelIds.map((id) => <option value={id} key={id}>{id}</option>)}</select></label>
          <label className="field"><span>Hugging Face token (optional)</span><input type="password" autoComplete="off" value={hfToken} onChange={(event) => setHfToken(event.target.value)} placeholder="hf_..." /></label>
          {modelId && <button className="button hf-remove-model" type="button" onClick={removeModel}><Trash2 size={15} />Remove model</button>}
        </div>
        <form className="add-hf-model" onSubmit={addModel}>
          <label className="field"><span>Add a Hugging Face model</span><input value={newModelId} onChange={(event) => setNewModelId(event.target.value)} placeholder="organization/model-repo" /></label>
          <button className="button" type="submit"><Plus size={15} />Add model</button>
          <p>Enter a public Hub repository ID supported by an Inference Provider. Model IDs are saved in this browser; weights are not uploaded.</p>
        </form>
        <div className="chat-starters">
          <span><Sparkles size={14} /> Try asking</span>
          {hfStarterPrompts.map((prompt) => <button className="starter-button" key={prompt} onClick={() => void sendMessage(prompt)} disabled={busy}>{prompt}</button>)}
        </div>
        <div className="chat-history" aria-live="polite">
          {messages.map((message, index) => <article className={`chat-message ${message.role}`} key={`${index}-${message.role}`}><span>{message.role === 'user' ? 'You' : 'Hugging Face model'}</span><p>{message.content}</p></article>)}
          {busy && <div className="chat-pending"><span className="spinner" />Thinking with Hugging Face Inference API...</div>}
          <div ref={bottomRef} />
        </div>
        <form className="chat-compose" onSubmit={submit}>
          <textarea value={input} onChange={(event) => setInput(event.target.value)} placeholder="Ask the model a question..." aria-label="Message" rows={2} />
          <button className="button primary" disabled={busy || !input.trim()}><Send size={15} />{busy ? 'Sending...' : 'Send'}</button>
        </form>
        <p className="chat-footnote">The model runs remotely through the Hugging Face Inference API. This section sends only the conversation; it does not include workspace datasets or latest predictions.</p>
      </section>
    </div>
  )
}

export default function App() {
  return <BrowserRouter><Shell /></BrowserRouter>
}
