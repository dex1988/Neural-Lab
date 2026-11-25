import { useCallback, useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import {
  BookOpen, Check, ChevronRight, CircleAlert, Clock3, ExternalLink, FileText,
  Globe2, LoaderCircle, MessageCircle, Plus, RefreshCw, Search, Send, Sparkles,
  Trash2, X,
} from 'lucide-react'
import { api } from './services/api'
import type { WebAIChatMessage, WebAIModel, WebAISource, WebAISourceDetail } from './services/api'
import './WebAITrainer.css'
import './WebAITrainerStatus.css'
import './WebAITrainerCredentials.css'

type SourceKind = 'url' | 'query' | 'text'
type ChatTurn = WebAIChatMessage & { sources?: { title: string; url: string }[] }

function messageOf(error: unknown) {
  return error instanceof Error ? error.message : 'Something went wrong. Please try again.'
}

function formatDate(value: string | null) {
  return value ? new Date(value).toLocaleString() : 'Not retrieved yet'
}

function sourceStatusLabel(status: WebAISource['status']) {
  return status.charAt(0).toUpperCase() + status.slice(1)
}

function sourceActivity(source: WebAISource) {
  if (source.kind !== 'query') return formatDate(source.retrieved_at)
  if (source.status === 'processed') return 'Search completed; pages indexed'
  if (source.status === 'failed') return 'Search failed'
  if (source.status === 'retrieving') return 'Searching the web…'
  if (source.status === 'retrieved' || source.status === 'processing') return 'Search results found; indexing pages'
  return 'Search not started'
}

export default function WebAITrainer() {
  const [models, setModels] = useState<WebAIModel[]>([])
  const [activeId, setActiveId] = useState<number | null>(null)
  const [sources, setSources] = useState<WebAISource[]>([])
  const [creating, setCreating] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [topic, setTopic] = useState('')
  const [initialUrls, setInitialUrls] = useState('')
  const [initialQueries, setInitialQueries] = useState('')
  const [sourceKind, setSourceKind] = useState<SourceKind>('url')
  const [sourceValue, setSourceValue] = useState('')
  const [hfToken, setHfToken] = useState('')
  const [braveSearchEnabled, setBraveSearchEnabled] = useState(false)
  const [braveSearchApiKey, setBraveSearchApiKey] = useState('')
  const [sourceDetail, setSourceDetail] = useState<WebAISourceDetail | null>(null)
  const [messages, setMessages] = useState<ChatTurn[]>([])
  const [question, setQuestion] = useState('')
  const [busy, setBusy] = useState(false)
  const chatEnd = useRef<HTMLDivElement>(null)
  const activeModel = models.find((model) => model.id === activeId) ?? null
  const learning = activeModel?.status === 'queued' || activeModel?.status === 'running'
  const activeBraveSearchApiKey = braveSearchEnabled ? braveSearchApiKey : ''

  const loadModels = useCallback(async () => {
    const records = await api.webAIModels()
    setModels(records)
    if (activeId && !records.some((model) => model.id === activeId)) setActiveId(null)
  }, [activeId])

  const loadSources = useCallback(async (id: number) => {
    setSources(await api.webAISources(id))
  }, [])

  useEffect(() => {
    api.webAIModels()
      .then(setModels)
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    if (!activeId) return
    let cancelled = false
    api.webAISources(activeId)
      .then((items) => { if (!cancelled) setSources(items) })
      .catch((err: Error) => { if (!cancelled) setError(err.message) })
    return () => { cancelled = true }
  }, [activeId])

  useEffect(() => {
    if (!activeId || !learning) return
    let cancelled = false
    const pollProgress = async () => {
      try {
        const [records, items] = await Promise.all([
          api.webAIModels(),
          api.webAISources(activeId),
        ])
        if (cancelled) return
        setModels(records)
        setSources(items)
      } catch (err) {
        if (!cancelled) setError(messageOf(err))
      }
    }
    void pollProgress()
    const interval = window.setInterval(() => void pollProgress(), 1000)
    return () => {
      cancelled = true
      window.clearInterval(interval)
    }
  }, [activeId, learning])

  useEffect(() => {
    chatEnd.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages])

  const createModel = async (event: FormEvent) => {
    event.preventDefault()
    setBusy(true); setError(''); setNotice('')
    try {
      const urls = initialUrls.split('\n').map((url) => url.trim()).filter(Boolean)
      const model = await api.createWebAIModel({
        name: name.trim(),
        description: description.trim(),
        topic: topic.trim(),
        urls,
        queries: initialQueries.split('\n').map((query) => query.trim()).filter(Boolean),
      })
      setModels((previous) => [model, ...previous])
      setActiveId(model.id); setCreating(false)
      setName(''); setDescription(''); setTopic(''); setInitialUrls(''); setInitialQueries('')
      if (urls.length > 0) {
        const updated = await api.learnWebAI(model.id, activeBraveSearchApiKey)
        setModels((previous) => previous.map((item) => item.id === updated.id ? updated : item))
        setNotice('AI model created. Website crawl and indexing started.')
      } else {
        setNotice('Your AI model is ready. Select sources and start learning.')
      }
    } catch (err) {
      setError(messageOf(err))
    } finally { setBusy(false) }
  }

  const addSource = async (event: FormEvent) => {
    event.preventDefault()
    if (!activeId || !sourceValue.trim()) return
    const kind = sourceKind
    setBusy(true); setError(''); setNotice('')
    try {
      await api.addWebAISource(activeId, kind, sourceValue.trim())
      setSourceValue('')
      if (kind === 'url') {
        const updated = await api.learnWebAI(activeId, activeBraveSearchApiKey)
        setModels((previous) => previous.map((model) => model.id === updated.id ? updated : model))
        setNotice('Website added. Crawl and indexing started.')
      } else {
        setNotice(kind === 'query' ? 'Search topic added to this AI.' : 'Information added to this AI.')
      }
      await loadSources(activeId)
      await loadModels()
    } catch (err) { setError(messageOf(err)) } finally { setBusy(false) }
  }

  const startLearning = async (refresh = false) => {
    if (!activeId) return
    setBusy(true); setError(''); setNotice('')
    try {
      const updated = refresh
        ? await api.refreshWebAI(activeId, activeBraveSearchApiKey)
        : await api.learnWebAI(activeId, activeBraveSearchApiKey)
      setModels((previous) => previous.map((model) => model.id === updated.id ? updated : model))
      setNotice(refresh ? 'Knowledge refresh started.' : 'Research and indexing started.')
    } catch (err) { setError(messageOf(err)) } finally { setBusy(false) }
  }

  const toggleSource = async (source: WebAISource) => {
    if (!activeId) return
    setError('')
    try {
      await api.selectWebAISource(activeId, source.id, !source.selected)
      await loadSources(activeId)
      await loadModels()
    } catch (err) { setError(messageOf(err)) }
  }

  const removeSource = async (source: WebAISource) => {
    if (!activeId || !window.confirm('Remove this source and its indexed knowledge?')) return
    setError('')
    try {
      await api.deleteWebAISource(activeId, source.id)
      setSourceDetail(null)
      await loadSources(activeId)
      await loadModels()
    } catch (err) { setError(messageOf(err)) }
  }

  const inspectSource = async (source: WebAISource) => {
    if (!activeId) return
    setError('')
    try { setSourceDetail(await api.inspectWebAISource(activeId, source.id)) }
    catch (err) { setError(messageOf(err)) }
  }

  const deleteModel = async (model: WebAIModel) => {
    if (!window.confirm(`Delete "${model.name}" and all of its sources and indexed knowledge?`)) return
    setError('')
    try {
      await api.deleteWebAIModel(model.id)
      setModels((previous) => previous.filter((item) => item.id !== model.id))
      if (activeId === model.id) {
        setActiveId(null); setMessages([]); setSourceDetail(null)
      }
    } catch (err) { setError(messageOf(err)) }
  }

  const sendMessage = async (event: FormEvent) => {
    event.preventDefault()
    if (!activeId || !question.trim() || busy || !activeModel?.chat_ready) return
    const nextMessages: WebAIChatMessage[] = [...messages, { role: 'user', content: question.trim() }]
    setMessages(nextMessages); setQuestion(''); setBusy(true); setError('')
    try {
      const result = await api.chatWebAI(activeId, nextMessages, hfToken)
      setMessages([...nextMessages, { role: 'assistant', content: result.reply, sources: result.sources }])
    } catch (err) {
      setMessages(nextMessages.slice(0, -1))
      setQuestion(nextMessages[nextMessages.length - 1].content)
      setError(messageOf(err))
    } finally { setBusy(false) }
  }

  return (
    <div className="page web-ai-page">
      <div className="web-ai-heading">
        <div>
          <div className="eyebrow"><Globe2 size={14} /> INDEPENDENT KNOWLEDGE WORKSPACE</div>
          <h1>Web AI Trainer</h1>
          <p>Give an AI your own information or websites, then chat with answers grounded in its knowledge.</p>
        </div>
        <button className="button primary" onClick={() => { setCreating(true); setActiveId(null); setError(''); setNotice('') }}>
          <Plus size={16} /> Create AI model
        </button>
      </div>

      <div className="rag-explainer">
        <Sparkles size={17} />
        <p><strong>Knowledge retrieval, not model retraining.</strong> NeuralLab builds a private, searchable knowledge base (RAG) for each AI. It does not retrain the underlying foundation model from scratch.</p>
      </div>
      <div className="web-ai-provider-note">
        <CircleAlert size={15} />
        <span>Website URLs crawl up to 25 allowed pages on the same site without Brave Search. Enable Brave only for search queries. Keys entered below are kept only in this page; server environment keys remain available as fallbacks.</span>
      </div>
      <div className="web-ai-credentials">
        <label>
          <span>Hugging Face token <small>for chat</small></span>
          <input type="password" autoComplete="new-password" value={hfToken} onChange={(event) => setHfToken(event.target.value)} placeholder="hf_…" />
        </label>
        <div className="web-ai-brave-credential">
          <label className="web-ai-brave-toggle">
            <input type="checkbox" checked={braveSearchEnabled} onChange={(event) => setBraveSearchEnabled(event.target.checked)} />
            <span>Enable Brave Search API key <small>for search queries</small></span>
          </label>
          {braveSearchEnabled && <label>
            <span>Brave Search API key</span>
            <input type="password" autoComplete="new-password" value={braveSearchApiKey} onChange={(event) => setBraveSearchApiKey(event.target.value)} placeholder="Enter your Brave Search key" />
          </label>}
        </div>
      </div>
      {error && <div className="web-ai-alert error"><CircleAlert size={16} /><span>{error}</span><button aria-label="Dismiss error" onClick={() => setError('')}><X size={15} /></button></div>}
      {notice && <div className="web-ai-alert success"><Check size={16} /><span>{notice}</span><button aria-label="Dismiss message" onClick={() => setNotice('')}><X size={15} /></button></div>}

      <div className="web-ai-layout">
        <aside className="panel web-ai-library">
          <div className="web-ai-section-title">          <div><h2>Model library</h2><p>{models.length} saved AI{models.length === 1 ? '' : 's'}</p></div><button className="web-ai-small-button" title="Create model" onClick={() => { setCreating(true); setActiveId(null) }}><Plus size={16} /></button></div>
          {loading ? <div className="web-ai-loading"><LoaderCircle className="web-ai-spin" size={18} /> Loading models</div>
            : models.length === 0 ? <div className="web-ai-library-empty"><BookOpen size={22} /><strong>Your model library is empty</strong><span>Create your first AI assistant to get started.</span></div>
              : <div className="web-ai-model-list">{models.map((model) => (
                <button key={model.id} className={`web-ai-model-card ${activeId === model.id ? 'active' : ''}`} onClick={() => { setCreating(false); setActiveId(model.id); setMessages([]); setSourceDetail(null); setError(''); setNotice('') }}>
                  <span className="web-ai-model-icon"><Sparkles size={16} /></span>
                  <span className="web-ai-model-copy"><strong>{model.name}</strong><span>{model.topic}</span><small>{model.source_count} sources · {model.chunk_count} chunks</small></span>
                  <span className={`web-ai-status-dot ${model.status}`} title={model.status} />
                </button>
              ))}</div>}
        </aside>

        <section className="web-ai-main">
          {creating ? (
            <section className="panel web-ai-create">
              <div className="web-ai-section-title"><div><h2>Create your AI</h2><p>Give it a focus, then choose its first sources.</p></div><button className="web-ai-close" aria-label="Close form" onClick={() => setCreating(false)}><X size={17} /></button></div>
              <form onSubmit={createModel}>
                <div className="web-ai-form-grid">
                  <label className="field"><span>AI name</span><input required maxLength={120} value={name} onChange={(event) => setName(event.target.value)} placeholder="e.g. Garden Guide" /></label>
                  <label className="field"><span>What should it learn about?</span><input required maxLength={2000} value={topic} onChange={(event) => setTopic(event.target.value)} placeholder="e.g. Indoor herbs and plant care" /></label>
                </div>
                <label className="field"><span>Description <small>(optional)</small></span><textarea maxLength={1000} value={description} onChange={(event) => setDescription(event.target.value)} placeholder="Describe who this assistant is for and what it can help with." /></label>
                <div className="web-ai-form-grid">
                  <label className="field"><span>Website URLs <small>(one per line)</small></span><textarea value={initialUrls} onChange={(event) => setInitialUrls(event.target.value)} placeholder={'https://example.com/guides\nhttps://another-site.org/article'} /></label>
                  <label className="field"><span>Search queries <small>(one per line)</small></span><textarea value={initialQueries} onChange={(event) => setInitialQueries(event.target.value)} placeholder={'beginner guide to indoor herb gardens\nhow to care for basil indoors'} /></label>
                </div>
                <div className="web-ai-form-foot"><span>Only public pages allowed by their robots.txt are retrieved. Respect site terms and copyright; use sources you have permission to process. Logins and paywalls are never bypassed.</span><button className="button primary" disabled={busy || !name.trim() || !topic.trim()}>{busy ? 'Creating…' : initialUrls.trim() ? 'Create AI & crawl websites' : 'Create AI model'}<ChevronRight size={15} /></button></div>
              </form>
            </section>
          ) : activeModel ? (
            <>
              <section className="panel web-ai-model-overview">
                <div className="web-ai-overview-top">
                  <span className="web-ai-large-icon"><Sparkles size={22} /></span>
                  <div className="web-ai-overview-name"><span className={`status ${activeModel.status}`}>{activeModel.status}</span><h2>{activeModel.name}</h2><p>{activeModel.description || activeModel.topic}</p></div>
                  <button className="icon-danger" aria-label="Delete AI model" title="Delete AI model" disabled={learning} onClick={() => void deleteModel(activeModel)}><Trash2 size={17} /></button>
                </div>
                <div className="web-ai-stats">
                  <div><span>Sources discovered</span><strong>{activeModel.discovered_count}</strong></div>
                  <div><span>Retrieved</span><strong>{activeModel.retrieved_count}</strong></div>
                  <div><span>Sources indexed</span><strong>{activeModel.processed_count}</strong></div>
                  <div><span>Knowledge chunks</span><strong>{activeModel.chunk_count}</strong></div>
                  <div><span>Failed sources</span><strong className={activeModel.error_count ? 'has-errors' : ''}>{activeModel.error_count}</strong></div>
                </div>
                {(learning || activeModel.status_message) && <div className="web-ai-progress-line">
                  {learning && <LoaderCircle size={15} className="web-ai-spin" />}
                  <span>{activeModel.status_message || (learning ? 'Research is running…' : '')}</span>
                  {learning && <span className="web-ai-live-tag">LIVE</span>}
                </div>}
              </section>

              <div className="web-ai-workspace-grid">
                <section className="panel web-ai-sources-panel">
                  <div className="web-ai-section-title">
                    <div><h2>Knowledge sources</h2><p>Choose what this AI can use in its answers.</p></div>
                    <button className="web-ai-small-button" title="Refresh selected source content" disabled={busy || learning || sources.length === 0} onClick={() => void startLearning(true)}><RefreshCw size={15} /></button>
                  </div>
                  <form className="web-ai-add-source" onSubmit={addSource}>
                    <select aria-label="Source type" value={sourceKind} onChange={(event) => setSourceKind(event.target.value as SourceKind)}>
                      <option value="url">Website URL</option><option value="query">Search query</option><option value="text">Your information</option>
                    </select>
                    {sourceKind === 'text'
                      ? <textarea required maxLength={60000} value={sourceValue} onChange={(event) => setSourceValue(event.target.value)} placeholder="Paste notes, facts, or other information for this AI to learn…" rows={3} />
                      : <input required value={sourceValue} onChange={(event) => setSourceValue(event.target.value)} placeholder={sourceKind === 'url' ? 'https://website.com/article' : 'Add a topic to search'} />}
                    <button className="button" disabled={busy || learning}><Plus size={15} /><span>{sourceKind === 'url' ? 'Add & crawl' : 'Add'}</span></button>
                  </form>
                  <div className="web-ai-sources-toolbar">
                    <span><Check size={13} /> {sources.filter((source) => source.selected).length} included</span>
                    <button className="button primary" disabled={busy || learning || !sources.some((source) => source.selected && ['pending', 'failed'].includes(source.status))} onClick={() => void startLearning()}>
                      {learning ? <><LoaderCircle className="web-ai-spin" size={15} /> Learning…</> : <><Sparkles size={15} /> Start learning</>}
                    </button>
                  </div>
                  <div className="web-ai-sources-list">
                    {sources.length === 0 ? <div className="web-ai-empty"><Globe2 size={22} /><strong>No sources yet</strong><span>Add a website or search topic to give this AI knowledge.</span></div>
                      : sources.map((source) => (
                        <article className={`web-ai-source ${source.selected ? '' : 'excluded'}`} key={source.id}>
                          <label className="web-ai-source-check" title={source.selected ? 'Exclude source' : 'Include source'}>
                            <input type="checkbox" checked={source.selected} disabled={learning} onChange={() => void toggleSource(source)} />
                          </label>
                          <div className="web-ai-source-icon">{source.kind === 'query' ? <Search size={16} /> : source.kind === 'text' ? <FileText size={16} /> : <Globe2 size={16} />}</div>
                          <div className="web-ai-source-content">
                            <strong>{source.title || source.value}</strong>
                            <span>{source.kind === 'query' ? `Search · ${source.value}` : source.kind === 'text' ? 'Manually provided information' : source.url || source.value}</span>
                            <small><Clock3 size={11} /> {sourceActivity(source)} {source.chunk_count > 0 && `· ${source.chunk_count} chunks`}</small>
                            {source.error && <small className="web-ai-source-error"><CircleAlert size={11} /> {source.error}</small>}
                          </div>
                          <span className={`web-ai-source-status ${source.status}`}>{sourceStatusLabel(source.status)}</span>
                          {source.retrieved_at && <button className="web-ai-icon-action" title="Inspect source content" onClick={() => void inspectSource(source)}><FileText size={15} /></button>}
                          <button className="web-ai-icon-action danger" aria-label={`Remove ${source.title || source.value}`} onClick={() => void removeSource(source)} disabled={learning}><Trash2 size={15} /></button>
                        </article>
                      ))}
                  </div>
                  {activeModel.error_count > 0 && <div className="web-ai-errors-summary"><CircleAlert size={14} /> Failed sources and details are listed above; correct the issue or remove a source and try again.</div>}
                </section>

                <section className="panel web-ai-chat-panel">
                  <div className="web-ai-section-title"><div><h2><MessageCircle size={16} /> Chat with your AI</h2><p>Answers are grounded in this model's selected sources.</p></div><span className="web-ai-private-badge"><span /> Isolated knowledge</span></div>
                  <div className="web-ai-chat-messages">
                    {messages.length === 0 ? <div className="web-ai-chat-empty"><span><Sparkles size={20} /></span><strong>Ask about {activeModel.name}</strong><p>{activeModel.chat_ready ? 'Your AI will answer from its indexed knowledge and cite the source websites.' : 'Start learning from selected sources to enable grounded chat.'}</p><small>If it cannot find the answer in its sources, it will say so.</small></div>
                      : messages.map((message, index) => <div className={`web-ai-message ${message.role}`} key={`${index}-${message.role}`}><span className="web-ai-message-icon">{message.role === 'assistant' ? <Sparkles size={14} /> : 'You'}</span><div className="web-ai-message-body"><p>{message.content}</p>{message.sources && message.sources.length > 0 && <div className="web-ai-citations">{message.sources.map((source, citationIndex) => source.url ? <a key={`${source.url}-${citationIndex}`} href={source.url} target="_blank" rel="noreferrer">[{citationIndex + 1}] {source.title}<ExternalLink size={10} /></a> : <span key={`${source.title}-${citationIndex}`}>[{citationIndex + 1}] {source.title}</span>)}</div>}</div></div>)}
                    {busy && messages.at(-1)?.role === 'user' && <div className="web-ai-message assistant"><span className="web-ai-message-icon"><Sparkles size={14} /></span><p><LoaderCircle className="web-ai-spin" size={15} /> Searching knowledge…</p></div>}
                    <div ref={chatEnd} />
                  </div>
                  <form className="web-ai-chat-form" onSubmit={sendMessage}>
                    <textarea value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); event.currentTarget.form?.requestSubmit() } }} placeholder={activeModel.chat_ready ? 'Ask a question about this AI’s sources…' : 'Build this AI’s knowledge first'} disabled={!activeModel.chat_ready || busy} maxLength={4000} rows={2} />
                    <button className="button primary" aria-label="Send message" disabled={!question.trim() || !activeModel.chat_ready || busy}><Send size={15} /></button>
                  </form>
                  <div className="web-ai-chat-foot">Answers use only the selected source content. Source instructions are treated as untrusted information.</div>
                </section>
              </div>
              {sourceDetail && <div className="web-ai-inspector-backdrop" onClick={() => setSourceDetail(null)}>
                <section className="panel web-ai-inspector" onClick={(event) => event.stopPropagation()}>
                  <div className="web-ai-section-title"><div><h2>{sourceDetail.title || sourceDetail.value}</h2><p>{sourceDetail.kind === 'text' ? 'Manually provided information' : sourceDetail.url || sourceDetail.value} · Added {formatDate(sourceDetail.retrieved_at)}</p></div><button className="web-ai-close" aria-label="Close source details" onClick={() => setSourceDetail(null)}><X size={17} /></button></div>
                  <pre>{sourceDetail.content || 'No extracted text is available for this source.'}</pre>
                  {sourceDetail.url && <a className="web-ai-source-link" href={sourceDetail.url} target="_blank" rel="noreferrer"><ExternalLink size={14} /> Open original website</a>}
                </section>
              </div>}
            </>
          ) : (
            <section className="panel web-ai-welcome">
              <div className="web-ai-welcome-icon"><Globe2 size={28} /></div>
              <h2>Build an AI that knows what you care about</h2>
              <p>Create a focused assistant, add your own information or trusted public websites, then chat with answers grounded in its knowledge.</p>
              <ul><li><Check size={14} /> Each AI keeps its sources and knowledge separate</li><li><Check size={14} /> Inspect, refresh, include, or remove any source</li><li><Check size={14} /> No guessing: answers are grounded in retrieved pages</li></ul>
              <button className="button primary" onClick={() => setCreating(true)}><Plus size={16} /> Create your first AI</button>
            </section>
          )}
        </section>
      </div>
    </div>
  )
}
