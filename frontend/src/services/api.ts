export interface Dataset {
  id: number
  name: string
  filename: string
  size_bytes: number
  row_count: number
  column_count: number
  columns: string[]
  created_at: string
}

export interface ColumnInfo {
  name: string
  dtype: string
  missing: number
  unique: number
  example: string | null
}

export interface DatasetPreview {
  dataset: Dataset
  columns: ColumnInfo[]
  statistics: Record<string, unknown>
  preview: Record<string, unknown>[]
}

export interface TrainedModel {
  id: number
  name: string
  model_type: string
  problem_type: 'classification' | 'regression'
  feature_columns: string[]
  target_column: string
  hyperparameters: Record<string, unknown>
  metrics: Record<string, unknown>
  status: 'queued' | 'running' | 'completed' | 'failed'
  error: string | null
  duration_seconds: number | null
  created_at: string
}

export interface DashboardSummary {
  total_datasets: number
  total_models: number
  total_experiments: number
  completed_models: number
  recent_experiments: { id: number; name: string; description: string | null; created_at: string }[]
  best_model: { id: number; name: string; model_type: string; score: number; problem_type: string } | null
  training_history: { date: string; models: number }[]
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface LatestPrediction {
  model_id: number
  model_name: string
  problem_type: 'classification' | 'regression'
  target_column: string
  features: Record<string, unknown>
  prediction: unknown
  probabilities: Record<string, number> | null
}

export interface WebAIModel {
  id: number
  name: string
  description: string | null
  topic: string
  status: 'idle' | 'queued' | 'running' | 'completed' | 'failed'
  status_message: string | null
  created_at: string
  updated_at: string
  source_count: number
  chunk_count: number
  discovered_count: number
  retrieved_count: number
  processed_count: number
  error_count: number
  skipped_count: number
  chat_ready: boolean
}

export interface WebAISource {
  id: number
  model_id: number
  parent_id: number | null
  kind: 'url' | 'query' | 'webpage' | 'text'
  value: string
  url: string | null
  title: string | null
  status: 'pending' | 'retrieving' | 'retrieved' | 'processing' | 'processed' | 'failed'
  selected: boolean
  error: string | null
  retrieved_at: string | null
  chunk_count: number
}

export interface WebAISourceDetail extends WebAISource {
  content: string | null
}

export interface WebAIChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface WebAIChatReply {
  reply: string
  model: string
  sources: { title: string; url: string }[]
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, options)
  if (!response.ok) {
    let detail = `Request failed (${response.status})`
    try {
      const body = await response.json() as { detail?: string; message?: string; error?: string }
      detail = body.detail || body.message || body.error || detail
    } catch {
      // Keep the status-based fallback.
    }
    throw new Error(detail)
  }
  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

export const api = {
  dashboard: () => request<DashboardSummary>('/api/v1/dashboard/summary'),
  datasets: () => request<Dataset[]>('/api/v1/datasets'),
  dataset: (id: number) => request<DatasetPreview>(`/api/v1/datasets/${id}`),
  uploadDataset: (file: File, name: string) => {
    const form = new FormData()
    form.append('file', file)
    return request<Dataset>(`/api/v1/datasets?name=${encodeURIComponent(name)}`, {
      method: 'POST',
      body: form,
    })
  },
  deleteDataset: (id: number) => request<void>(`/api/v1/datasets/${id}`, { method: 'DELETE' }),
  experiments: () => request<{ id: number; name: string; description: string | null; created_at: string }[]>('/api/v1/experiments'),
  createExperiment: (name: string, description: string) => request<unknown>('/api/v1/experiments', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, description: description || null }),
  }),
  deleteExperiment: (id: number) => request<void>(`/api/v1/experiments/${id}`, { method: 'DELETE' }),
  models: () => request<TrainedModel[]>('/api/v1/models'),
  train: (body: Record<string, unknown>) => request<TrainedModel>('/api/v1/training', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }),
  deleteModel: (id: number) => request<void>(`/api/v1/models/${id}`, { method: 'DELETE' }),
  predict: (id: number, features: Record<string, unknown>) => request<{ prediction: unknown; probabilities: Record<string, number> | null }>(
    `/api/v1/models/${id}/predict`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features }),
    },
  ),
  chat: (messages: ChatMessage[], datasetId: number | undefined, modelId: string, hfToken: string, latestPrediction: LatestPrediction | null, contextMode: 'data' | 'standalone' = 'data') =>
    request<{ reply: string; model: string }>('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages,
        dataset_id: datasetId,
        model_id: modelId || undefined,
        hf_token: hfToken.trim() || undefined,
        latest_prediction: latestPrediction,
        context_mode: contextMode,
      }),
    }),
  webAIModels: () => request<WebAIModel[]>('/api/v1/web-ai/models'),
  createWebAIModel: (body: { name: string; description: string; topic: string; urls: string[]; queries: string[] }) =>
    request<WebAIModel>('/api/v1/web-ai/models', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  deleteWebAIModel: (id: number) => request<void>(`/api/v1/web-ai/models/${id}`, { method: 'DELETE' }),
  webAISources: (id: number) => request<WebAISource[]>(`/api/v1/web-ai/models/${id}/sources`),
  addWebAISource: (id: number, kind: 'url' | 'query' | 'text', value: string) =>
    request<WebAISource>(`/api/v1/web-ai/models/${id}/sources`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ kind, value }),
    }),
  inspectWebAISource: (modelId: number, sourceId: number) =>
    request<WebAISourceDetail>(`/api/v1/web-ai/models/${modelId}/sources/${sourceId}`),
  selectWebAISource: (modelId: number, sourceId: number, selected: boolean) =>
    request<WebAISource>(`/api/v1/web-ai/models/${modelId}/sources/${sourceId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ selected }),
    }),
  deleteWebAISource: (modelId: number, sourceId: number) =>
    request<void>(`/api/v1/web-ai/models/${modelId}/sources/${sourceId}`, { method: 'DELETE' }),
  learnWebAI: (id: number, braveSearchApiKey: string) =>
    request<WebAIModel>(`/api/v1/web-ai/models/${id}/learn`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ brave_search_api_key: braveSearchApiKey.trim() || undefined }),
    }),
  refreshWebAI: (id: number, braveSearchApiKey: string) =>
    request<WebAIModel>(`/api/v1/web-ai/models/${id}/refresh`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ brave_search_api_key: braveSearchApiKey.trim() || undefined }),
    }),
  chatWebAI: (id: number, messages: WebAIChatMessage[], hfToken: string) =>
    request<WebAIChatReply>(`/api/v1/web-ai/models/${id}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: messages.map(({ role, content }) => ({ role, content })),
        hf_token: hfToken.trim() || undefined,
      }),
    }),
}
