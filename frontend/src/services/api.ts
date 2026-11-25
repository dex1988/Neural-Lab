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
}
