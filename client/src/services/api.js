async function request(path, options = {}) {
  const response = await fetch(path, options)
  let body = null
  try { body = await response.json() } catch { body = null }
  if (!response.ok) {
    const message = body?.detail || `${path} failed with HTTP ${response.status}`
    const error = new Error(message)
    error.status = response.status
    throw error
  }
  return body
}

export const getHealth = () => request('/api/health')
export const getMetadata = () => request('/api/metadata')
export const getBenchmarks = () => request('/api/benchmarks')
export const getManifold = () => request('/api/manifold')
export const getSearchHistory = () => request('/api/autoresearch/history')
export const getTopAnomalies = () => request('/api/anomalies/top')
export const getRetrainOptions = () => request('/api/retrain/options')
export const scoreTelemetry = (payload) => request('/api/anomaly/score', {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
})
export const retrain = (payload) => request('/api/retrain', {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
})
