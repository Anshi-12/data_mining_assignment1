import { useEffect, useState } from 'react'
import { getHealth } from '../services/api'

export function HealthCard() {
  const [state, setState] = useState({ status: 'loading', data: null, error: null })

  useEffect(() => {
    let active = true
    getHealth()
      .then((data) => active && setState({ status: 'ready', data, error: null }))
      .catch((error) => active && setState({ status: 'error', data: null, error: error.message }))
    return () => { active = false }
  }, [])

  if (state.status === 'loading') {
    return <div className="card"><strong>Backend status</strong><p>Connecting to FastAPI…</p></div>
  }

  if (state.status === 'error') {
    return (
      <div className="card error">
        <strong>Backend unavailable</strong>
        <p>{state.error}</p>
        <small>Start FastAPI on port 8006, then refresh this page.</small>
      </div>
    )
  }

  return (
    <div className="card success" data-testid="health-ready">
      <strong>Backend connected</strong>
      <p>{state.data.service} returned <code>{state.data.status}</code>.</p>
      <dl>
        <div><dt>API phase</dt><dd>{state.data.phase}</dd></div>
        <div><dt>ML ready</dt><dd>{String(state.data.ml_ready)}</dd></div>
        <div><dt>Scorer</dt><dd>{state.data.scorer}</dd></div>
        <div><dt>Artifact schema</dt><dd>{state.data.artifact_schema_version}</dd></div>
      </dl>
    </div>
  )
}
