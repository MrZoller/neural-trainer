import { useEffect, useReducer, useState } from 'react'
import { getJSON, openRunSocket, postJSON } from '../api.js'
import DrawCanvas from './DrawCanvas.jsx'
import RunCharts, { LayerStatsTable } from './RunCharts.jsx'
import StatusBadge from './StatusBadge.jsx'
import UploadPredict from './UploadPredict.jsx'

const RESUMABLE = new Set(['stopped', 'killed', 'failed', 'interrupted'])

const initial = {
  status: null,
  device: null,
  params: null,
  error: null,
  batch: [],
  epochs: [],
  layers: null,
  hasCheckpoint: false,
}

// Everything on screen derives from the persisted event stream — a refresh
// replays it identically (DESIGN.md §4).
function reduce(state, event) {
  const p = event.payload
  switch (event.type) {
    case 'run_status':
      return {
        ...state,
        status: p.status,
        device: p.device ?? state.device,
        params: p.params ?? state.params,
        error: p.error ?? state.error,
      }
    case 'batch_metrics':
      return { ...state, batch: [...state.batch, p] }
    case 'epoch_metrics':
      return { ...state, epochs: [...state.epochs, p] }
    case 'layer_stats':
      return { ...state, layers: p.layers }
    case 'checkpoint_saved':
      return { ...state, hasCheckpoint: true }
    default:
      return state
  }
}

export default function RunView({ runId, onNavigate }) {
  const [state, dispatch] = useReducer(reduce, initial)
  const [run, setRun] = useState(null)
  const [actionError, setActionError] = useState(null)

  useEffect(() => {
    getJSON(`/api/runs/${runId}`)
      .then(setRun)
      .catch(() => {})
    return openRunSocket(runId, dispatch)
  }, [runId])

  const act = async (action) => {
    setActionError(null)
    try {
      await postJSON(`/api/runs/${runId}/actions`, { action })
    } catch (e) {
      setActionError(e.message)
    }
  }

  const resume = async () => {
    setActionError(null)
    try {
      const child = await postJSON('/api/runs', { parent_run_id: runId })
      onNavigate({ name: 'run', id: child.id })
    } catch (e) {
      setActionError(e.message)
    }
  }

  const canStop = state.status === 'running'
  const canKill = ['queued', 'running', 'stopping'].includes(state.status)
  const canResume = RESUMABLE.has(state.status) && state.hasCheckpoint
  const config = run?.config

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3 flex-wrap">
        <button
          className="text-slate-400 hover:text-slate-200 text-sm"
          onClick={() => onNavigate({ name: 'home' })}
        >
          ← runs
        </button>
        <h2 className="font-mono text-sm text-slate-300">{runId}</h2>
        {state.status && <StatusBadge status={state.status} />}
        {state.device && (
          <span className="text-xs text-slate-500">
            {state.device.toUpperCase()} · {state.params?.toLocaleString()} params
            {config &&
              ` · lr ${config.lr} · batch ${config.batch_size} · hidden [${config.hidden}]`}
          </span>
        )}
        <span className="flex-1" />
        {canStop && (
          <button
            className="bg-amber-800 hover:bg-amber-700 rounded px-3 py-1 text-sm"
            onClick={() => act('stop')}
          >
            Stop (checkpoint &amp; exit)
          </button>
        )}
        {canKill && (
          <button
            className="bg-rose-900 hover:bg-rose-800 rounded px-3 py-1 text-sm"
            onClick={() => act('kill')}
          >
            Kill
          </button>
        )}
        {canResume && (
          <button
            className="bg-sky-700 hover:bg-sky-600 rounded px-3 py-1 text-sm"
            onClick={resume}
          >
            Resume from checkpoint
          </button>
        )}
      </div>

      {run?.parent_run_id && (
        <p className="text-xs text-slate-500">
          resumed from{' '}
          <button
            className="underline hover:text-slate-300"
            onClick={() => onNavigate({ name: 'run', id: run.parent_run_id })}
          >
            {run.parent_run_id}
          </button>
        </p>
      )}
      {actionError && <p className="text-rose-400 text-sm">{actionError}</p>}
      {state.error && (
        <pre className="bg-rose-950/50 border border-rose-900 rounded-lg p-3 text-xs text-rose-300 whitespace-pre-wrap">
          {state.error}
        </pre>
      )}

      <RunCharts epochs={state.epochs} batch={state.batch} />
      <LayerStatsTable layers={state.layers} />
      {(state.hasCheckpoint || run?.latest_checkpoint) &&
        (config?.track === 'custom_finetune' ? (
          <UploadPredict runId={runId} />
        ) : (
          <DrawCanvas runId={runId} />
        ))}
    </div>
  )
}
