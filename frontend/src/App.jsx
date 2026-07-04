import { useCallback, useEffect, useState } from 'react'
import { getJSON } from './api.js'
import NewRunPanel from './components/NewRunPanel.jsx'
import RunView from './components/RunView.jsx'
import StatusBadge from './components/StatusBadge.jsx'

export default function App() {
  const [device, setDevice] = useState(null)
  const [offline, setOffline] = useState(false)
  const [runs, setRuns] = useState([])
  const [runId, setRunId] = useState(null)

  const refresh = useCallback(() => {
    getJSON('/api/runs').then(setRuns).catch(() => {})
  }, [])

  useEffect(() => {
    getJSON('/api/device').then(setDevice).catch(() => setOffline(true))
    refresh()
    const t = setInterval(refresh, 5000)
    return () => clearInterval(t)
  }, [refresh])

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans">
      <header className="border-b border-slate-800 px-6 py-3 flex items-center gap-4">
        <h1 className="font-bold tracking-tight">neural-trainer</h1>
        {device ? (
          <span className="px-2.5 py-0.5 rounded-full bg-emerald-900/60 text-emerald-300 text-xs">
            {device.device.toUpperCase()} · {device.name} · torch {device.torch}
          </span>
        ) : offline ? (
          <span className="px-2.5 py-0.5 rounded-full bg-rose-900/60 text-rose-300 text-xs">
            backend offline
          </span>
        ) : null}
      </header>

      <main className="max-w-5xl mx-auto p-6">
        {runId ? (
          <RunView runId={runId} onNavigate={setRunId} />
        ) : (
          <div className="space-y-6">
            <NewRunPanel onCreated={(run) => { refresh(); setRunId(run.id) }} />
            <div className="bg-slate-900 rounded-xl border border-slate-800">
              <h2 className="font-semibold px-5 pt-4 pb-2">Runs</h2>
              {runs.length === 0 ? (
                <p className="text-slate-500 text-sm px-5 pb-4">No runs yet — start one above.</p>
              ) : (
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-slate-500 text-left text-xs">
                      <th className="px-5 py-1">run</th>
                      <th>track</th>
                      <th>epochs</th>
                      <th>status</th>
                      <th>created</th>
                    </tr>
                  </thead>
                  <tbody>
                    {runs.map((r) => (
                      <tr
                        key={r.id}
                        className="border-t border-slate-800 hover:bg-slate-800/50 cursor-pointer"
                        onClick={() => setRunId(r.id)}
                      >
                        <td className="px-5 py-2 font-mono text-xs">
                          {r.id}
                          {r.parent_run_id && <span className="text-slate-600"> ↩</span>}
                        </td>
                        <td>{r.config.track}</td>
                        <td>{r.config.epochs}</td>
                        <td><StatusBadge status={r.status} /></td>
                        <td className="text-slate-500 text-xs pr-5">
                          {new Date(r.created_at).toLocaleTimeString()}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
