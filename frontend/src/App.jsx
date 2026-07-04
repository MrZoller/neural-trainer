import { useCallback, useEffect, useState } from 'react'
import { getJSON, postJSON } from './api.js'
import DatasetView from './components/DatasetView.jsx'
import NewRunPanel from './components/NewRunPanel.jsx'
import RunView from './components/RunView.jsx'
import StatusBadge from './components/StatusBadge.jsx'

export default function App() {
  const [device, setDevice] = useState(null)
  const [offline, setOffline] = useState(false)
  const [runs, setRuns] = useState([])
  const [datasets, setDatasets] = useState([])
  const [newDataset, setNewDataset] = useState('')
  const [view, setView] = useState({ name: 'home' })

  const refresh = useCallback(() => {
    getJSON('/api/runs').then(setRuns).catch(() => {})
    getJSON('/api/datasets').then(setDatasets).catch(() => {})
  }, [])

  useEffect(() => {
    getJSON('/api/device').then(setDevice).catch(() => setOffline(true))
    refresh()
    const t = setInterval(refresh, 5000)
    return () => clearInterval(t)
  }, [refresh])

  const createDataset = async () => {
    if (!newDataset.trim()) return
    const ds = await postJSON('/api/datasets', { name: newDataset.trim() })
    setNewDataset('')
    refresh()
    setView({ name: 'dataset', id: ds.id })
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans">
      <header className="border-b border-slate-800 px-6 py-3 flex items-center gap-4">
        <button className="font-bold tracking-tight" onClick={() => setView({ name: 'home' })}>
          neural-trainer
        </button>
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
        {view.name === 'run' && <RunView runId={view.id} onNavigate={setView} />}
        {view.name === 'dataset' && <DatasetView datasetId={view.id} onNavigate={setView} />}
        {view.name === 'home' && (
          <div className="space-y-6">
            <div className="grid md:grid-cols-2 gap-6">
              <NewRunPanel onCreated={(run) => { refresh(); setView({ name: 'run', id: run.id }) }} />
              <div className="bg-slate-900 rounded-xl p-5 border border-slate-800">
                <h2 className="font-semibold mb-1">Teach it something of yours</h2>
                <p className="text-sm text-slate-400 mb-4">
                  Track 2: curate your own photos into classes, fine-tune a pretrained
                  network, and use it on new images — ~30+ photos per class is enough.
                </p>
                {datasets.map((d) => (
                  <button key={d.id}
                    className="w-full flex items-center gap-3 text-left px-3 py-2 rounded-lg hover:bg-slate-800 text-sm"
                    onClick={() => setView({ name: 'dataset', id: d.id })}>
                    <span className="flex-1">{d.name}</span>
                    <span className="text-xs text-slate-500">
                      {d.n_labeled || 0} labeled / {d.n_images || 0} images
                    </span>
                  </button>
                ))}
                <div className="flex gap-2 mt-3">
                  <input
                    className="flex-1 bg-slate-800 rounded px-2 py-1 text-sm border border-slate-700 focus:border-sky-600 outline-none"
                    placeholder="new dataset name (e.g. my-pets)"
                    value={newDataset}
                    onChange={(e) => setNewDataset(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && createDataset()}
                  />
                  <button className="bg-sky-700 hover:bg-sky-600 rounded px-3 py-1 text-sm"
                          onClick={createDataset}>
                    Create
                  </button>
                </div>
              </div>
            </div>

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
                        onClick={() => setView({ name: 'run', id: r.id })}
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
