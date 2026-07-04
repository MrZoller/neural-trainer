import { useState } from 'react'
import { postJSON } from '../api.js'

// Every knob gets a plain-language explanation (DESIGN.md §6, config panel).
const FIELDS = [
  {
    key: 'epochs', label: 'Epochs', type: 'int', def: 3,
    help: 'One epoch = one full pass over all 60,000 training images. More epochs give the optimizer more chances to improve — until the model starts memorizing instead of learning (watch train vs. validation loss diverge).',
  },
  {
    key: 'lr', label: 'Learning rate', type: 'float', def: 0.1,
    help: 'How big a step gradient descent takes each batch. Too high and the loss oscillates or explodes; too low and training crawls. 0.1 with momentum works well for this MLP.',
  },
  {
    key: 'batch_size', label: 'Batch size', type: 'int', def: 128,
    help: 'How many images are averaged into each gradient estimate. Bigger batches give smoother, slower-per-example updates; smaller ones are noisier but sometimes generalize better.',
  },
  {
    key: 'hidden', label: 'Hidden layers', type: 'list', def: '128, 64',
    help: 'Widths of the hidden layers, e.g. "128, 64" = two hidden layers. This is the same architecture choice as neural-viz, scaled up: each layer is z = W·x + b followed by ReLU.',
  },
]

export default function NewRunPanel({ onCreated }) {
  const [values, setValues] = useState(Object.fromEntries(FIELDS.map((f) => [f.key, String(f.def)])))
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [open, setOpen] = useState(null) // which help text is expanded

  const start = async () => {
    setBusy(true)
    setError(null)
    try {
      const run = await postJSON('/api/runs', {
        track: 'mnist_mlp',
        epochs: parseInt(values.epochs, 10),
        lr: parseFloat(values.lr),
        batch_size: parseInt(values.batch_size, 10),
        hidden: values.hidden.split(',').map((s) => parseInt(s.trim(), 10)).filter(Boolean),
      })
      onCreated(run)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="bg-slate-900 rounded-xl p-5 border border-slate-800">
      <h2 className="font-semibold mb-1">Train MNIST from scratch</h2>
      <p className="text-sm text-slate-400 mb-4">
        Track 1: a multilayer perceptron learning handwritten digits — the network
        neural-viz taught you, at real scale. 109k parameters, 60k images, on whatever
        device this backend has.
      </p>
      <div className="grid grid-cols-2 gap-3 mb-4">
        {FIELDS.map((f) => (
          <div key={f.key}>
            <button
              className="text-xs text-slate-400 hover:text-slate-200 flex items-center gap-1"
              onClick={() => setOpen(open === f.key ? null : f.key)}
              title="What does this do?"
            >
              {f.label} <span className="text-slate-600">ⓘ</span>
            </button>
            <input
              className="w-full bg-slate-800 rounded px-2 py-1 mt-1 text-sm border border-slate-700 focus:border-sky-600 outline-none"
              value={values[f.key]}
              onChange={(e) => setValues({ ...values, [f.key]: e.target.value })}
            />
            {open === f.key && (
              <p className="text-xs text-slate-400 mt-1 leading-relaxed">{f.help}</p>
            )}
          </div>
        ))}
      </div>
      {error && <p className="text-rose-400 text-sm mb-2">{error}</p>}
      <button
        className="bg-sky-700 hover:bg-sky-600 disabled:opacity-50 rounded px-4 py-1.5 text-sm font-medium"
        onClick={start}
        disabled={busy}
      >
        {busy ? 'Starting…' : 'Start training'}
      </button>
    </div>
  )
}
