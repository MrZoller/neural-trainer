import {
  CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'

const AXIS = { stroke: '#475569', fontSize: 11 }
const GRID = { stroke: '#1e293b' }
const TOOLTIP = {
  contentStyle: { background: '#0f172a', border: '1px solid #334155', fontSize: 12 },
}

function Panel({ title, children }) {
  return (
    <div className="bg-slate-900 rounded-xl p-4 border border-slate-800">
      <h3 className="text-sm text-slate-300 mb-2">{title}</h3>
      <div className="h-52">{children}</div>
    </div>
  )
}

export default function RunCharts({ epochs, batch }) {
  return (
    <div className="grid md:grid-cols-2 gap-4">
      <Panel title="Loss per epoch — the overfitting story: watch for val rising while train falls">
        <ResponsiveContainer>
          <LineChart data={epochs}>
            <CartesianGrid {...GRID} />
            <XAxis dataKey="epoch" {...AXIS} />
            <YAxis {...AXIS} />
            <Tooltip {...TOOLTIP} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Line type="monotone" dataKey="train_loss" stroke="#38bdf8" dot={false} name="train" />
            <Line type="monotone" dataKey="val_loss" stroke="#f59e0b" dot={false} name="validation" />
          </LineChart>
        </ResponsiveContainer>
      </Panel>
      <Panel title="Accuracy per epoch">
        <ResponsiveContainer>
          <LineChart data={epochs}>
            <CartesianGrid {...GRID} />
            <XAxis dataKey="epoch" {...AXIS} />
            <YAxis {...AXIS} domain={[0.8, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
            <Tooltip {...TOOLTIP} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Line type="monotone" dataKey="train_acc" stroke="#38bdf8" dot={false} name="train" />
            <Line type="monotone" dataKey="val_acc" stroke="#f59e0b" dot={false} name="validation" />
          </LineChart>
        </ResponsiveContainer>
      </Panel>
      <Panel title="Batch loss (live) — the raw, noisy signal the epoch curves average out">
        <ResponsiveContainer>
          <LineChart data={batch}>
            <CartesianGrid {...GRID} />
            <XAxis dataKey="global_step" {...AXIS} />
            <YAxis {...AXIS} />
            <Tooltip {...TOOLTIP} />
            <Line type="monotone" dataKey="loss" stroke="#818cf8" dot={false} name="batch loss" />
          </LineChart>
        </ResponsiveContainer>
      </Panel>
    </div>
  )
}

export function LayerStatsTable({ layers }) {
  if (!layers?.length) return null
  return (
    <div className="bg-slate-900 rounded-xl p-4 border border-slate-800">
      <h3 className="text-sm text-slate-300 mb-2">
        Per-layer stats (last epoch) — gradient magnitude and dead-ReLU count, the
        numbers behind vanishing-gradient and dying-neuron stories
      </h3>
      <table className="w-full text-sm">
        <thead>
          <tr className="text-slate-500 text-left text-xs">
            <th className="py-1">layer</th>
            <th>shape [out, in]</th>
            <th>‖∂L/∂W‖ (mean)</th>
            <th>‖W‖</th>
            <th>dead ReLUs</th>
          </tr>
        </thead>
        <tbody>
          {layers.map((l) => (
            <tr key={l.name} className="border-t border-slate-800">
              <td className="py-1 font-mono text-xs">{l.name}</td>
              <td className="font-mono text-xs">{JSON.stringify(l.shape)}</td>
              <td>{l.grad_norm ?? '—'}</td>
              <td>{l.weight_norm}</td>
              <td>{l.dead_relu_pct != null ? `${l.dead_relu_pct}%` : '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
