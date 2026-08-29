const STYLES = {
  queued: 'bg-slate-800 text-slate-300',
  running: 'bg-sky-900/60 text-sky-300',
  stopping: 'bg-amber-900/60 text-amber-300',
  stopped: 'bg-amber-900/60 text-amber-300',
  done: 'bg-emerald-900/60 text-emerald-300',
  failed: 'bg-rose-900/60 text-rose-300',
  killed: 'bg-rose-900/60 text-rose-300',
  cancelled: 'bg-slate-800 text-slate-400',
  interrupted: 'bg-purple-900/60 text-purple-300',
}

export default function StatusBadge({ status }) {
  return (
    <span
      className={`px-2 py-0.5 rounded-full text-xs font-medium ${STYLES[status] || STYLES.queued}`}
    >
      {status}
    </span>
  )
}
