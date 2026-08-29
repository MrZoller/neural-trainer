import { useCallback, useEffect, useRef, useState } from 'react'
import { getJSON, postJSON } from '../api.js'

const CHUNK = 40 // files per upload request

// Deterministic pastel per capture group so bursts are visually obvious.
function groupColor(group) {
  let h = 0
  for (const c of group) h = (h * 31 + c.charCodeAt(0)) % 360
  return `hsl(${h} 60% 45%)`
}

export default function DatasetView({ datasetId, onNavigate }) {
  const [dataset, setDataset] = useState(null)
  const [images, setImages] = useState([])
  const [report, setReport] = useState(null)
  const [selectedLabel, setSelectedLabel] = useState(null)
  const [newLabel, setNewLabel] = useState('')
  const [progress, setProgress] = useState(null)
  const [trainCfg, setTrainCfg] = useState({ epochs: '10', lr: '0.001' })
  const [trainError, setTrainError] = useState(null)
  const folderRef = useRef(null)
  const filesRef = useRef(null)

  const refresh = useCallback(() => {
    getJSON(`/api/datasets/${datasetId}`)
      .then(setDataset)
      .catch(() => {})
    getJSON(`/api/datasets/${datasetId}/images`)
      .then(setImages)
      .catch(() => {})
    getJSON(`/api/datasets/${datasetId}/report`)
      .then(setReport)
      .catch(() => {})
  }, [datasetId])

  useEffect(() => {
    refresh()
  }, [refresh])

  const upload = async (fileList) => {
    const files = [...fileList]
    let done = 0
    setProgress({ done, total: files.length })
    for (let i = 0; i < files.length; i += CHUNK) {
      const form = new FormData()
      for (const f of files.slice(i, i + CHUNK)) {
        form.append('files', f, f.webkitRelativePath || f.name)
      }
      await fetch(`/api/datasets/${datasetId}/images`, { method: 'POST', body: form })
      done += Math.min(CHUNK, files.length - i)
      setProgress({ done, total: files.length })
    }
    setProgress(null)
    refresh()
  }

  const labelImage = async (img) => {
    if (!selectedLabel) return
    await fetch(`/api/images/${img.id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label: selectedLabel }),
    })
    refresh()
  }

  const toggleExclude = async (img, e) => {
    e.stopPropagation()
    await fetch(`/api/images/${img.id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ excluded: !img.excluded }),
    })
    refresh()
  }

  const startTraining = async () => {
    setTrainError(null)
    try {
      const run = await postJSON('/api/runs', {
        track: 'custom_finetune',
        dataset_id: datasetId,
        epochs: parseInt(trainCfg.epochs, 10),
        lr: parseFloat(trainCfg.lr),
      })
      onNavigate({ name: 'run', id: run.id })
    } catch (e) {
      setTrainError(e.message)
    }
  }

  const classes = report
    ? [...new Set([...Object.keys(report.classes), ...(selectedLabel ? [selectedLabel] : [])])]
    : selectedLabel
      ? [selectedLabel]
      : []

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3 flex-wrap">
        <button
          className="text-slate-400 hover:text-slate-200 text-sm"
          onClick={() => onNavigate({ name: 'home' })}
        >
          ← home
        </button>
        <h2 className="font-semibold">{dataset?.name}</h2>
        <span className="text-xs text-slate-500">
          {report && `${report.n_labeled} labeled / ${report.n_images} images`}
        </span>
        <span className="flex-1" />
        <button
          className="bg-slate-800 hover:bg-slate-700 rounded px-3 py-1 text-sm"
          onClick={() => folderRef.current.click()}
        >
          Import folder
        </button>
        <button
          className="bg-slate-800 hover:bg-slate-700 rounded px-3 py-1 text-sm"
          onClick={() => filesRef.current.click()}
        >
          Add images
        </button>
        <input
          ref={folderRef}
          type="file"
          webkitdirectory=""
          multiple
          className="hidden"
          onChange={(e) => upload(e.target.files)}
        />
        <input
          ref={filesRef}
          type="file"
          accept="image/*"
          multiple
          className="hidden"
          onChange={(e) => upload(e.target.files)}
        />
      </div>

      {progress && (
        <p className="text-sm text-sky-400">
          Uploading {progress.done}/{progress.total}…
        </p>
      )}

      {report && (
        <div className="bg-slate-900 rounded-xl p-4 border border-slate-800">
          <h3 className="text-sm text-slate-300 mb-2">Dataset report</h3>
          <div className="flex gap-6 flex-wrap text-sm">
            {Object.entries(report.classes).map(([label, c]) => (
              <div key={label}>
                <span className="text-slate-200 font-medium">{label}</span>{' '}
                <span className="text-slate-500 text-xs">
                  {c.images} images · {c.groups} session{c.groups !== 1 ? 's' : ''} · split{' '}
                  {c.train}/{c.val}
                </span>
              </div>
            ))}
          </div>
          {report.warnings.length > 0 && (
            <ul className="mt-2 space-y-1">
              {report.warnings.map((w) => (
                <li key={w} className="text-amber-400/90 text-xs">
                  ⚠ {w}
                </li>
              ))}
            </ul>
          )}
          {report.n_unlabeled > 0 && (
            <p className="text-xs text-slate-500 mt-2">
              {report.n_unlabeled} unlabeled images will be skipped at training time.
            </p>
          )}
        </div>
      )}

      <div className="bg-slate-900 rounded-xl p-4 border border-slate-800">
        <div className="flex items-center gap-2 flex-wrap mb-3">
          <span className="text-sm text-slate-400">Label brush:</span>
          {classes.map((c) => (
            <button
              key={c}
              className={`px-2.5 py-0.5 rounded-full text-xs ${
                selectedLabel === c
                  ? 'bg-sky-700 text-white'
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
              onClick={() => setSelectedLabel(selectedLabel === c ? null : c)}
            >
              {c}
            </button>
          ))}
          <input
            className="bg-slate-800 rounded px-2 py-0.5 text-xs border border-slate-700 w-28"
            placeholder="new class…"
            value={newLabel}
            onChange={(e) => setNewLabel(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && newLabel.trim()) {
                setSelectedLabel(newLabel.trim())
                setNewLabel('')
              }
            }}
          />
          {selectedLabel && (
            <span className="text-xs text-slate-500">
              click images to label them “{selectedLabel}”
            </span>
          )}
        </div>
        {images.length === 0 ? (
          <p className="text-slate-500 text-sm">
            No images yet. “Import folder” auto-labels from folder names (cat/, dog/, …); colored
            dots mark capture sessions — photos taken seconds apart stay on the same side of the
            train/val split.
          </p>
        ) : (
          <div className="grid grid-cols-[repeat(auto-fill,minmax(96px,1fr))] gap-2">
            {images.map((img) => (
              <div
                key={img.id}
                className={`relative rounded-lg overflow-hidden cursor-pointer group
                     ${img.excluded ? 'opacity-30' : ''}
                     ${!img.label && !img.excluded ? 'ring-2 ring-amber-600/70' : ''}`}
                onClick={() => labelImage(img)}
                title={`${img.filename}\nsession: ${img.capture_group}`}
              >
                <img
                  src={`/api/images/${img.id}/file`}
                  alt={img.filename}
                  loading="lazy"
                  className="w-full h-24 object-cover"
                />
                <span
                  className="absolute top-1 left-1 w-2.5 h-2.5 rounded-full"
                  style={{ background: groupColor(img.capture_group) }}
                />
                {img.label && (
                  <span className="absolute bottom-0 inset-x-0 bg-black/70 text-[10px] px-1 py-0.5 truncate">
                    {img.label}
                  </span>
                )}
                <button
                  className="absolute top-0.5 right-0.5 hidden group-hover:block bg-black/70 rounded px-1 text-xs"
                  onClick={(e) => toggleExclude(img, e)}
                  title={img.excluded ? 'restore' : 'exclude from training'}
                >
                  {img.excluded ? '↩' : '✕'}
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="bg-slate-900 rounded-xl p-4 border border-slate-800">
        <h3 className="text-sm text-slate-300 mb-1">Train on this dataset</h3>
        <p className="text-xs text-slate-500 mb-3 max-w-2xl">
          Fine-tunes a MobileNetV3 pretrained on ImageNet: the backbone (927k weights that already
          know edges, textures, shapes) stays frozen — only a small classifier head learns your
          classes. That&apos;s why ~30–100 photos per class is enough. Starting a run freezes a
          snapshot of the current labels and split, so this run stays reproducible even as the
          dataset grows.
        </p>
        <div className="flex items-end gap-3 flex-wrap">
          <label className="text-xs text-slate-400">
            epochs
            <input
              className="block bg-slate-800 rounded px-2 py-1 mt-1 text-sm border border-slate-700 w-20"
              value={trainCfg.epochs}
              onChange={(e) => setTrainCfg({ ...trainCfg, epochs: e.target.value })}
            />
          </label>
          <label className="text-xs text-slate-400">
            learning rate
            <input
              className="block bg-slate-800 rounded px-2 py-1 mt-1 text-sm border border-slate-700 w-24"
              value={trainCfg.lr}
              onChange={(e) => setTrainCfg({ ...trainCfg, lr: e.target.value })}
            />
          </label>
          <button
            className="bg-sky-700 hover:bg-sky-600 rounded px-4 py-1.5 text-sm font-medium"
            onClick={startTraining}
          >
            Freeze dataset &amp; train
          </button>
        </div>
        {trainError && <p className="text-rose-400 text-sm mt-2">{trainError}</p>}
      </div>
    </div>
  )
}
