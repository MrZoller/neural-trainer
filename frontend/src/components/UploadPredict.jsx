import { useRef, useState } from 'react'
import { postJSON } from '../api.js'

// Track 2 inference: drop/choose a photo, see the 224×224 center crop the
// model actually saw plus per-class confidence.
export default function UploadPredict({ runId }) {
  const inputRef = useRef(null)
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  const predictFile = (file) => {
    if (!file) return
    setBusy(true)
    setError(null)
    const reader = new FileReader()
    reader.onload = async () => {
      try {
        setResult(await postJSON(`/api/runs/${runId}/predict`, { image: reader.result }))
      } catch (e) {
        setError(e.message)
      } finally {
        setBusy(false)
      }
    }
    reader.readAsDataURL(file)
  }

  return (
    <div className="bg-slate-900 rounded-xl p-4 border border-slate-800">
      <h3 className="text-sm text-slate-300 mb-2">Use your model — try a photo</h3>
      <div className="flex gap-6 flex-wrap items-start">
        <div
          className="w-64 h-40 border-2 border-dashed border-slate-700 rounded-lg flex items-center justify-center text-slate-500 text-sm cursor-pointer hover:border-sky-700"
          onClick={() => inputRef.current.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault()
            predictFile(e.dataTransfer.files[0])
          }}
        >
          {busy ? 'Predicting…' : 'Drop an image or click to choose'}
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => predictFile(e.target.files[0])}
          />
        </div>
        {error && <p className="text-rose-400 text-sm">{error}</p>}
        {result && (
          <div className="flex gap-6">
            <div>
              <p className="text-xs text-slate-500 mb-1">what the model sees (224×224 crop)</p>
              <img
                src={result.preview_url}
                alt="model input"
                className="w-36 h-36 rounded object-cover"
              />
            </div>
            <div>
              <p className="text-xs text-slate-500 mb-1">
                prediction:{' '}
                <span className="text-xl text-slate-100 font-bold">{result.prediction}</span>
              </p>
              {result.class_mapping.map((cls, i) => (
                <div key={cls} className="flex items-center gap-2 text-xs mb-0.5">
                  <span className="w-20 text-slate-400 truncate text-right">{cls}</span>
                  <div className="w-40 bg-slate-800 rounded h-2.5">
                    <div
                      className={`h-2.5 rounded ${cls === result.prediction ? 'bg-emerald-500' : 'bg-slate-600'}`}
                      style={{ width: `${Math.max(1, result.probs[i] * 100)}%` }}
                    />
                  </div>
                  <span className="text-slate-500 w-12">{(result.probs[i] * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
