import { useEffect, useRef, useState } from 'react'
import { postJSON } from '../api.js'

const SIZE = 280

// Draw white-on-black like MNIST. The backend still runs the full
// crop/center/normalize pipeline — a canvas drawing is NOT automatically
// MNIST-distributed, and the preview shows exactly what survives it
// (DESIGN.md §9: the app's first distribution-shift lesson).
export default function DrawCanvas({ runId }) {
  const canvasRef = useRef(null)
  const drawing = useRef(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const previewRef = useRef(null)

  useEffect(() => {
    const ctx = canvasRef.current.getContext('2d')
    ctx.fillStyle = 'black'
    ctx.fillRect(0, 0, SIZE, SIZE)
  }, [])

  useEffect(() => {
    if (!result || !previewRef.current) return
    const ctx = previewRef.current.getContext('2d')
    const img = ctx.createImageData(28, 28)
    result.preprocessed.flat().forEach((v, i) => {
      img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = v
      img.data[i * 4 + 3] = 255
    })
    ctx.putImageData(img, 0, 0)
  }, [result])

  const pos = (e) => {
    const rect = canvasRef.current.getBoundingClientRect()
    return [e.clientX - rect.left, e.clientY - rect.top]
  }

  const start = (e) => {
    drawing.current = true
    const ctx = canvasRef.current.getContext('2d')
    ctx.strokeStyle = 'white'
    ctx.lineWidth = 18
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.beginPath()
    ctx.moveTo(...pos(e))
  }

  const move = (e) => {
    if (!drawing.current) return
    const ctx = canvasRef.current.getContext('2d')
    ctx.lineTo(...pos(e))
    ctx.stroke()
  }

  const clear = () => {
    const ctx = canvasRef.current.getContext('2d')
    ctx.fillStyle = 'black'
    ctx.fillRect(0, 0, SIZE, SIZE)
    setResult(null)
    setError(null)
  }

  const predict = async () => {
    setError(null)
    try {
      setResult(await postJSON(`/api/runs/${runId}/predict`, {
        image: canvasRef.current.toDataURL('image/png'),
      }))
    } catch (e) {
      setError(e.message)
    }
  }

  return (
    <div className="bg-slate-900 rounded-xl p-4 border border-slate-800">
      <h3 className="text-sm text-slate-300 mb-2">Use your model — draw a digit</h3>
      <div className="flex gap-6 flex-wrap">
        <div>
          <canvas
            ref={canvasRef}
            width={SIZE}
            height={SIZE}
            className="rounded-lg cursor-crosshair touch-none"
            onPointerDown={start}
            onPointerMove={move}
            onPointerUp={() => (drawing.current = false)}
            onPointerLeave={() => (drawing.current = false)}
          />
          <div className="flex gap-2 mt-2">
            <button className="bg-sky-700 hover:bg-sky-600 rounded px-3 py-1 text-sm" onClick={predict}>
              Predict
            </button>
            <button className="bg-slate-800 hover:bg-slate-700 rounded px-3 py-1 text-sm" onClick={clear}>
              Clear
            </button>
          </div>
          {error && <p className="text-rose-400 text-sm mt-2">{error}</p>}
        </div>
        {result && (
          <div className="flex gap-6">
            <div>
              <p className="text-xs text-slate-500 mb-1">what the model sees (28×28)</p>
              <canvas
                ref={previewRef}
                width={28}
                height={28}
                className="rounded"
                style={{ width: 112, height: 112, imageRendering: 'pixelated' }}
              />
            </div>
            <div>
              <p className="text-xs text-slate-500 mb-1">
                prediction: <span className="text-2xl text-slate-100 font-bold">{result.prediction}</span>
              </p>
              {result.probs.map((p, digit) => (
                <div key={digit} className="flex items-center gap-2 text-xs">
                  <span className="w-3 text-slate-400 font-mono">{digit}</span>
                  <div className="w-40 bg-slate-800 rounded h-2.5">
                    <div
                      className={`h-2.5 rounded ${digit === result.prediction ? 'bg-emerald-500' : 'bg-slate-600'}`}
                      style={{ width: `${Math.max(1, p * 100)}%` }}
                    />
                  </div>
                  <span className="text-slate-500 w-12">{(p * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
