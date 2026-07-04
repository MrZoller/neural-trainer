export async function getJSON(path) {
  const r = await fetch(path)
  if (!r.ok) throw new Error((await r.json()).detail || r.statusText)
  return r.json()
}

export async function postJSON(path, body) {
  const r = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error((await r.json()).detail || r.statusText)
  return r.json()
}

// Event stream with automatic reconnect. On reconnect the server replays
// everything after the last seq we saw — the UI never misses an event
// across refreshes or network blips (DESIGN.md §4).
export function openRunSocket(runId, onEvent) {
  let ws = null
  let closed = false
  let lastSeq = 0
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'

  const connect = () => {
    ws = new WebSocket(`${proto}://${location.host}/ws/runs/${runId}?last_seq=${lastSeq}`)
    ws.onmessage = (m) => {
      const event = JSON.parse(m.data)
      lastSeq = event.seq
      onEvent(event)
    }
    ws.onclose = () => {
      if (!closed) setTimeout(connect, 1000)
    }
  }
  connect()
  return () => {
    closed = true
    ws?.close()
  }
}
