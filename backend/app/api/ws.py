"""WebSocket event stream with last_seq gap replay (DESIGN.md §4).

Subscribe-then-replay ordering guarantees no gap: live events that arrive
during replay are buffered in the subscriber queue and deduplicated by seq.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.api.security import redeem_ticket

router = APIRouter()


@router.websocket("/ws/runs/{run_id}")
async def run_events(ws: WebSocket, run_id: str, last_seq: int = 0,
                     ticket: str | None = None):
    if not redeem_ticket(ticket):
        await ws.close(code=4401)
        return
    db = ws.app.state.db
    manager = ws.app.state.manager
    if not db.get_run(run_id):
        await ws.close(code=4404)
        return

    await ws.accept()
    sub = manager.subscribe(run_id)
    sent = last_seq
    try:
        for event in db.events_after(run_id, last_seq):
            await ws.send_json(event)
            sent = event["seq"]
        while True:
            event = await sub.get()
            if event["seq"] <= sent:
                continue
            await ws.send_json(event)
            sent = event["seq"]
    except WebSocketDisconnect:
        pass
    finally:
        manager.unsubscribe(run_id, sub)
