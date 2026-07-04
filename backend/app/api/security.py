"""Remote-mode auth (DESIGN.md §3 Security posture).

No NT_TOKEN set (the localhost default) → no auth. With NT_TOKEN: REST wants
`Authorization: Bearer <token>`; WebSockets use a short-lived single-use
ticket fetched over authenticated REST — tokens never ride in WS query
strings, where they'd leak into logs and browser history.
"""

import secrets
import threading
import time

from fastapi import HTTPException, Request

from app import config

_tickets: dict[str, float] = {}
_tickets_lock = threading.Lock()
TICKET_TTL_SECONDS = 30


def require_auth(request: Request) -> None:
    if not config.TOKEN:
        return
    header = request.headers.get("authorization", "")
    if not secrets.compare_digest(header, f"Bearer {config.TOKEN}"):
        raise HTTPException(status_code=401, detail="missing or invalid bearer token")


def issue_ticket() -> str:
    ticket = secrets.token_urlsafe(24)
    with _tickets_lock:
        now = time.time()
        for t, exp in list(_tickets.items()):
            if exp < now:
                del _tickets[t]
        _tickets[ticket] = now + TICKET_TTL_SECONDS
    return ticket


def redeem_ticket(ticket: str | None) -> bool:
    if not config.TOKEN:
        return True
    if not ticket:
        return False
    with _tickets_lock:
        return _tickets.pop(ticket, 0) > time.time()
