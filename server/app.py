"""FastAPI WebSocket server for PSL recognition.

Protocol:
- Client connects to /ws/translate?mode=words (default) or ?mode=alphabets
- Each frame: send the JPEG bytes as a binary WebSocket message
- Server replies with a JSON text message describing pipeline state
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from sign_session import SignSession, _ensure_model_loaded
from alphabet_session import AlphabetSession, _ensure_alpha_model_loaded

log = logging.getLogger("psl.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eager-load both models so the first connection isn't slow.
    _ensure_model_loaded()
    try:
        _ensure_alpha_model_loaded()
    except Exception as exc:
        log.warning("alphabet model not loaded: %s", exc)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True}


@app.websocket("/ws/translate")
async def ws_translate(ws: WebSocket) -> None:
    await ws.accept()
    mode = (ws.query_params.get("mode") or "words").lower()
    if mode == "alphabets":
        session = AlphabetSession()
    else:
        mode = "words"
        session = SignSession()
    log.info("ws connected: %s mode=%s", ws.client, mode)
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            data = msg.get("bytes")
            if data is None:
                text = msg.get("text")
                if text == "ping":
                    await ws.send_text(json.dumps({"pong": True}))
                continue

            snapshot = await asyncio.to_thread(session.process_jpeg, data)
            snapshot["mode"] = mode
            await ws.send_text(json.dumps(snapshot))
    except WebSocketDisconnect:
        log.info("ws disconnected: %s", ws.client)
    except Exception as exc:
        log.exception("ws error: %s", exc)
        try:
            await ws.close()
        except Exception:
            pass
    finally:
        session.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="info")
