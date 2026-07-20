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
import os
import socket
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from sign_session import SignSession, _ensure_model_loaded
from alphabet_session import AlphabetSession, _ensure_alpha_model_loaded
from suggestions import word_completions, sentence_completions

# Hosts like Railway/Render/Cloud Run inject PORT and route their public
# domain to it; local dev keeps the 8000 default from SETUP.md.
PORT = int(os.getenv("PORT", "8000"))

log = logging.getLogger("psl.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _lan_ip() -> str | None:
    """The Mac's actual LAN IPv4 (the one packets to the internet leave from).

    UDP "connect" to a public IP picks the right outbound interface without
    sending a packet. Avoids relying on getaddrinfo(gethostname()), which
    on macOS often returns stale or squatter-cached entries.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 53))
            ip = s.getsockname()[0]
            return ip if not ip.startswith("127.") else None
    except Exception:
        return None


def _print_banner() -> None:
    ip = _lan_ip()
    bar = "─" * 64
    print(f"\n{bar}")
    print("  Listen inference server is up.")
    print("  Set this in the app  Profile → Server URL :")
    if ip:
        print(f"    ws://{ip}:{PORT}/ws/translate")
    else:
        print(f"    ws://<your-mac-LAN-IP>:{PORT}/ws/translate")
    print("  Phone must be on the same Wi-Fi network as this Mac.")
    print(f"{bar}\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eager-load both models so the first connection isn't slow.
    _ensure_model_loaded()
    try:
        _ensure_alpha_model_loaded()
    except Exception as exc:
        log.warning("alphabet model not loaded: %s", exc)
    _print_banner()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True}


@app.get("/suggest/words")
async def suggest_words(prefix: str = "", limit: int = 6) -> dict:
    return {"prefix": prefix, "suggestions": word_completions(prefix, limit)}


@app.get("/suggest/sentences")
async def suggest_sentences(prefix: str = "", limit: int = 6) -> dict:
    return {"prefix": prefix, "suggestions": sentence_completions(prefix, limit)}


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
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, log_level="info")
