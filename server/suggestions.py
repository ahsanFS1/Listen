"""Urdu word + sentence completion via Neon Postgres.

Mirrors the predict_words / suggest_phrases queries in
src/inference/psl-v1.py on the `main` branch. Exposed as cheap REST
endpoints (see app.py) so the mobile client can fetch suggestions on
demand without holding any DB state in the WebSocket session.

Connection is lazy and silently disabled if the DB is unreachable —
the inference pipeline must keep working even when suggestions don't.
"""

from __future__ import annotations

import os
import threading
from typing import List, Optional

try:
    import psycopg2
except Exception:
    psycopg2 = None


_DB_HOST = os.getenv("DB_HOST", "ep-falling-river-aql8vty6.c-8.us-east-1.aws.neon.tech")
_DB_PORT = os.getenv("DB_PORT", "5432")
_DB_NAME = os.getenv("DB_NAME", "neondb")
_DB_USER = os.getenv("DB_USER", "neondb_owner")
_DB_PASSWORD = os.getenv("DB_PASSWORD", "npg_Kprkc1Po3ZHA")
_DB_SSLMODE = os.getenv("PGSSLMODE", "require")

_lock = threading.Lock()
_conn = None


def _ensure_conn():
    """Return a healthy connection, reconnecting on demand.

    Networks change (Wi-Fi swap, IP renewal), so a previously-good Neon
    socket can silently die. We don't permanently disable on first error —
    instead we drop the dead handle and try again on the next call.
    """
    global _conn
    if psycopg2 is None:
        return None
    if _conn is not None:
        # Cheap liveness probe — closes/marks bad if the socket is gone.
        try:
            if getattr(_conn, "closed", 0):
                _conn = None
            else:
                with _conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
        except Exception:
            try:
                _conn.close()
            except Exception:
                pass
            _conn = None
    if _conn is not None:
        return _conn
    with _lock:
        if _conn is not None:
            return _conn
        try:
            _conn = psycopg2.connect(
                host=_DB_HOST, port=_DB_PORT, dbname=_DB_NAME,
                user=_DB_USER, password=_DB_PASSWORD, sslmode=_DB_SSLMODE,
                connect_timeout=4,
            )
            _conn.autocommit = True
            print(f"[suggest] connected to {_DB_HOST}/{_DB_NAME}")
        except Exception as exc:
            print(f"[suggest] DB connect failed: {exc}")
            _conn = None
    return _conn


def _query(sql: str, params: tuple, limit: int) -> List[str]:
    conn = _ensure_conn()
    if conn is None:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return [r[0] for r in cur.fetchall()][:limit]
    except Exception as exc:
        print(f"[suggest] query failed: {exc}")
        # Drop the broken handle so the next call reconnects from scratch.
        global _conn
        try:
            conn.close()
        except Exception:
            pass
        _conn = None
        return []


def word_completions(prefix: str, limit: int = 6) -> List[str]:
    p = (prefix or "").strip()
    if not p:
        return []
    # Prefix match first (true completion of the live spelling), then fall
    # back to substring so the user still sees something when no word in
    # the dictionary literally begins with their letters.
    rows = _query(
        "SELECT word FROM urdu_words WHERE word LIKE %s LIMIT %s",
        (f"{p}%", limit),
        limit,
    )
    if rows:
        return rows
    return _query(
        "SELECT word FROM urdu_words WHERE word LIKE %s LIMIT %s",
        (f"%{p}%", limit),
        limit,
    )


def sentence_completions(prefix: str, limit: int = 6) -> List[str]:
    p = (prefix or "").strip()
    if not p:
        return _query(
            "SELECT sentence FROM urdu_sentences LIMIT %s",
            (limit,),
            limit,
        )
    # Try a prefix match first (sentences that *start* with the signed
    # text), then any sentence that *contains* it. Finally fall back to
    # generic sentences so the user always sees suggestions.
    rows = _query(
        "SELECT sentence FROM urdu_sentences "
        "WHERE sentence LIKE %s OR sentence LIKE %s LIMIT %s",
        (f"{p}%", f"{p} %", limit),
        limit,
    )
    if rows:
        return rows
    rows = _query(
        "SELECT sentence FROM urdu_sentences WHERE sentence LIKE %s LIMIT %s",
        (f"%{p}%", limit),
        limit,
    )
    if rows:
        return rows
    return _query(
        "SELECT sentence FROM urdu_sentences LIMIT %s",
        (limit,),
        limit,
    )
