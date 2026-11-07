"""
resilient_conn.py -- a psycopg connection that heals itself.

Neon / hosted Postgres closes idle connections. A plain psycopg connection then fails with
"OperationalError: the connection is closed". This wrapper:
  * reconnects if the connection is closed/broken,
  * pings (SELECT 1) after being idle for a while and reconnects if the ping fails,
  * retries a single execute() once after a reconnect (only when NOT inside a transaction).

Everything else (cursor(), transaction(), info, ...) is forwarded to the real connection,
so existing code like `conn.execute(...)`, `with conn.cursor()`, `with conn.transaction()` is unchanged.

Usage:
    conn = ResilientConnection(get_connection)     # pass the FACTORY, not a connection
"""
from __future__ import annotations

import threading
import time

import psycopg
from psycopg.pq import TransactionStatus

IDLE_PING_AFTER = 20.0          # seconds of inactivity before we verify the connection

_IN_TXN = {TransactionStatus.INTRANS, TransactionStatus.INERROR, TransactionStatus.ACTIVE}


class ResilientConnection:
    def __init__(self, factory, idle_ping_after: float = IDLE_PING_AFTER):
        self._factory = factory
        self._idle_ping_after = idle_ping_after
        self._lock = threading.RLock()
        self._conn = factory()
        self._last_used = time.monotonic()

    # ------------------------------------------------------------------ internals
    def _dead(self) -> bool:
        c = self._conn
        return c is None or getattr(c, "closed", False) or getattr(c, "broken", False)

    def _in_txn(self) -> bool:
        try:
            return self._conn.info.transaction_status in _IN_TXN
        except Exception:                                   # noqa: BLE001
            return False

    def reconnect(self) -> None:
        with self._lock:
            try:
                if self._conn is not None:
                    self._conn.close()
            except Exception:                               # noqa: BLE001
                pass
            self._conn = self._factory()
            self._last_used = time.monotonic()

    def _ensure(self) -> None:
        with self._lock:
            if self._dead():
                self.reconnect()
                return
            if self._in_txn():                              # never touch a connection mid-transaction
                return
            if time.monotonic() - self._last_used > self._idle_ping_after:
                try:
                    self._conn.execute("SELECT 1")
                except psycopg.OperationalError:
                    self.reconnect()
            self._last_used = time.monotonic()

    # ------------------------------------------------------------------ public API
    def execute(self, *args, **kwargs):
        self._ensure()
        try:
            return self._conn.execute(*args, **kwargs)
        except psycopg.OperationalError:
            if self._in_txn():                              # a half-done transaction must not be replayed
                raise
            self.reconnect()
            return self._conn.execute(*args, **kwargs)

    def __getattr__(self, name):
        if name.startswith("_"):                            # avoid recursion before __init__ finishes
            raise AttributeError(name)
        self._ensure()
        return getattr(self._conn, name)