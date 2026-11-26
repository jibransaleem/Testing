"""
resilient_conn.py -- a psycopg connection that heals itself.

Neon / hosted Postgres closes idle connections. This wrapper:
  * reconnects if the connection is closed/broken,
  * pings (SELECT 1) after being idle for a while and reconnects if the ping fails,
  * retries a single execute() once after a reconnect (only when NOT inside a transaction).

conn.transaction() is wrapped so the wrapper KNOWS when a transaction block is open. Inside one it never
reconnects or replays: a dropped connection raises, the block rolls back, and put_item reports WRITE_FAILED.

Usage:
    conn = ResilientConnection(get_connection)     # pass the FACTORY, not a connection
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager

import psycopg
from psycopg.pq import TransactionStatus

IDLE_PING_AFTER = 20.0

_IN_TXN = {TransactionStatus.INTRANS, TransactionStatus.INERROR, TransactionStatus.ACTIVE}
_CONN_ERRORS = (psycopg.OperationalError, psycopg.InterfaceError)


class ResilientConnection:
    def __init__(self, factory, idle_ping_after: float = IDLE_PING_AFTER):
        self._factory = factory
        self._idle_ping_after = idle_ping_after
        self._lock = threading.RLock()
        self._txn_depth = 0
        self._conn = factory()
        self._last_used = time.monotonic()

    def _dead(self) -> bool:
        c = self._conn
        return c is None or getattr(c, "closed", False) or getattr(c, "broken", False)

    def _in_txn(self) -> bool:
        if self._txn_depth > 0:
            return True
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
            if self._txn_depth > 0:
                return
            if self._dead():
                self.reconnect()
                return
            if self._in_txn():
                return
            if time.monotonic() - self._last_used > self._idle_ping_after:
                try:
                    self._conn.execute("SELECT 1")
                except _CONN_ERRORS:
                    self.reconnect()
            self._last_used = time.monotonic()

    def execute(self, *args, **kwargs):
        with self._lock:
            self._ensure()
            try:
                result = self._conn.execute(*args, **kwargs)
                self._last_used = time.monotonic()
                return result
            except _CONN_ERRORS:
                if self._in_txn():
                    raise
                self.reconnect()
                return self._conn.execute(*args, **kwargs)

    @contextmanager
    def transaction(self, *args, **kwargs):
        with self._lock:
            self._ensure()
            self._txn_depth += 1
            try:
                with self._conn.transaction(*args, **kwargs) as tx:
                    yield tx
            finally:
                self._txn_depth -= 1
                self._last_used = time.monotonic()

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        self._ensure()
        return getattr(self._conn, name)