from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

try:
    from langgraph.errors import GraphInterrupt
except ImportError:  # pragma: no cover
    GraphInterrupt = None  # type: ignore[assignment,misc]

DEFAULT_DB_PATH = Path("data/runtime/observability.db")


class ObservabilityStore:
    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    report_id TEXT NOT NULL,
                    agent TEXT NOT NULL,
                    status TEXT NOT NULL,
                    duration_ms REAL NOT NULL,
                    details TEXT
                )
                """
            )

    def _insert(
        self,
        report_id: str,
        agent: str,
        status: str,
        duration_ms: float,
        details: str | None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO agent_events
                    (timestamp, report_id, agent, status, duration_ms, details)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    report_id,
                    agent,
                    status,
                    duration_ms,
                    details,
                ),
            )

    @contextmanager
    def trace(self, report_id: str, agent: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        except Exception as exc:
            # langgraph's interrupt() pauses execution by raising GraphInterrupt —
            # that is not a node failure, so record it distinctly from real errors.
            duration_ms = (time.perf_counter() - start) * 1000
            if GraphInterrupt is not None and isinstance(exc, GraphInterrupt):
                self._insert(report_id, agent, "interrupted", duration_ms, None)
            else:
                self._insert(report_id, agent, "error", duration_ms, str(exc))
            raise
        else:
            duration_ms = (time.perf_counter() - start) * 1000
            self._insert(report_id, agent, "success", duration_ms, None)

    def recent_events(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM agent_events ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]


observer = ObservabilityStore()
