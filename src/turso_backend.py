"""
A thin sqlite3.Connection-shaped wrapper around libsql, so the rest of this
codebase — written against sqlite3.Connection and sqlite3.Row, using
`row["column_name"]` throughout — works unchanged against a remote Turso
database. Used only when TURSO_DATABASE_URL is set; local dev and the test
suite never import this module.

libsql's Python client is otherwise close to sqlite3's API (execute,
executescript, commit, close), but two gaps aren't drop-in, both confirmed
against a real Turso database rather than assumed from documentation:

  1. Rows come back as plain tuples, not sqlite3.Row — no `row["col"]`
     access, which this codebase relies on everywhere.
  2. There's no `.row_factory` attribute at all (AttributeError on assign).

Both are closed here. `_Row.__getitem__` raises IndexError (not ValueError
or KeyError) for a missing column, matching sqlite3.Row's actual behavior —
tracker.py's `_row_get()` specifically catches IndexError, so this has to
match exactly, not just "raise something."

Also: libsql's client is an "embedded replica" — a local SQLite file that
syncs with the remote database, not a stateless HTTP-per-query client. That
local file lives on Render's ephemeral disk and does NOT need to survive a
restart: `connect()` does an initial sync from Turso on startup, and
`commit()` here calls `.sync()` immediately after, so Turso — not the local
file — is what actually makes a write durable.
"""
import libsql


class _Row:
    """sqlite3.Row-alike: supports row["col"], row[0], and iteration."""
    __slots__ = ("_values", "_columns")

    def __init__(self, values: tuple, columns: list):
        self._values = values
        self._columns = columns

    def __getitem__(self, key):
        if isinstance(key, str):
            try:
                idx = self._columns.index(key)
            except ValueError:
                raise IndexError(key)  # match sqlite3.Row's exception type
            return self._values[idx]
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def keys(self):
        return list(self._columns)

    def __repr__(self):
        return f"<TursoRow {dict(zip(self._columns, self._values))}>"


class _Cursor:
    """Wraps a libsql cursor/result so fetch methods yield _Row objects
    instead of plain tuples."""

    def __init__(self, raw):
        self._raw = raw
        cols = getattr(raw, "description", None)
        self._columns = [d[0] for d in cols] if cols else []
        self.lastrowid = getattr(raw, "lastrowid", None)

    def _wrap(self, row):
        return _Row(tuple(row), self._columns) if row is not None else None

    def fetchall(self):
        return [self._wrap(r) for r in self._raw.fetchall()]

    def fetchone(self):
        return self._wrap(self._raw.fetchone())

    def __iter__(self):
        # libsql's raw cursor isn't itself iterable (confirmed against a
        # real query: `for row in raw_cursor` raises TypeError) -- only
        # .fetchall()/.fetchone() are supported, unlike sqlite3.Cursor.
        # tracker.py does `for row in conn.execute(...)` directly, so this
        # has to work here even though the thing underneath doesn't.
        return iter(self.fetchall())


class TursoConnection:
    """Facade matching the sqlite3.Connection surface this codebase
    actually calls: execute, executescript, commit, close, and a settable
    (if inert) row_factory attribute -- see module docstring for why each
    of these needed wrapping rather than using libsql's connection as-is.
    """

    def __init__(self, database_url: str, auth_token: str, local_replica: str = "/tmp/watchdog_replica.db"):
        self._conn = libsql.connect(local_replica, sync_url=database_url, auth_token=auth_token)
        self._conn.sync()
        self.row_factory = None  # accepted for API parity with sqlite3.Connection; ignored

    def execute(self, sql: str, params: tuple = ()):
        raw = self._conn.execute(sql, params)
        return _Cursor(raw)

    def executescript(self, script: str) -> None:
        self._conn.executescript(script)

    def commit(self) -> None:
        self._conn.commit()
        self._conn.sync()

    def close(self) -> None:
        self._conn.close()
