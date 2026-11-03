# Daily Log

## 2026-11-03
In `src/aggregator.py`, sliding window calculations were updated to prevent memory accumulation during long-running daemon execution.

## 2026-11-03
The SQLite backend in `src/db.py` uses WAL mode to support concurrent writes during high-frequency telemetry logging without locking errors.
