# Daily Log

## 2026-11-03
In `src/aggregator.py`, sliding window calculations were updated to prevent memory accumulation during long-running daemon execution.

## 2026-11-03
The SQLite backend in `src/db.py` uses WAL mode to support concurrent writes during high-frequency telemetry logging without locking errors.

## 2026-11-04
Added exception handling in `src/alerts.py` to prevent webhook dispatch failures from crashing the main background monitoring loop.

## 2026-11-04
Added exception handling in `src/alerts.py` to prevent webhook dispatch failures from crashing the main background monitoring loop.

## 2026-11-04
In `dashboard/app.py`, date range filtering uses timezone-naive ISO strings which can cause edge-case discrepancies for users behind UTC.
