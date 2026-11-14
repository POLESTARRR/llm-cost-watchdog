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

## 2026-11-05
The SQLite backend in `src/db.py` uses WAL mode to support concurrent writes during high-frequency telemetry logging without locking errors.

## 2026-11-05
Refactored cost calculation logic in `src/pricing.py` to cache model pricing rates for 24 hours, reducing redundant lookups.

## 2026-11-05
In `src/cost_tracker.py`, token counting for streaming responses uses an approximate heuristic when chunk usage metadata is omitted by the provider API.

## 2026-11-05
In `dashboard/app.py`, date range filtering uses timezone-naive ISO strings which can cause edge-case discrepancies for users behind UTC.

## 2026-11-06
Added exception handling in `src/alerts.py` to prevent webhook dispatch failures from crashing the main background monitoring loop.

## 2026-11-06
The CLI command in `ccost_cli.py` now supports outputting cost summaries in CSV format for easier spreadsheet export.

## 2026-11-06
In `dashboard/app.py`, date range filtering uses timezone-naive ISO strings which can cause edge-case discrepancies for users behind UTC.

## 2026-11-14
In `src/aggregator.py`, sliding window calculations were updated to prevent memory accumulation during long-running daemon execution.

## 2026-11-14
Added unit tests in `tests/test_tracker.py` covering rate-limit retry backoff behavior under simulated 429 responses.
