# Forecast monitor

The dashboard reads the forecast service’s SQLite ledger and reports. It refreshes every 30 seconds and runs on the ROCm host’s loopback interface.

From the repository root, run

    .venv/bin/python -B reports/forecast_dashboard/serve.py

Open http://127.0.0.1:8765. Use the --port option to choose a different port or --state for a separate forecast run.

- Live results shows the timetable, job statuses, filters for publication timing and geography, and CSV downloads.
- Backtest compares exploratory historical windows and skill on matched hours.
- Feed readiness shows preflight coverage and searchable bidding-zone identifiers.

The dashboard reads data from the ledger in read-only mode.
