# Alpha Evolve – Live Dashboard (monitor-only)

This dashboard accompanies the iterative improver, streaming live progress and per‑generation diagnostics. Configuration remains in TOML; the UI is for monitoring and insight.

## Run

1) Start the backend API (serves SSE and live control):

```
uv run scripts/iterative_dashboard_server.py
```

2) Run the dashboard UI:

```
cd dashboard-ui
npm install
npm run dev
```

Open http://localhost:5173. The app connects to http://127.0.0.1:8000 by default.

## Features

- Start/stop an iterative run (auto_improve) from the UI
- Live status, candidate info, Sharpe(best) sparkline
- Per‑generation snapshot table (best fitness, median, quantiles, eval seconds)
- Last run folder + best backtest Sharpe
- Live log tail

## Notes

- The backend parses `DIAG {json}` lines emitted per generation by evolve_alphas.py and forwards them as SSE events.
- For production, consider adding auth to POST /api/run and restricting CORS origins.

