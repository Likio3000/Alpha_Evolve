# Run and deploy your AI Studio app

This contains everything you need to run your app locally.

## Run Locally

Prerequisites: Node.js for the UI; Python env for the backend (see project README).

1) Install UI deps and run the dev server

```
npm install
npm run dev
```

2) Start the Iterative Dashboard backend (for live mode)

```
uv run scripts/iterative_dashboard_server.py
```

3) Open the UI in your browser (Vite default: http://localhost:5173)

Use the mode toggle at the top to switch among:

- Pipeline: Generate a `run_pipeline.py` command.
- Iterative: Generate a `scripts/auto_improve.py` command with passthrough flags.
- Dashboard: Launch and monitor the iterative improver with live updates, best Sharpe sparkline, current candidate info, and last run summary.
