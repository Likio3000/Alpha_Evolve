# Rebuilding the Dashboard UI Assets

The Python dashboard server ships with a pre-built static UI under
`dashboard-ui/dist/`. If you customise the front-end or maintain the private
React/Vite project that generates those files, follow this workflow to refresh
the assets served by FastAPI.

1. Clone or update the UI source repository (internal/private) locally.
2. Install the Node toolchain (Node.js 18+, plus your package manager of
   choice – `npm`, `yarn`, or `pnpm`).
3. From the UI project root run the normal build command, for example:

   ```bash
   pnpm install
   pnpm build
   # or: npm install && npm run build
   ```

4. Copy the generated `dist/` output into this repository’s
   `dashboard-ui/dist/` directory (replace the existing HTML/asset files).

During development you can run the UI in dev-server mode and point it at the
FastAPI backend by exporting `AE_PIPELINE_DIR` (so the API knows where to find
runs) and launching `uv run scripts/run_dashboard.py`. Configure the dev UI to
proxy API requests to `http://127.0.0.1:8000`.

When you rebuild, remember to commit the updated static files if you want them
bundled for other users, or keep them untracked if they are environment
specific.
