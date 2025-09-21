from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from .ui_meta import router as ui_meta_router
from .routes.run_auto_improve import router as auto_router
from .routes.run_pipeline import router as pipeline_router
from .routes.config import router as config_router
from .routes.runs import router as runs_router
from .routes.selfplay import router as selfplay_router
from .helpers import ROOT
from .health import router as health_router


def create_app() -> FastAPI:
    app = FastAPI(title="Alpha Evolve Dashboard API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Serve built UI (if available) under /ui
    ui_dir: Path = ROOT / "dashboard-ui" / "dist"
    if ui_dir.exists():
        app.mount("/ui", StaticFiles(directory=str(ui_dir), html=True), name="ui")
    
    # Redirect root to /ui for convenience
    @app.get("/", include_in_schema=False)
    def _root_redirect():
        return RedirectResponse(url="/ui")

    # Routers
    app.include_router(health_router)
    app.include_router(ui_meta_router)
    app.include_router(auto_router)
    app.include_router(pipeline_router)
    app.include_router(config_router)
    app.include_router(runs_router)
    app.include_router(selfplay_router)
    return app


app = create_app()
