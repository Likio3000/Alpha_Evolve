from __future__ import annotations

import importlib

import httpx
import pytest


@pytest.mark.asyncio
async def test_ui_redirects_to_trailing_slash_and_serves_assets(tmp_path, monkeypatch):
    ui_dist = tmp_path / "ui_dist"
    (ui_dist / "assets").mkdir(parents=True)
    (ui_dist / "index.html").write_text(
        '<!doctype html><html><head><script type="module" src="./assets/app.js"></script></head>'
        '<body><div id="root"></div></body></html>',
        encoding="utf-8",
    )
    (ui_dist / "assets" / "app.js").write_text("console.log('ok');\n", encoding="utf-8")
    (ui_dist / "favicon.svg").write_text("<svg></svg>\n", encoding="utf-8")

    urls_mod = importlib.import_module("alpha_evolve.dashboard.api.django_project.urls")
    monkeypatch.setattr(urls_mod, "UI_DIST", ui_dist)

    app_mod = importlib.reload(importlib.import_module("alpha_evolve.dashboard.api.app"))
    transport = httpx.ASGITransport(app=app_mod.create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.get("/ui", follow_redirects=False)
        assert resp.status_code in {301, 302, 307, 308}
        assert resp.headers.get("location") == "/ui/"

        ok = await client.get("/ui/")
        assert ok.status_code == 200
        assert "root" in ok.text

        asset = await client.get("/ui/assets/app.js")
        assert asset.status_code == 200
        assert "console.log" in asset.text

