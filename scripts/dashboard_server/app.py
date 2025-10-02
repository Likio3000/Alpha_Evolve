from __future__ import annotations

import os
from functools import lru_cache

import django
from django.conf import settings
from django.core.asgi import get_asgi_application


DJANGO_SETTINGS_MODULE = "scripts.dashboard_server.django_project.settings"


def _ensure_setup() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", DJANGO_SETTINGS_MODULE)
    if not settings.configured:
        django.setup()


@lru_cache(maxsize=1)
def create_app():
    """Return the Django ASGI application, initializing Django on first use."""
    _ensure_setup()
    return get_asgi_application()


app = create_app()
