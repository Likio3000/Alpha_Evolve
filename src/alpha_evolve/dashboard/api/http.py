from __future__ import annotations

from typing import Any

from django.http import JsonResponse


def json_response(data: Any, status: int = 200) -> JsonResponse:
    """Return a JsonResponse mirroring FastAPI's default serialization."""
    safe = not isinstance(data, (list, tuple))
    return JsonResponse(data, status=status, safe=safe)


def json_error(detail: str, status: int) -> JsonResponse:
    return json_response({"detail": detail}, status=status)
