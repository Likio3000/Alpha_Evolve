from __future__ import annotations

from django.http import HttpResponse
from django.utils.cache import patch_vary_headers


class SimpleCORS:
    """Minimal CORS middleware to mirror the permissive FastAPI setup."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        origin = request.headers.get("Origin")
        is_preflight = request.method == "OPTIONS"

        response = HttpResponse(status=204) if is_preflight else self.get_response(request)

        allow_methods = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
        response["Access-Control-Allow-Methods"] = allow_methods

        requested_headers = request.headers.get("Access-Control-Request-Headers")
        if requested_headers:
            response["Access-Control-Allow-Headers"] = requested_headers
        else:
            response.setdefault("Access-Control-Allow-Headers", "*")

        if origin:
            response["Access-Control-Allow-Origin"] = origin
            response["Access-Control-Allow-Credentials"] = "true"
            patch_vary_headers(response, ["Origin"])
        else:
            response.setdefault("Access-Control-Allow-Origin", "*")
        # 0-length body for preflight requests helps some clients
        if is_preflight:
            response["Content-Length"] = "0"
        return response
