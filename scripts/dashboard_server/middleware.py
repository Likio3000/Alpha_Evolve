from __future__ import annotations

from django.http import HttpResponse


class SimpleCORS:
    """Minimal CORS middleware to mirror the permissive FastAPI setup."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.method == "OPTIONS":
            response = HttpResponse()
        else:
            response = self.get_response(request)
        # Mirror FastAPI CORSMiddleware defaults
        response["Access-Control-Allow-Origin"] = "*"
        response["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
        requested_headers = request.headers.get("Access-Control-Request-Headers")
        response["Access-Control-Allow-Headers"] = requested_headers or "*"
        response["Access-Control-Allow-Credentials"] = "true"
        return response
