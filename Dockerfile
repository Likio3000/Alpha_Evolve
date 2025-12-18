FROM node:20-alpine AS ui

WORKDIR /app
COPY dashboard-ui/package.json dashboard-ui/package-lock.json ./dashboard-ui/
RUN npm --prefix dashboard-ui ci
COPY dashboard-ui ./dashboard-ui
RUN npm --prefix dashboard-ui run build


FROM python:3.12-slim AS runtime

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    AE_PIPELINE_DIR=/app/pipeline_runs_cs

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md LICENSE ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY docs ./docs

# UI bundle (built in the ui stage)
COPY --from=ui /app/dashboard-ui/dist ./dashboard-ui/dist
COPY --from=ui /app/dashboard-ui/index.html ./dashboard-ui/index.html
COPY --from=ui /app/dashboard-ui/public ./dashboard-ui/public

RUN pip install --no-cache-dir -r requirements.txt && pip install -e .

EXPOSE 8000

CMD ["python", "scripts/run_dashboard.py"]
