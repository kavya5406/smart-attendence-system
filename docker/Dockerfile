# syntax=docker/dockerfile:1
#
# Self-contained production image: the React app is built in a Node stage and
# the FastAPI backend serves the compiled bundle from the same origin, so the
# browser camera works over one HTTPS URL with no CORS setup.
#
# Build:  docker build -f docker/Dockerfile -t smart-attendance .
# Run:    docker run -p 8000:8000 -v "$(pwd)/data:/app/data" smart-attendance

# ---------------------------------------------------------------- frontend ---
FROM node:20-slim AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ----------------------------------------------------------------- backend ---
FROM python:3.11-slim

# OpenCV needs these at runtime; build-essential is not required because every
# wheel used here ships prebuilt binaries.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FRONTEND_DIST=/app/frontend/dist

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Writable state lives here: mount a volume to keep the database and the
# registered dataset across restarts.
RUN mkdir -p /app/data/raw /app/data/processed /app/data/database /app/mlruns /app/model_registry

COPY config/ config/
COPY src/ src/
COPY scripts/ scripts/
COPY tests/ tests/
# The four existing trained artifacts. Face images are never baked into the
# image: they are uploaded at runtime through the Dataset page.
COPY models/ models/

# The compiled frontend lives exactly where the backend looks for it.
COPY --from=frontend /build/dist/ frontend/dist/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
