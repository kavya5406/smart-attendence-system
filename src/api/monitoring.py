"""Prometheus metrics for the recognition API.

These are operational counters only. They never influence a prediction result
and the service runs fine without Prometheus or Grafana.
"""

from __future__ import annotations

from fastapi import FastAPI, Request, Response

try:  # pragma: no cover - optional dependency
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

    PROMETHEUS_AVAILABLE = True
except Exception:  # pragma: no cover
    PROMETHEUS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"

PREDICTIONS = Counter(
    "attendance_predictions_total",
    "Total face prediction requests",
    ["outcome"],
)
PREDICTION_LATENCY = Histogram(
    "attendance_prediction_latency_seconds",
    "End-to-end face recognition latency",
    buckets=(0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
ATTENDANCE_MARKED = Counter(
    "attendance_marked_total",
    "Attendance records marked",
    ["result"],
)
API_ERRORS = Counter(
    "attendance_api_errors_total",
    "API errors",
    ["path"],
)


def build_metrics_router(app: FastAPI) -> None:
    @app.middleware("http")
    async def observe(request: Request, call_next):
        try:
            response = await call_next(request)
        except Exception:
            API_ERRORS.labels(path=request.url.path).inc()
            raise
        if response.status_code >= 500:
            API_ERRORS.labels(path=request.url.path).inc()
        return response

    @app.get("/metrics", include_in_schema=False)
    def metrics():
        if not PROMETHEUS_AVAILABLE:
            return Response(
                "prometheus_client is not installed; metrics unavailable.\n",
                media_type="text/plain",
            )
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
