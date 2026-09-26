"""FastAPI backend for the Smart Attendance System.

The recognition pipeline is::

    browser camera frame
        -> face detection (Haar / optional dlib)
        -> preprocessing (64x64 gray, equalise, blur)
        -> feature extraction (HOG + LBP + geometric + statistical + shape)
        -> StandardScaler
        -> FeatureSelector
        -> existing trained LogisticRegression
        -> LabelEncoder -> student id
        -> prototype verification (unknown-face rejection)
        -> attendance record

Nothing in this module fabricates a result: every response is derived from the
trained artifacts, the registered dataset and the database.
"""

from __future__ import annotations

import base64
import binascii
import io
import time
import zipfile
from contextlib import asynccontextmanager
from datetime import date as date_cls
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from ..core.settings import get_settings
from ..data.database import AttendanceDatabase
from ..data.dataset import (
    DatasetManager,
    VALID_EXTENSIONS,
    is_ignored_dir,
    is_ignored_file,
    sanitise_student_id,
)
from ..features.feature_extractor import FeatureExtractor
from ..models.predict import AttendancePredictor
from ..preprocessing.face_detector import FaceDetector
from ..preprocessing.image_processor import ImageProcessor
from ..verification.verifier import FaceVerifier
from . import schemas
from .monitoring import ATTENDANCE_MARKED, PREDICTIONS, PREDICTION_LATENCY, build_metrics_router

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = settings
    app.state.face_detector = FaceDetector()
    app.state.image_processor = ImageProcessor()
    app.state.feature_extractor = FeatureExtractor()
    app.state.predictor = AttendancePredictor()
    app.state.database = AttendanceDatabase()
    app.state.dataset = DatasetManager()
    app.state.verifier = FaceVerifier(app.state.predictor)

    # Build/load identity prototypes from the registered dataset.
    app.state.verifier.build(
        app.state.feature_extractor,
        app.state.image_processor,
        app.state.face_detector,
    )
    yield
    app.state.database.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Smart Attendance System API",
        description=(
            "Face recognition attendance using the existing classical ML model "
            "(LogisticRegression + StandardScaler + FeatureSelector)."
        ),
        version="2.0.0",
        lifespan=lifespan,
    )

    # When the frontend is served from the same origin (the production setup)
    # any origin is harmless. Set CORS_ORIGINS to a comma separated list to lock
    # it down when the frontend is hosted separately.
    allow_origins = settings.cors_origins or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    build_metrics_router(app)
    _register_routes(app)
    return app


def _student_row(row: dict, image_count: int) -> dict:
    """Normalise a student record for the API (datetimes -> ISO strings)."""
    data = dict(row)
    created = data.get("created_at")
    if hasattr(created, "isoformat"):
        data["created_at"] = created.isoformat()
    elif created is not None:
        data["created_at"] = str(created)
    data["image_count"] = image_count
    return data


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _predict_image(app: FastAPI, image: np.ndarray) -> schemas.PredictionResponse:
    """Run the full recognition pipeline for one BGR image."""
    start = time.time()

    face = app.state.face_detector.crop_face(image)
    face_detected = face is not None
    # Matches train.py: fall back to the whole frame when no face is found.
    source = face if face is not None else image

    processed = app.state.image_processor.preprocess(source)
    features = app.state.feature_extractor.extract_all(
        cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    )

    model_student_id, confidence, diagnostics = app.state.predictor.predict_detailed(features)
    verification = app.state.verifier.verify(features, model_student_id)
    latency_ms = (time.time() - start) * 1000

    if not verification.accepted:
        PREDICTIONS.labels(outcome="unknown").inc()
        PREDICTION_LATENCY.observe(latency_ms / 1000.0)
        app.state.database.log_prediction(
            student_id=None,
            confidence=confidence,
            similarity=verification.similarity,
            status="unknown",
            latency_ms=latency_ms,
            model_student_id=model_student_id,
        )
        return schemas.PredictionResponse(
            status="unknown",
            student_id=None,
            student_name=None,
            confidence=confidence,
            similarity=verification.similarity,
            threshold=verification.threshold,
            attendance_marked=False,
            message="UNKNOWN STUDENT. Please try again.",
            face_detected=face_detected,
            model_student_id=model_student_id,
            model_agreed=verification.student_id == model_student_id,
            latency_ms=round(latency_ms, 2),
            probabilities=diagnostics.get("probabilities"),
            verification_reason=verification.reason,
        )

    student = app.state.database.get_student(verification.student_id)
    student_name = (student or {}).get("student_name") or verification.student_id
    if student is None:
        # Recognised face with no database record: create the student so the
        # identifier stays consistent with the trained label set.
        app.state.database.register_student(verification.student_id, student_name)

    marked = app.state.database.mark_attendance(
        verification.student_id, student_name, confidence=confidence
    )
    PREDICTIONS.labels(outcome="recognized").inc()
    PREDICTION_LATENCY.observe(latency_ms / 1000.0)
    ATTENDANCE_MARKED.labels(result="duplicate" if marked.get("already_marked") else "marked").inc()
    app.state.database.log_prediction(
        student_id=verification.student_id,
        confidence=confidence,
        similarity=verification.similarity,
        status="recognized",
        latency_ms=latency_ms,
        model_student_id=model_student_id,
    )

    return schemas.PredictionResponse(
        status="recognized",
        student_id=verification.student_id,
        student_name=student_name,
        confidence=confidence,
        similarity=verification.similarity,
        threshold=verification.threshold,
        attendance_marked=bool(marked.get("success")),
        already_marked=bool(marked.get("already_marked")),
        message=(
            "Attendance already marked for today."
            if marked.get("already_marked")
            else "Attendance marked successfully."
        ),
        face_detected=face_detected,
        model_student_id=model_student_id,
        model_agreed=verification.student_id == model_student_id,
        latency_ms=round(latency_ms, 2),
        probabilities=diagnostics.get("probabilities"),
        verification_reason=verification.reason,
    )


def _decode_upload(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return image


def _decode_base64(payload: str) -> np.ndarray:
    if "," in payload and payload.strip().startswith("data:"):
        payload = payload.split(",", 1)[1]
    try:
        raw = base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="Invalid base64 image data")
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return image


# ----------------------------------------------------------------------
# routes
# ----------------------------------------------------------------------
def _register_routes(app: FastAPI) -> None:
    # Every API route lives under /api so the frontend can own clean page paths
    # such as /students, /attendance and /dataset without colliding with them.
    router = APIRouter()

    # ---------------- system ----------------
    @router.get("/health", response_model=schemas.HealthResponse)
    def health():
        predictor: AttendancePredictor = app.state.predictor
        database: AttendanceDatabase = app.state.database
        verifier: FaceVerifier = app.state.verifier

        db_ok = True
        try:
            database.count_students()
        except Exception:
            db_ok = False

        try:
            dataset_images = sum(
                len(app.state.dataset.list_images(sid)) for sid in app.state.dataset.student_ids()
            )
        except Exception:
            dataset_images = 0

        info = predictor.model_info()
        return schemas.HealthResponse(
            status="healthy" if (predictor.is_ready and db_ok) else "degraded",
            model_loaded=predictor.is_ready,
            database_connected=db_ok,
            verifier_ready=verifier.built,
            total_students_registered=database.count_students() if db_ok else 0,
            model_type=info.get("model_type"),
            known_students=info.get("known_students", []),
            dataset_images=dataset_images,
            errors=predictor.load_errors,
        )

    @router.get("/model/info")
    def model_info():
        """Descriptive model metadata. No accuracy is invented."""
        return {
            "model": app.state.predictor.model_info(),
            "verification": app.state.verifier.info(),
        }

    # ---------------- prediction ----------------
    @router.post("/predict", response_model=schemas.PredictionResponse)
    async def predict(file: UploadFile = File(...)):
        data = await file.read()
        return _predict_image(app, _decode_upload(data))

    @router.post("/predict-base64", response_model=schemas.PredictionResponse)
    async def predict_base64(payload: dict):
        if "image_data" not in payload:
            raise HTTPException(status_code=422, detail="Field 'image_data' is required")
        return _predict_image(app, _decode_base64(payload["image_data"]))

    # ---------------- students ----------------
    @router.get("/students", response_model=List[schemas.StudentResponse])
    def list_students():
        database: AttendanceDatabase = app.state.database
        dataset: DatasetManager = app.state.dataset
        out = []
        for row in database.get_all_students():
            out.append(
                schemas.StudentResponse(
                    **_student_row(row, len(dataset.list_images(row["student_id"])))
                )
            )
        return out

    @router.post("/students", response_model=schemas.StudentResponse, status_code=201)
    def create_student(body: schemas.StudentCreate):
        student_id = sanitise_student_id(body.student_id)
        if not student_id:
            raise HTTPException(status_code=400, detail="Invalid student id")
        if app.state.database.get_student(student_id):
            raise HTTPException(status_code=409, detail=f"Student {student_id} already exists")
        app.state.database.register_student(
            student_id, body.student_name, body.email, body.department, body.year
        )
        row = app.state.database.get_student(student_id)
        return _student_row(row, len(app.state.dataset.list_images(student_id)))

    @router.get("/students/{student_id}", response_model=schemas.StudentResponse)
    def get_student(student_id: str):
        row = app.state.database.get_student(student_id)
        if not row:
            raise HTTPException(status_code=404, detail="Student not found")
        return _student_row(row, len(app.state.dataset.list_images(student_id)))

    @router.patch("/students/{student_id}", response_model=schemas.StudentResponse)
    def update_student(student_id: str, body: schemas.StudentUpdate):
        if not app.state.database.get_student(student_id):
            raise HTTPException(status_code=404, detail="Student not found")
        row = app.state.database.update_student(student_id, **body.model_dump(exclude_none=True))
        return _student_row(row, len(app.state.dataset.list_images(student_id)))

    @router.delete("/students/{student_id}")
    def delete_student(student_id: str, remove_images: bool = Query(False)):
        if not app.state.database.get_student(student_id):
            raise HTTPException(status_code=404, detail="Student not found")
        app.state.database.delete_student(student_id)
        if remove_images:
            app.state.dataset.remove_student(student_id)
            _rebuild_verifier(app)
        return {"success": True, "student_id": student_id, "images_removed": remove_images}

    # ---------------- dataset ----------------
    @router.get("/dataset/stats", response_model=schemas.DatasetStatsResponse)
    def dataset_stats(deep: bool = Query(False), check_faces: bool = Query(False)):
        dataset: DatasetManager = app.state.dataset
        database: AttendanceDatabase = app.state.database
        result = dataset.scan(check_faces=check_faces, deep=deep)

        registered = {s["student_id"] for s in database.get_all_students()}
        students = []
        for payload in result["students"]:
            payload = dict(payload)
            payload["registered"] = payload["student_id"] in registered
            payload["status"] = "registered" if payload["registered"] else "not_registered"
            if payload["registered"]:
                row = database.get_student(payload["student_id"])
                if row and row.get("student_name"):
                    payload["student_name"] = row["student_name"]
            students.append(payload)

        return schemas.DatasetStatsResponse(
            dataset_path=result["dataset_path"],
            dataset_exists=result["dataset_exists"],
            total_students=result["total_students"],
            total_images=result["total_images"],
            valid_images=result["valid_images"],
            invalid_images=result["invalid_images"],
            duplicate_images=result["duplicate_images"],
            students=students,
        )

    @router.post("/students/upload")
    async def upload_student_images(
        files: List[UploadFile] = File(...),
        student_id: str = Form(...),
        student_name: str = Form(...),
        department: str = Form(""),
        year: str = Form(""),
    ):
        sid = sanitise_student_id(student_id)
        if not sid:
            raise HTTPException(status_code=400, detail="Invalid student id")

        folder = app.state.dataset.ensure_student_dir(sid)
        saved, skipped = 0, 0
        for upload in files:
            name = Path(upload.filename or "").name
            if not name or is_ignored_file(name) or Path(name).suffix.lower() not in VALID_EXTENSIONS:
                skipped += 1
                continue
            (folder / name).write_bytes(await upload.read())
            saved += 1

        app.state.database.register_student(sid, student_name, "", department, year)
        _rebuild_verifier(app)
        return {
            "success": True,
            "student_id": sid,
            "student_name": student_name,
            "images_saved": saved,
            "skipped": skipped,
        }

    @router.post("/students/upload-zip")
    async def upload_dataset_zip(file: UploadFile = File(...), register: bool = Query(True)):
        """Import a dataset ZIP laid out as ``<STUDENT_ID>/<image>``.

        Ignores ``__MACOSX/``, ``.DS_Store`` and ``._*`` entries, and refuses
        any name that would escape the dataset directory.
        """
        raw = await file.read()
        try:
            archive = zipfile.ZipFile(io.BytesIO(raw))
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Invalid zip file")

        imported: dict = {}
        skipped = 0
        with archive:
            for entry in archive.namelist():
                if entry.endswith("/"):
                    continue
                parts = [p for p in Path(entry).parts if p not in (".",)]
                if len(parts) < 2:
                    skipped += 1
                    continue
                if any(is_ignored_dir(p) or is_ignored_file(p) for p in parts):
                    skipped += 1
                    continue

                student_id = sanitise_student_id(parts[0])
                filename = Path(parts[-1]).name
                if not student_id or not filename:
                    skipped += 1
                    continue
                if Path(filename).suffix.lower() not in VALID_EXTENSIONS:
                    skipped += 1
                    continue

                folder = app.state.dataset.ensure_student_dir(student_id)
                (folder / filename).write_bytes(archive.read(entry))
                imported[student_id] = imported.get(student_id, 0) + 1

        if register:
            for sid in imported:
                if not app.state.database.get_student(sid):
                    app.state.database.register_student(sid, f"Student {sid}")

        _rebuild_verifier(app)
        return {
            "success": True,
            "students_imported": len(imported),
            "images_imported": sum(imported.values()),
            "skipped_entries": skipped,
            "details": [{"student_id": k, "images": v} for k, v in sorted(imported.items())],
        }

    @router.post("/dataset/rebuild-verifier")
    def rebuild_verifier():
        ok = _rebuild_verifier(app, force=True)
        return {"success": ok, "verification": app.state.verifier.info()}

    # ---------------- attendance ----------------
    @router.post("/attendance", response_model=schemas.MarkAttendanceResponse)
    def mark_attendance(body: schemas.AttendanceCreate):
        database: AttendanceDatabase = app.state.database
        student = database.get_student(body.student_id)
        name = body.student_name or (student or {}).get("student_name") or body.student_id
        return database.mark_attendance(
            body.student_id,
            name,
            confidence=body.confidence,
            status=body.status,
            attendance_date=body.date,
        )

    @router.get("/attendance", response_model=List[schemas.AttendanceRecord])
    def get_attendance(
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        student_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = Query(1000, le=10000),
    ):
        return app.state.database.get_attendance_history(
            start_date, end_date, student_id, status, limit
        )

    @router.get("/attendance/today", response_model=List[schemas.AttendanceRecord])
    def attendance_today():
        return app.state.database.get_attendance_by_date()

    @router.get("/attendance/date/{day}", response_model=List[schemas.AttendanceRecord])
    def attendance_by_date(day: str):
        return app.state.database.get_attendance_by_date(day)

    @router.get("/attendance/student/{student_id}", response_model=List[schemas.AttendanceRecord])
    def attendance_for_student(student_id: str, limit: int = Query(30, le=500)):
        return app.state.database.get_student_attendance(student_id, limit)

    @router.get("/attendance/export.csv")
    def export_attendance_csv(
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        student_id: Optional[str] = None,
        status: Optional[str] = None,
    ):
        import csv
        import io as _io

        rows = app.state.database.get_attendance_history(start_date, end_date, student_id, status)
        buffer = _io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["Student ID", "Student Name", "Date", "Time", "Status", "Confidence"])
        for row in rows:
            writer.writerow([
                row["student_id"], row["student_name"], row["date"],
                row["time"], row["status"],
                "" if row["confidence"] is None else round(row["confidence"], 4),
            ])
        stamp = date_cls.today().isoformat()
        return Response(
            content=buffer.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="attendance-{stamp}.csv"'},
        )

    # ---------------- dashboard ----------------
    @router.get("/dashboard/stats", response_model=schemas.DashboardStats)
    def dashboard_stats():
        return app.state.database.get_dashboard_stats()

    @router.get("/stats/predictions")
    def prediction_stats():
        return app.state.database.get_prediction_stats()

    # ---------------- frontend ----------------
    def _mount_frontend(application: FastAPI) -> None:
        dist = settings.frontend_dist
        if not dist.exists():
            @application.get("/", include_in_schema=False)
            def no_frontend():
                return JSONResponse(
                    {
                        "message": "Frontend build not found. Run 'npm install && npm run build' in frontend/.",
                        "api_docs": "/docs",
                    }
                )
            return

        assets = dist / "assets"
        if assets.exists():
            application.mount("/assets", StaticFiles(directory=assets), name="assets")

        index_file = dist / "index.html"

        @application.get("/{full_path:path}", include_in_schema=False)
        def spa(full_path: str):
            # The API is mounted under /api, so only that prefix (plus the
            # Prometheus endpoint) is a backend path here. Everything else is a
            # frontend page and must return index.html for client-side routing.
            parts = [p for p in full_path.split("/") if p]
            if parts and parts[0] in {"api", "metrics"}:
                return JSONResponse({"detail": "Not Found"}, status_code=404)

            candidate = (dist / full_path).resolve()
            try:
                candidate.relative_to(dist.resolve())
            except ValueError:
                return JSONResponse({"detail": "Not Found"}, status_code=404)
            if candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(index_file)

    app.include_router(router, prefix="/api")
    _mount_frontend(app)


def _rebuild_verifier(app: FastAPI, force: bool = False) -> bool:
    return app.state.verifier.build(
        app.state.feature_extractor,
        app.state.image_processor,
        app.state.face_detector,
        force=force,
    )


app = create_app()
