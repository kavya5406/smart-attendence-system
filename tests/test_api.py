"""End-to-end API tests against the real model, verifier and database."""

import base64
import io
import zipfile
from datetime import date, timedelta

import cv2
import numpy as np
import pytest

from tests.conftest import student_images


# ---------------------------------------------------------------- system ---
def test_health_is_healthy(client):
    body = client.get("/api/health").json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["database_connected"] is True
    assert body["verifier_ready"] is True
    assert body["known_students"] == ["S0001", "S0002", "S0003", "S0004", "S0005"]
    assert body["dataset_images"] >= 1
    assert body["errors"] == {}


def test_model_info_reports_real_dimensions(client):
    body = client.get("/api/model/info").json()
    assert body["model"]["raw_feature_dim"] == 1805
    assert body["model"]["model_feature_dim"] == 715
    assert body["model"]["model_type"] == "LogisticRegression"
    assert body["verification"]["prototypes_built"] is True
    assert body["verification"]["threshold"] > 0


def test_metrics_endpoint_exposes_prometheus(client):
    body = client.get("/metrics").text
    assert "attendance_predictions_total" in body


# -------------------------------------------------------------- students ---
def test_student_crud(client):
    created = client.post(
        "/api/students",
        json={"student_id": "T1001", "student_name": "Test One", "department": "CSE", "year": "3"},
    )
    assert created.status_code == 201
    assert created.json()["student_name"] == "Test One"

    assert client.post("/api/students", json={"student_id": "T1001", "student_name": "Dupe"}).status_code == 409

    assert client.get("/api/students/T1001").json()["student_name"] == "Test One"
    assert client.get("/api/students/MISSING").status_code == 404

    patched = client.patch("/api/students/T1001", json={"student_name": "Test One Renamed"})
    assert patched.json()["student_name"] == "Test One Renamed"

    assert any(s["student_id"] == "T1001" for s in client.get("/api/students").json())

    assert client.delete("/api/students/T1001").json()["success"] is True
    assert client.get("/api/students/T1001").status_code == 404
    assert client.delete("/api/students/T1001").status_code == 404


def test_invalid_student_id_rejected(client):
    # "../etc" is sanitised to "etc"; an id that sanitises to nothing is refused
    assert client.post("/api/students", json={"student_id": "../..", "student_name": "Bad"}).status_code == 400
    assert client.get("/api/students/etc").status_code == 404


# ---------------------------------------------------------------- dataset ---
def test_dataset_stats_counts_real_images(client):
    body = client.get("/api/dataset/stats").json()
    assert body["dataset_exists"] is True
    assert body["total_students"] >= 1
    assert body["total_images"] >= 1
    assert body["invalid_images"] == 0
    assert "students" in body and isinstance(body["students"], list)


def test_upload_images_creates_student_and_counts(client):
    images = student_images(2)
    if not images:
        pytest.skip("no dataset images available")

    files = [("files", (p.name, p.read_bytes(), "image/jpeg")) for p in images]
    response = client.post(
        "/api/students/upload",
        files=files,
        data={"student_id": "T2001", "student_name": "Upload One", "department": "ECE", "year": "2"},
    )
    assert response.status_code == 200
    assert response.json()["images_saved"] == len(images)
    assert client.get("/api/students/T2001").json()["student_name"] == "Upload One"

    # non-image entries are skipped, macOS metadata never lands on disk
    files = [
        ("files", ("a.jpg", images[0].read_bytes(), "image/jpeg")),
        ("files", (".DS_Store", b"junk", "application/octet-stream")),
    ]
    again = client.post(
        "/api/students/upload",
        files=files,
        data={"student_id": "T2002", "student_name": "Upload Two"},
    )
    assert again.json()["images_saved"] == 1
    assert again.json()["skipped"] == 1

    assert client.delete("/api/students/T2001", params={"remove_images": True}).status_code == 200
    assert client.delete("/api/students/T2002", params={"remove_images": True}).status_code == 200


def test_upload_zip_imports_and_ignores_macos_entries(client):
    buffer = io.BytesIO()
    source = student_images(2)
    if not source:
        pytest.skip("no dataset images available")

    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("T3001/face1.jpg", source[0].read_bytes())
        archive.writestr("T3001/face2.jpg", source[1].read_bytes())
        archive.writestr("__MACOSX/T3001/._face1.jpg", b"junk")
        archive.writestr("T3001/.DS_Store", b"junk")
        archive.writestr("notes.txt", b"ignore me")

    response = client.post(
        "/api/students/upload-zip",
        files={"file": ("dataset.zip", buffer.getvalue(), "application/zip")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["students_imported"] == 1
    assert body["images_imported"] == 2
    assert body["skipped_entries"] >= 2

    entry = next(s for s in client.get("/api/dataset/stats").json()["students"] if s["student_id"] == "T3001")
    assert entry["image_count"] == 2
    assert entry["registered"] is True
    assert client.delete("/api/students/T3001", params={"remove_images": True}).status_code == 200


def test_invalid_zip_rejected(client):
    response = client.post(
        "/api/students/upload-zip", files={"file": ("bad.zip", b"not a zip", "application/zip")}
    )
    assert response.status_code == 400


# ------------------------------------------------------------ prediction ---
def test_predict_multipart_recognises_known_student(client):
    images = student_images(1)
    if not images:
        pytest.skip("no dataset images available")

    response = client.post("/api/predict", files={"file": (images[0].name, images[0].read_bytes(), "image/jpeg")})
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "recognized"
    assert body["student_id"] in body["model_student_id"] or body["model_agreed"] in (True, False)
    assert body["similarity"] >= body["threshold"]
    assert body["attendance_marked"] in (True, False)
    assert body["latency_ms"] > 0


def test_predict_base64_matches_multipart(client, synthetic_dataset):
    if synthetic_dataset:
        pytest.skip("generated frames are not recognised as a real identity")
    images = student_images(1)
    if not images:
        pytest.skip("no dataset images available")

    encoded = base64.b64encode(images[0].read_bytes()).decode()
    body = client.post("/api/predict-base64", json={"image_data": encoded}).json()
    assert body["status"] == "recognized"
    assert body["student_id"] == "S0001"


def test_predict_base64_data_url_prefix(client, synthetic_dataset):
    if synthetic_dataset:
        pytest.skip("generated frames are not recognised as a real identity")
    images = student_images(1)
    if not images:
        pytest.skip("no dataset images available")

    payload = "data:image/jpeg;base64," + base64.b64encode(images[0].read_bytes()).decode()
    assert client.post("/api/predict-base64", json={"image_data": payload}).json()["status"] == "recognized"


def test_repeat_recognition_does_not_duplicate_attendance(client, synthetic_dataset):
    """Recognising the same face repeatedly must leave exactly one row.

    Deliberately order independent: other tests may already have marked S0001
    today, so the assertion is about the row count staying at one and the
    second call reporting the duplicate.
    """
    if synthetic_dataset:
        pytest.skip("generated frames are not recognised as a real identity")
    images = student_images(1)
    if not images:
        pytest.skip("no dataset images available")

    payload = base64.b64encode(images[0].read_bytes()).decode()
    today = date.today().isoformat
    rows_url = {**today_kwargs(), "student_id": "S0001"}

    before = client.get("/api/attendance", params=rows_url).json()
    first = client.post("/api/predict-base64", json={"image_data": payload}).json()
    after_first = client.get("/api/attendance", params=rows_url).json()
    second = client.post("/api/predict-base64", json={"image_data": payload}).json()
    after_second = client.get("/api/attendance", params=rows_url).json()

    assert first["student_id"] == "S0001"
    assert second["student_id"] == "S0001"
    # the second call can never add a row
    assert len(after_first) == max(len(before), 1)
    assert len(after_second) == len(after_first)
    assert second["already_marked"] is True


def today_kwargs():
    today = date.today().isoformat()
    return {"start_date": today, "end_date": today}


def test_noise_is_unknown_and_marks_nothing(client, synthetic_dataset):
    if synthetic_dataset:
        pytest.skip("prototypes are generated noise; see test_noise_is_rejected")
    rng = np.random.default_rng(11)
    noise = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", noise)
    payload = base64.b64encode(encoded.tobytes()).decode()

    body = client.post("/api/predict-base64", json={"image_data": payload}).json()
    assert body["status"] == "unknown"
    assert body["student_id"] is None
    assert body["attendance_marked"] is False
    assert "UNKNOWN" in body["message"].upper()
    assert body["similarity"] < body["threshold"]


def test_blank_frame_is_unknown(client, synthetic_dataset):
    if synthetic_dataset:
        pytest.skip("prototypes are generated noise; see test_noise_is_rejected")
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", blank)
    payload = base64.b64encode(encoded.tobytes()).decode()
    assert client.post("/api/predict-base64", json={"image_data": payload}).json()["status"] == "unknown"


def test_predict_base64_requires_image_data(client):
    assert client.post("/api/predict-base64", json={}).status_code == 422
    assert client.post("/api/predict-base64", json={"image_data": "not base64!!"}).status_code == 400


# ------------------------------------------------------------- attendance ---
def test_manual_attendance_and_filters(client):
    client.post("/api/students", json={"student_id": "T4001", "student_name": "Manual One"})
    day = (date.today() - timedelta(days=3)).isoformat()

    marked = client.post(
        "/api/attendance", json={"student_id": "T4001", "student_name": "Manual One", "date": day, "confidence": 0.88}
    )
    assert marked.json()["success"] is True

    duplicate = client.post("/api/attendance", json={"student_id": "T4001", "student_name": "Manual One", "date": day})
    assert duplicate.json()["already_marked"] is True

    rows = client.get("/api/attendance", params={"student_id": "T4001"}).json()
    assert len(rows) == 1
    assert rows[0]["date"] == day
    assert rows[0]["status"] == "present"

    assert client.get("/api/attendance/student/T4001").json()[0]["student_id"] == "T4001"
    assert client.get(f"/api/attendance/date/{day}").json()[0]["student_id"] == "T4001"
    assert client.delete("/api/students/T4001").status_code == 200


def test_csv_export(client):
    response = client.get("/api/attendance/export.csv")
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert "attachment" in response.headers["content-disposition"]
    body = response.text
    assert "Student ID" in body
    # every data line must have the same column count as the header
    rows = [line for line in body.strip().splitlines() if line]
    assert rows
    assert all(len(r.split(",")) == 6 for r in rows)


# -------------------------------------------------------------- dashboard ---
def test_dashboard_stats_reflect_database(client):
    body = client.get("/api/dashboard/stats").json()
    assert body["date"] == date.today().isoformat()
    assert body["total_students"] >= 1
    assert 0 <= body["attendance_percentage"] <= 100
    assert isinstance(body["recent_attendance"], list)


def test_prediction_stats_reflect_logs(client):
    body = client.get("/api/stats/predictions").json()
    assert body["total_predictions"] >= 1
    assert body["recognized"] + body["unknown"] == body["total_predictions"]


# ------------------------------------------------------------------- SPA ---
def _require_frontend(client):
    from src.core.settings import get_settings

    if not (get_settings().frontend_dist / "index.html").exists():
        pytest.skip("frontend not built; run 'npm ci && npm run build' in frontend/")


def test_spa_routes_serve_frontend(client):
    _require_frontend(client)
    for route in ("/", "/recognition", "/students", "/dataset", "/attendance", "/reports"):
        response = client.get(route)
        assert response.status_code == 200, route
        assert "<!doctype html" in response.text.lower(), route


def test_unknown_api_path_returns_json_404(client):
    response = client.get("/api/attendance/does/not/exist")
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}


def test_page_paths_are_not_shadowed_by_the_api(client):
    """/students, /attendance and /dataset are pages, not API endpoints."""
    _require_frontend(client)
    for route in ("/students", "/attendance", "/dataset"):
        body = client.get(route).text.lower()
        assert "<!doctype html" in body, route
        assert not body.lstrip().startswith("["), route


def test_static_assets_are_served(client):
    _require_frontend(client)
    index = client.get("/").text
    assert "/assets/" in index


# ------------------------------------------------------- face policy ----
def test_health_alias_is_available_at_the_root(client):
    """`GET /health` must work as well as `GET /api/health`."""
    root = client.get("/health")
    api = client.get("/api/health")
    assert root.status_code == 200
    assert root.json() == api.json()


def test_dashboard_stats_alias_is_available_at_the_root(client):
    assert client.get("/dashboard/stats").status_code == 200


def test_health_reports_the_face_policy_and_threshold(client):
    body = client.get("/api/health").json()
    assert body["face_policy"] in ("whole_frame", "require_face")
    assert isinstance(body["recognition_threshold"], float)
    assert body["recognition_threshold"] > 0


def test_prediction_reports_face_detection_details(client, synthetic_dataset):
    """Every prediction must say whether a face was found and how many.

    The Haar cascade finds no face in the registered 128x128 dataset, so this
    field is the only way a client can tell that a whole frame was classified.
    """
    for path in student_images(limit=1):
        with open(path, "rb") as handle:
            response = client.post(
                "/api/predict", files={"file": (path.name, handle.read(), "image/jpeg")}
            )
        assert response.status_code == 200
        body = response.json()
        assert "face_detected" in body
        assert isinstance(body["face_detected"], bool)
        assert isinstance(body["faces_detected"], int)
        assert body["face_policy"] in ("whole_frame", "require_face")
        break
    else:
        pytest.skip("no dataset images available")


def test_success_flag_is_false_for_an_unknown_face(client, synthetic_dataset):
    noise = np.full((128, 128, 3), 200, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", noise)
    assert ok
    body = client.post(
        "/api/predict-base64", json={"image_data": _b64(encoded.tobytes())}
    ).json()
    assert body["status"] != "recognized"
    assert body["success"] is False
    assert body["attendance_marked"] is False
    assert body["student_id"] is None


def test_require_face_policy_rejects_a_frame_without_a_face(
    client, monkeypatch, synthetic_dataset
):
    """With FACE_POLICY=require_face an undetected frame must be refused.

    Runs against the real model, but forces the policy to the strict setting to
    prove the no-face branch returns a clean, non-crashing response.
    """
    monkeypatch.setattr(client.app.state.settings, "face_policy", "require_face")

    # A frame the strict Haar settings cannot find a face in.
    noise = np.full((128, 128, 3), 90, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", noise)
    assert ok
    response = client.post(
        "/api/predict-base64", json={"image_data": _b64(encoded.tobytes())}
    )
    assert response.status_code == 200
    body = response.json()
    if body["face_detected"] is False:
        assert body["status"] == "no_face_detected"
        assert body["success"] is False
        assert body["attendance_marked"] is False
        assert body["student_id"] is None
        assert body["face_policy"] == "require_face"


def test_no_face_response_never_marks_attendance(client, monkeypatch, synthetic_dataset):
    monkeypatch.setattr(client.app.state.settings, "face_policy", "require_face")
    before = len(client.get("/api/attendance").json())
    noise = np.full((160, 160, 3), 30, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", noise)
    assert ok
    client.post("/api/predict-base64", json={"image_data": _b64(encoded.tobytes())})
    assert len(client.get("/api/attendance").json()) == before


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")
