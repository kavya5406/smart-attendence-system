"""Pydantic request/response schemas for the API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ----------------------------------------------------------------------
# prediction
# ----------------------------------------------------------------------
class PredictionResponse(BaseModel):
    success: bool = Field(
        True,
        description=(
            "False whenever attendance was NOT recorded for a student: unknown "
            "face, no face detected, or an error. Check this before showing a "
            "success message in the UI."
        ),
    )
    status: str = Field(
        ...,
        description=(
            "recognized | unknown | no_face_detected | invalid_image | error. "
            "'no_face_detected' only occurs when FACE_POLICY=require_face."
        ),
    )
    student_id: Optional[str] = None
    student_name: Optional[str] = None
    confidence: Optional[float] = Field(
        None,
        description=(
            "max(predict_proba) from the trained model. NOTE: this value is "
            "saturated (~1.0) and is not a reliable identity signal; use "
            "'similarity' for the calibrated verification score."
        ),
    )
    similarity: Optional[float] = Field(
        None, description="Calibrated cosine similarity to the matched student's prototypes."
    )
    threshold: Optional[float] = None
    attendance_marked: bool = False
    already_marked: bool = False
    message: str = ""
    face_detected: bool = Field(
        True,
        description=(
            "False when the detector found no face and the whole frame was "
            "used instead. Set FACE_POLICY=require_face to reject such frames."
        ),
    )
    faces_detected: int = Field(
        0,
        description=(
            "Number of faces found by the detector. When more than one is "
            "present the largest box wins (deterministic rule)."
        ),
    )
    face_policy: str = Field(
        "whole_frame",
        description="'whole_frame' (training-consistent) or 'require_face'.",
    )
    model_student_id: Optional[str] = Field(
        None, description="Student id predicted by the trained model before verification."
    )
    model_agreed: Optional[bool] = None
    latency_ms: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    verification_reason: Optional[str] = None


# ----------------------------------------------------------------------
# students
# ----------------------------------------------------------------------
class StudentCreate(BaseModel):
    student_id: str
    student_name: str
    email: str = ""
    department: str = ""
    year: str = ""


class StudentUpdate(BaseModel):
    student_name: Optional[str] = None
    email: Optional[str] = None
    department: Optional[str] = None
    year: Optional[str] = None


class StudentResponse(BaseModel):
    student_id: str
    student_name: str
    email: str = ""
    department: str = ""
    year: str = ""
    created_at: Optional[str] = None
    image_count: int = 0
    registered: bool = True


class StudentDatasetEntryResponse(BaseModel):
    student_id: str
    student_name: str = ""
    image_count: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    duplicate_count: int = 0
    images_with_faces: int = 0
    registered: bool = False
    status: str = "not_registered"


class DatasetStatsResponse(BaseModel):
    dataset_path: str
    dataset_exists: bool
    total_students: int
    total_images: int
    valid_images: int
    invalid_images: int
    duplicate_images: int
    students: List[StudentDatasetEntryResponse] = []


# ----------------------------------------------------------------------
# attendance
# ----------------------------------------------------------------------
class AttendanceCreate(BaseModel):
    student_id: str
    student_name: str = ""
    status: str = "present"
    confidence: Optional[float] = None
    date: Optional[str] = None


class AttendanceRecord(BaseModel):
    id: Optional[int] = None
    student_id: str
    student_name: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    timestamp: Optional[str] = None
    status: str = "present"
    confidence: Optional[float] = None


class MarkAttendanceResponse(BaseModel):
    success: bool
    already_marked: bool = False
    student_id: str
    status: str = "present"
    date: Optional[str] = None
    time: Optional[str] = None
    message: Optional[str] = None


# ----------------------------------------------------------------------
# dashboard / system
# ----------------------------------------------------------------------
class DashboardStats(BaseModel):
    total_students: int
    present_today: int
    absent_today: int
    attendance_percentage: float
    date: str
    recent_attendance: List[AttendanceRecord] = []


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    database_connected: bool
    verifier_ready: bool = False
    total_students_registered: int = 0
    model_type: Optional[str] = None
    known_students: List[str] = []
    dataset_images: int = 0
    face_policy: str = "whole_frame"
    recognition_threshold: Optional[float] = None
    errors: Dict[str, str] = {}
