"""Persistence layer.

Uses SQLAlchemy Core so the same code runs on SQLite (default, zero setup) and
PostgreSQL (production, via ``DATABASE_URL``).

Duplicate attendance is prevented at the **database** level with a unique
constraint on ``(student_id, attendance_date)`` in addition to the
application-level check, so the rule cannot be bypassed by a direct write.
"""

from __future__ import annotations

import json
from datetime import datetime, date as date_cls
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    func,
    select,
    insert,
    update,
    delete,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

from ..core.settings import get_settings

metadata = MetaData()

students = Table(
    "students",
    metadata,
    Column("student_id", String(64), primary_key=True),
    Column("student_name", String(255), nullable=False),
    Column("email", String(255), nullable=False, server_default=""),
    Column("department", String(128), nullable=False, server_default=""),
    Column("year", String(32), nullable=False, server_default=""),
    Column("created_at", DateTime, nullable=False, server_default=func.now()),
)

attendance = Table(
    "attendance",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("student_id", String(64), nullable=False),
    Column("student_name", String(255), nullable=False, server_default=""),
    Column("attendance_date", Date, nullable=False),
    Column("marked_at", DateTime, nullable=False, server_default=func.now()),
    Column("status", String(32), nullable=False, server_default="present"),
    Column("confidence", Float, nullable=True),
    # One attendance row per student per day, enforced by the database.
    UniqueConstraint("student_id", "attendance_date", name="uq_attendance_student_date"),
)

prediction_logs = Table(
    "prediction_logs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("student_id", String(64), nullable=True),
    Column("confidence", Float, nullable=True),
    Column("similarity", Float, nullable=True),
    Column("status", String(32), nullable=False),
    Column("latency_ms", Float, nullable=True),
    Column("model_student_id", String(64), nullable=True),
    Column("created_at", DateTime, nullable=False, server_default=func.now()),
)

Index("ix_attendance_date", attendance.c.attendance_date)
Index("ix_attendance_student", attendance.c.student_id)
Index("ix_prediction_logs_created", prediction_logs.c.created_at)
Index("ix_students_name", students.c.student_name)


def _today() -> str:
    return date_cls.today().isoformat()


def _now() -> datetime:
    return datetime.now()


class AttendanceDatabase:
    def __init__(self, database_url: Optional[str] = None, db_path: Optional[Path] = None):
        if database_url:
            self.database_url = database_url
        elif db_path:
            path = Path(db_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.database_url = f"sqlite:///{path.as_posix()}"
        else:
            self.database_url = get_settings().database_url

        connect_args = {}
        if self.database_url.startswith("sqlite"):
            connect_args = {"check_same_thread": False}

        self.engine: Engine = create_engine(self.database_url, future=True, connect_args=connect_args)
        self.is_sqlite = self.database_url.startswith("sqlite")
        self.create_tables()

    def create_tables(self) -> None:
        metadata.create_all(self.engine)

    def close(self) -> None:
        self.engine.dispose()

    # ------------------------------------------------------------------
    # students
    # ------------------------------------------------------------------
    def register_student(
        self,
        student_id: str,
        student_name: str,
        email: str = "",
        department: str = "",
        year: str = "",
    ) -> Dict[str, Any]:
        """Insert a student, or update the name if the id already exists."""
        with self.engine.begin() as conn:
            existing = conn.execute(
                select(students.c.student_name).where(students.c.student_id == student_id)
            ).first()
            if existing is None:
                conn.execute(
                    insert(students).values(
                        student_id=student_id,
                        student_name=student_name,
                        email=email or "",
                        department=department or "",
                        year=year or "",
                    )
                )
            elif student_name and existing[0] in ("", None, f"Student {student_id}"):
                conn.execute(
                    update(students)
                    .where(students.c.student_id == student_id)
                    .values(student_name=student_name)
                )
        return self.get_student(student_id) or {}

    def get_student(self, student_id: str) -> Optional[Dict[str, Any]]:
        with self.engine.connect() as conn:
            row = conn.execute(
                select(students).where(students.c.student_id == student_id)
            ).mappings().first()
        return dict(row) if row else None

    def get_all_students(self) -> List[Dict[str, Any]]:
        with self.engine.connect() as conn:
            rows = conn.execute(select(students).order_by(students.c.student_id)).mappings().all()
        return [dict(r) for r in rows]

    def update_student(self, student_id: str, **fields: Any) -> Optional[Dict[str, Any]]:
        allowed = {"student_name", "email", "department", "year"}
        values = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if values:
            with self.engine.begin() as conn:
                conn.execute(update(students).where(students.c.student_id == student_id).values(**values))
        return self.get_student(student_id)

    def delete_student(self, student_id: str) -> bool:
        with self.engine.begin() as conn:
            conn.execute(delete(attendance).where(attendance.c.student_id == student_id))
            conn.execute(delete(prediction_logs).where(prediction_logs.c.student_id == student_id))
            result = conn.execute(delete(students).where(students.c.student_id == student_id))
        return result.rowcount > 0

    def count_students(self) -> int:
        with self.engine.connect() as conn:
            return conn.execute(select(func.count()).select_from(students)).scalar_one()

    # ------------------------------------------------------------------
    # attendance
    # ------------------------------------------------------------------
    def mark_attendance(
        self,
        student_id: str,
        student_name: str,
        confidence: Optional[float] = None,
        status: str = "present",
        attendance_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Insert today's attendance for *student_id*.

        Returns ``{"success": bool, "already_marked": bool, ...}``. The unique
        constraint on ``(student_id, attendance_date)`` is the final authority,
        so a concurrent request cannot create a second row.
        """
        day = attendance_date or _today()
        try:
            day_value = date_cls.fromisoformat(day)
        except ValueError:
            day_value = date_cls.today()

        try:
            with self.engine.begin() as conn:
                conn.execute(
                    insert(attendance).values(
                        student_id=student_id,
                        student_name=student_name or "",
                        attendance_date=day_value,
                        marked_at=_now(),
                        status=status,
                        confidence=float(confidence) if confidence is not None else None,
                    )
                )
            return {
                "success": True,
                "already_marked": False,
                "student_id": student_id,
                "status": "present",
                "date": day_value.isoformat(),
                "time": _now().strftime("%H:%M:%S"),
            }
        except IntegrityError:
            existing = self.get_attendance_for_student_on(student_id, day_value.isoformat())
            return {
                "success": False,
                "already_marked": True,
                "student_id": student_id,
                "status": existing.get("status", "present") if existing else "present",
                "date": day_value.isoformat(),
                "time": (
                    existing.get("marked_at").strftime("%H:%M:%S")
                    if existing and existing.get("marked_at")
                    else ""
                ),
                "message": "Attendance already marked for today.",
            }

    def get_attendance_for_student_on(self, student_id: str, attendance_date: str) -> Optional[Dict[str, Any]]:
        with self.engine.connect() as conn:
            row = conn.execute(
                select(attendance).where(
                    attendance.c.student_id == student_id,
                    attendance.c.attendance_date == attendance_date,
                )
            ).mappings().first()
        return dict(row) if row else None

    def get_attendance_by_date(self, attendance_date: Optional[str] = None) -> List[Dict[str, Any]]:
        day = attendance_date or _today()
        try:
            day_value = date_cls.fromisoformat(day)
        except ValueError:
            return []
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(attendance)
                .where(attendance.c.attendance_date == day_value)
                .order_by(attendance.c.marked_at)
            ).mappings().all()
        return [self._serialise_attendance(r) for r in rows]

    def get_student_attendance(self, student_id: str, limit: int = 30) -> List[Dict[str, Any]]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(attendance)
                .where(attendance.c.student_id == student_id)
                .order_by(attendance.c.attendance_date.desc())
                .limit(limit)
            ).mappings().all()
        return [self._serialise_attendance(r) for r in rows]

    def get_attendance_history(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        student_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        query = select(attendance)
        if start_date:
            query = query.where(attendance.c.attendance_date >= date_cls.fromisoformat(start_date))
        if end_date:
            query = query.where(attendance.c.attendance_date <= date_cls.fromisoformat(end_date))
        if student_id:
            query = query.where(attendance.c.student_id == student_id)
        if status:
            query = query.where(attendance.c.status == status)
        query = query.order_by(attendance.c.attendance_date.desc(), attendance.c.marked_at.desc()).limit(limit)
        with self.engine.connect() as conn:
            rows = conn.execute(query).mappings().all()
        return [self._serialise_attendance(r) for r in rows]

    @staticmethod
    def _serialise_attendance(row: Any) -> Dict[str, Any]:
        data = dict(row)
        marked = data.get("marked_at")
        day = data.get("attendance_date")
        return {
            "id": data.get("id"),
            "student_id": data.get("student_id"),
            "student_name": data.get("student_name"),
            "date": day.isoformat() if hasattr(day, "isoformat") else day,
            "time": marked.strftime("%H:%M:%S") if hasattr(marked, "strftime") else None,
            "timestamp": marked.isoformat(sep=" ") if hasattr(marked, "isoformat") else None,
            "status": data.get("status"),
            "confidence": data.get("confidence"),
        }

    # ------------------------------------------------------------------
    # prediction logs
    # ------------------------------------------------------------------
    def log_prediction(
        self,
        student_id: Optional[str],
        confidence: Optional[float],
        status: str,
        latency_ms: Optional[float] = None,
        similarity: Optional[float] = None,
        model_student_id: Optional[str] = None,
    ) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                insert(prediction_logs).values(
                    student_id=student_id,
                    confidence=confidence,
                    similarity=similarity,
                    status=status,
                    latency_ms=latency_ms,
                    model_student_id=model_student_id,
                )
            )

    def get_prediction_stats(self) -> Dict[str, Any]:
        with self.engine.connect() as conn:
            total = conn.execute(select(func.count()).select_from(prediction_logs)).scalar_one()
            recognised = conn.execute(
                select(func.count()).select_from(prediction_logs).where(prediction_logs.c.status == "recognized")
            ).scalar_one()
            unknown = conn.execute(
                select(func.count()).select_from(prediction_logs).where(prediction_logs.c.status == "unknown")
            ).scalar_one()
            avg_latency = conn.execute(select(func.avg(prediction_logs.c.latency_ms))).scalar()
        return {
            "total_predictions": total,
            "recognized": recognised,
            "unknown": unknown,
            "avg_latency_ms": float(avg_latency) if avg_latency is not None else None,
        }

    # ------------------------------------------------------------------
    # dashboard
    # ------------------------------------------------------------------
    def get_dashboard_stats(self) -> Dict[str, Any]:
        today = _today()
        total_students = self.count_students()
        present = len(self.get_attendance_by_date(today))
        # Students are considered absent only if at least one is registered.
        absent = max(total_students - present, 0)
        percentage = round((present / total_students) * 100, 2) if total_students else 0.0

        recent = self.get_attendance_history(limit=10)
        return {
            "total_students": total_students,
            "present_today": present,
            "absent_today": absent,
            "attendance_percentage": percentage,
            "date": today,
            "recent_attendance": recent,
        }
