"""Database behaviour: CRUD, duplicate prevention and statistics."""

from datetime import date, timedelta

import pytest

from src.data.database import AttendanceDatabase


@pytest.fixture()
def db(sandbox):
    database = AttendanceDatabase(db_path=sandbox / f"unit-{__import__('uuid').uuid4().hex}.db")
    yield database
    database.close()


def test_register_and_fetch_student(db):
    db.register_student("S0001", "Student One", "a@example.com", "CSE", "3")
    row = db.get_student("S0001")
    assert row["student_id"] == "S0001"
    assert row["student_name"] == "Student One"
    assert row["department"] == "CSE"
    assert row["count"] if "count" in row else True  # schema sanity
    assert db.get_student("NOPE") is None


def test_register_is_idempotent_and_fills_placeholder_name(db):
    db.register_student("S0002", "Student S0002")
    db.register_student("S0002", "Real Name")
    assert db.get_student("S0002")["student_name"] == "Real Name"
    assert db.count_students() == 1


def test_update_student_ignores_unknown_fields(db):
    db.register_student("S0003", "Three", department="ECE")
    updated = db.update_student("S0003", student_name="Three Renamed", bogus="x")
    assert updated["student_name"] == "Three Renamed"
    assert updated["department"] == "ECE"
    assert "bogus" not in updated


def test_mark_attendance_is_idempotent_for_same_day(db):
    db.register_student("S0004", "Four")
    first = db.mark_attendance("S0004", "Four", confidence=0.9)
    second = db.mark_attendance("S0004", "Four", confidence=0.9)

    assert first["success"] is True
    assert first["already_marked"] is False
    assert second["success"] is False
    assert second["already_marked"] is True
    assert len(db.get_attendance_by_date()) == 1


def test_mark_attendance_allows_one_row_per_day(db):
    db.register_student("S0005", "Five")
    days = [(date.today() - timedelta(days=i)).isoformat() for i in range(4)]
    for day in days:
        db.mark_attendance("S0005", "Five", attendance_date=day)
    history = db.get_student_attendance("S0005", limit=50)
    assert len(history) == 4
    assert {row["date"] for row in history} == set(days)


def test_history_filters(db):
    db.register_student("S0006", "Six")
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    db.mark_attendance("S0006", "Six", attendance_date=yesterday, status="present")
    db.mark_attendance("S0006", "Six", attendance_date=date.today().isoformat(), status="absent")

    assert len(db.get_attendance_history(start_date=yesterday)) == 2
    assert len(db.get_attendance_history(status="absent")) == 1
    assert len(db.get_attendance_history(student_id="MISSING")) == 0
    assert len(db.get_attendance_history(start_date=yesterday, end_date=yesterday)) == 1


def test_prediction_stats_and_dashboard(db):
    db.register_student("S0007", "Seven")
    db.log_prediction("S0007", 0.9, "recognized", latency_ms=12.0, similarity=0.4)
    db.log_prediction(None, 0.9, "unknown", latency_ms=8.0, similarity=0.01)

    stats = db.get_prediction_stats()
    assert stats["total_predictions"] == 2
    assert stats["recognized"] == 1
    assert stats["unknown"] == 1
    assert 0 < stats["avg_latency_ms"] <= 20

    db.mark_attendance("S0007", "Seven", confidence=0.9)
    dash = db.get_dashboard_stats()
    assert dash["total_students"] == 1
    assert dash["present_today"] == 1
    assert dash["absent_today"] == 0
    assert dash["attendance_percentage"] == 100.0
    assert dash["date"] == date.today().isoformat()
    assert len(dash["recent_attendance"]) == 1


def test_delete_student_cascades(db):
    db.register_student("S0008", "Eight")
    db.mark_attendance("S0008", "Eight")
    db.log_prediction("S0008", 0.9, "recognized")

    assert db.delete_student("S0008") is True
    assert db.get_student("S0008") is None
    assert db.get_student_attendance("S0008") == []
    assert db.get_prediction_stats()["total_predictions"] == 0
    assert db.delete_student("S0008") is False
