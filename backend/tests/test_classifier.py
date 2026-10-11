from app.lab_report_analyzer import classify_report_statuses
from tests.conftest import make_param


def _report(*params):
    return {"patient_name": "T", "report_date": "2026-01-01", "parameters": list(params)}


def test_flags_high_low_and_normal():
    report = _report(
        make_param("A", 120, 70, 100),
        make_param("B", 5, 10, 20),
        make_param("C", 15, 10, 20),
    )
    statuses = [p["status"] for p in classify_report_statuses(report)["parameters"]]
    assert statuses == ["High", "Low", "Normal"]


def test_boundaries_are_normal():
    report = _report(make_param("Low edge", 70, 70, 100), make_param("High edge", 100, 70, 100))
    statuses = [p["status"] for p in classify_report_statuses(report)["parameters"]]
    assert statuses == ["Normal", "Normal"]


def test_overrides_wrong_llm_status():
    report = _report(make_param("A", 150, 70, 100, status="Normal"))
    assert classify_report_statuses(report)["parameters"][0]["status"] == "High"


def test_input_is_not_mutated():
    report = _report(make_param("A", 150, 70, 100, status="Normal"))
    classify_report_statuses(report)
    assert report["parameters"][0]["status"] == "Normal"


def test_empty_parameters():
    assert classify_report_statuses(_report())["parameters"] == []
