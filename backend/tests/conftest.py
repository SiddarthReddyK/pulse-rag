"""Shared fixtures. Tests never call Groq or download models."""

import os

# Must be set before the app is imported so settings pick it up.
os.environ.setdefault("EMBEDDING_PROVIDER", "hashing")
os.environ.setdefault("GROQ_API_KEY", "test-key-not-used")

import pytest  # noqa: E402

from app.schemas import LabParameter, LabReport  # noqa: E402


def make_param(name, value, low, high, unit="mg/dL", status="Normal"):
    return {
        "name": name,
        "value": value,
        "unit": unit,
        "reference_low": low,
        "reference_high": high,
        "status": status,
    }


@pytest.fixture
def sample_report():
    return {
        "patient_name": "Test Patient",
        "report_date": "2026-01-15",
        "parameters": [
            make_param("Hemoglobin", 11.2, 13.0, 17.0, "g/dL"),
            make_param("Glucose", 95.0, 70.0, 100.0),
            make_param("LDL Cholesterol", 172.0, 0.0, 100.0),
        ],
    }


@pytest.fixture
def lab_report_model(sample_report):
    return LabReport(
        patient_name=sample_report["patient_name"],
        report_date=sample_report["report_date"],
        parameters=[LabParameter(**p) for p in sample_report["parameters"]],
    )
