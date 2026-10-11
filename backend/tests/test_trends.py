from app.lab_report_analyzer import analyze_trends, classify_report_statuses
from tests.conftest import make_param


def _report(date, *params):
    return classify_report_statuses(
        {"patient_name": "T", "report_date": date, "parameters": list(params)}
    )


def test_direction_and_percentage():
    older = _report("2026-01-01", make_param("Hb", 10.0, 12, 16), make_param("Glu", 100, 70, 99), make_param("TSH", 2.0, 0.5, 4.5))
    newer = _report("2026-02-01", make_param("Hb", 12.0, 12, 16), make_param("Glu", 90, 70, 99), make_param("TSH", 2.0, 0.5, 4.5))
    result = analyze_trends(older, newer)
    by_name = {t["name"]: t for t in result["trends"]}

    assert by_name["Hb"]["direction"] == "Increased"
    assert by_name["Hb"]["percentage_change"] == 20.0
    assert by_name["Hb"]["older_status"] == "Low" and by_name["Hb"]["newer_status"] == "Normal"
    assert by_name["Glu"]["direction"] == "Decreased"
    assert by_name["TSH"]["direction"] == "Unchanged"
    assert result["older_report_date"] == "2026-01-01"
    assert result["newer_report_date"] == "2026-02-01"


def test_zero_baseline_has_no_percentage():
    older = _report("2026-01-01", make_param("X", 0.0, 0, 5))
    newer = _report("2026-02-01", make_param("X", 3.0, 0, 5))
    trend = analyze_trends(older, newer)["trends"][0]
    assert trend["percentage_change"] is None
    assert trend["absolute_change"] == 3.0


def test_parameters_present_in_only_one_report():
    older = _report("2026-01-01", make_param("A", 1, 0, 5), make_param("Old", 1, 0, 5))
    newer = _report("2026-02-01", make_param("A", 2, 0, 5), make_param("New", 1, 0, 5))
    result = analyze_trends(older, newer)
    assert result["only_in_older"] == ["Old"]
    assert result["only_in_newer"] == ["New"]
    assert [t["name"] for t in result["trends"]] == ["A"]
