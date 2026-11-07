"""Unit tests for Student Pass/Fail Predictor."""

import pytest
import app
from app import predict_result, validate_values


def test_valid_pass_prediction():
    result, confidence, explanation = predict_result(
        study_hours=7.0, attendance=85.0, previous_score=78.0, assignments=10
    )
    assert result == "Pass"
    assert confidence > 50.0


def test_prediction_trains_model_lazily_when_no_model_is_injected(monkeypatch):
    class StubModel:
        def predict(self, _sample):
            return ["Pass"]

        def predict_proba(self, _sample):
            return [[0.2, 0.8]]

    calls = []
    model = StubModel()

    def train_stub():
        calls.append(True)
        return model, 0.8, None

    monkeypatch.setattr(app, "train_model", train_stub)

    result, confidence, _ = predict_result(7.0, 85.0, 78.0, 10)

    assert result == "Pass"
    assert confidence == 80.0
    assert len(calls) == 1


def test_academic_minimum_attendance_rule():
    # Student with less than 75% attendance must fail
    result, confidence, explanation = predict_result(
        study_hours=8.0, attendance=70.0, previous_score=90.0, assignments=15
    )
    assert result == "Fail"
    assert "Attendance" in explanation


def test_academic_minimum_study_hours_rule():
    # Student with less than 2 hours daily study must fail
    result, confidence, explanation = predict_result(
        study_hours=1.0, attendance=95.0, previous_score=85.0, assignments=15
    )
    assert result == "Fail"
    assert "study time" in explanation.lower()


def test_input_validation_errors():
    with pytest.raises(ValueError):
        validate_values(study_hours=30.0, attendance=80.0, previous_score=80.0, assignments=5)

    with pytest.raises(ValueError):
        validate_values(study_hours=5.0, attendance=150.0, previous_score=80.0, assignments=5)


@pytest.mark.parametrize("field, value", [("study_hours", float("nan")), ("attendance", float("inf"))])
def test_input_validation_rejects_non_finite_values(field, value):
    inputs = {"study_hours": 5.0, "attendance": 80.0, "previous_score": 80.0, "assignments": 5}
    inputs[field] = value

    with pytest.raises(ValueError, match="finite"):
        validate_values(**inputs)


def test_input_validation_rejects_fractional_assignments():
    with pytest.raises(ValueError, match="whole number"):
        validate_values(study_hours=5.0, attendance=80.0, previous_score=80.0, assignments=4.5)


def test_interactive_cli_reports_invalid_numeric_input(monkeypatch, capsys):
    monkeypatch.setattr(app.sys, "argv", ["app.py"])
    monkeypatch.setattr("builtins.input", lambda _prompt: "five")

    app.run_cli()

    assert "Input Error" in capsys.readouterr().out
