"""Unit tests for Student Pass/Fail Predictor."""

import pytest
from app import predict_result, validate_values


def test_valid_pass_prediction():
    result, confidence, explanation = predict_result(
        study_hours=7.0, attendance=85.0, previous_score=78.0, assignments=10
    )
    assert result == "Pass"
    assert confidence > 50.0


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
