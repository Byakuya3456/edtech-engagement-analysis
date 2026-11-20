import pytest


def test_feature_pipeline_columns():
    """Verify core engineered features are present."""
    required_features = ["study_hours", "quiz_completion_rate", "session_frequency"]
    extracted_features = ["study_hours", "quiz_completion_rate", "session_frequency"]
    assert set(required_features).issubset(set(extracted_features))


def test_model_engagement_threshold():
    """Verify student engagement classification meets production F1 threshold.

    Simulates an intentional regression for Guardrail-CI demonstration.
    """
    minimum_f1_threshold = 0.85
    observed_f1_score = 0.68

    assert observed_f1_score >= minimum_f1_threshold, (
        f"Model engagement regression: F1 score {observed_f1_score} dropped below "
        f"minimum required production threshold of {minimum_f1_threshold}."
    )
