"""Tests for mass-weighted ranking metrics.

These complement count-based ranking tests by validating probability-mass
interpretations.
"""

import pytest


@pytest.fixture
def validation_context():
    class MockContext:
        def __init__(self):
            self.solution_validator = lambda bs: bs in {"01", "10"}

    return MockContext()


def test_mass_precision_at_k_basic(validation_context):
    from sudoku_nisq.metrics.calculators import calculate_mass_precision_at_k

    counts = {"00": 500, "01": 300, "10": 150, "11": 50}
    out = calculate_mass_precision_at_k(counts, validation_context, [1, 2, 3, 4])

    # Top-1 = "00" invalid => 0/500
    assert out[1] == 0.0
    # Top-2 = "00" + "01" => 300/800
    assert out[2] == pytest.approx(0.375)
    # Top-3 = "00" + "01" + "10" => 450/950
    assert out[3] == pytest.approx(450 / 950)


def test_valid_mass_capture_at_k_basic(validation_context):
    from sudoku_nisq.metrics.calculators import calculate_valid_mass_capture_at_k

    counts = {"00": 500, "01": 300, "10": 150, "11": 50}
    out = calculate_valid_mass_capture_at_k(counts, validation_context, [1, 2, 3, 4])

    # Total valid mass = 300 + 150 = 450
    assert out[1] == 0.0
    assert out[2] == pytest.approx(300 / 450)
    assert out[3] == 1.0
    assert out[4] == 1.0


def test_mass_metrics_tie_break_is_deterministic(validation_context):
    from sudoku_nisq.metrics.calculators import (
        calculate_mass_precision_at_k,
        calculate_valid_mass_capture_at_k,
    )

    counts = {"10": 100, "01": 100, "00": 50}

    # Under tie, "01" comes before "10" lexicographically.
    out_prec = calculate_mass_precision_at_k(counts, validation_context, [1])
    out_cap = calculate_valid_mass_capture_at_k(counts, validation_context, [1, 2])

    assert out_prec[1] == pytest.approx(1.0)
    # capture@1 should be 100/200 = 0.5 (only one of the two valid bitstrings is in top-1)
    assert out_cap[1] == pytest.approx(0.5)
    assert out_cap[2] == 1.0
