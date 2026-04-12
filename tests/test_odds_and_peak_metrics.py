"""Tests for odds- and peak-based metrics.

These cover the newer, non-deprecated replacements for the legacy `calculate_snr`.
"""

import pytest


class TestValidOdds:
    def test_valid_odds_basic(self):
        from sudoku_nisq.metrics.calculators import calculate_valid_odds

        assert calculate_valid_odds(0.5) == pytest.approx(1.0)
        assert calculate_valid_odds(0.75) == pytest.approx(3.0)

    def test_valid_odds_edge_cases(self):
        from sudoku_nisq.metrics.calculators import calculate_valid_odds

        assert calculate_valid_odds(0.0) == 0.0
        assert calculate_valid_odds(-0.1) == 0.0
        assert calculate_valid_odds(1.0) is None
        assert calculate_valid_odds(1.1) is None

    def test_valid_odds_with_ci_transforms_bounds(self):
        from sudoku_nisq.metrics.calculators import calculate_valid_odds_with_ci

        odds, odds_ci, is_inf = calculate_valid_odds_with_ci(0.75, (0.6, 0.8))

        assert is_inf is False
        assert odds == pytest.approx(3.0)
        # Odds transform is monotone increasing in p
        assert odds_ci[0] == pytest.approx(0.6 / 0.4)
        assert odds_ci[1] == pytest.approx(0.8 / 0.2)
        assert odds_ci[0] < odds < odds_ci[1]

    def test_valid_odds_with_ci_infinite(self):
        from sudoku_nisq.metrics.calculators import calculate_valid_odds_with_ci

        odds, odds_ci, is_inf = calculate_valid_odds_with_ci(1.0, (0.95, 1.0))

        assert odds is None
        assert odds_ci == (None, None)
        assert is_inf is True


class TestPeakMetrics:
    @pytest.fixture
    def validation_context(self):
        class MockContext:
            def __init__(self):
                self.solution_validator = lambda bs: bs in {"01", "10"}

        return MockContext()

    def test_peak_metrics_basic(self, validation_context):
        from sudoku_nisq.metrics.calculators import calculate_peak_metrics

        counts = {"00": 30, "01": 450, "10": 500, "11": 20}
        out = calculate_peak_metrics(counts, validation_context)

        # Best valid is 0.50 ("10"), best invalid is 0.03 ("00")
        assert out["p_best_valid"] == pytest.approx(0.5)
        assert out["p_best_invalid"] == pytest.approx(0.03)
        assert out["peak_ratio"] == pytest.approx(0.5 / 0.03)
        assert out["peak_gap"] == pytest.approx(0.47)
        assert out["peak_ratio_is_infinite"] is False

    def test_peak_metrics_all_valid_is_infinite_ratio(self, validation_context):
        from sudoku_nisq.metrics.calculators import calculate_peak_metrics

        counts = {"01": 600, "10": 400}
        out = calculate_peak_metrics(counts, validation_context)

        assert out["p_best_valid"] == pytest.approx(0.6)
        assert out["p_best_invalid"] == 0.0
        assert out["peak_ratio"] is None
        assert out["peak_ratio_is_infinite"] is True

    def test_peak_metrics_empty_counts(self, validation_context):
        from sudoku_nisq.metrics.calculators import calculate_peak_metrics

        out = calculate_peak_metrics({}, validation_context)
        assert out["p_best_valid"] == 0.0
        assert out["p_best_invalid"] == 0.0
        assert out["peak_ratio"] == 0.0
        assert out["peak_gap"] == 0.0
        assert out["peak_ratio_is_infinite"] is False
