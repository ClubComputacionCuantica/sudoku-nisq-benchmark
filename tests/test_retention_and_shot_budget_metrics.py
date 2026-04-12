"""Tests for retention-based normalization and shot budget metrics."""

import pytest


class TestRetentionMetrics:
    def test_log_loss_per_2q_monotone(self):
        from sudoku_nisq.metrics.calculators import calculate_log_loss_per_2q

        # Higher p_succ => lower loss
        loss_hi = calculate_log_loss_per_2q(p_succ=0.9, two_qubit_gates=100)
        loss_lo = calculate_log_loss_per_2q(p_succ=0.1, two_qubit_gates=100)

        assert loss_hi is not None and loss_lo is not None
        assert loss_hi < loss_lo
        assert loss_hi >= 0.0

    def test_retention_per_2q_monotone(self):
        from sudoku_nisq.metrics.calculators import calculate_retention_per_2q

        # Higher p_succ => higher retention
        r_hi = calculate_retention_per_2q(p_succ=0.9, two_qubit_gates=100)
        r_lo = calculate_retention_per_2q(p_succ=0.1, two_qubit_gates=100)

        assert r_hi is not None and r_lo is not None
        assert 0.0 < r_lo < r_hi <= 1.0

    def test_log_loss_with_ci_reverses_bounds(self):
        from sudoku_nisq.metrics.calculators import calculate_log_loss_with_ci

        loss, (lo, hi) = calculate_log_loss_with_ci(0.75, (0.6, 0.8), denominator=10)
        assert loss is not None
        assert lo is not None and hi is not None
        # Because log-loss decreases with p, CI should be ordered lo < hi in loss-space
        assert lo < loss < hi

    def test_retention_with_ci_preserves_bounds(self):
        from sudoku_nisq.metrics.calculators import calculate_retention_with_ci

        retention, (lo, hi) = calculate_retention_with_ci(0.75, (0.6, 0.8), denominator=10)
        assert retention is not None
        assert lo is not None and hi is not None
        assert lo < retention < hi


class TestShotBudgetMetrics:
    def test_shots_to_detect_basic(self):
        from sudoku_nisq.metrics.calculators import shots_to_detect

        # With p=0.5 and reliability=0.99, N = ceil(log(0.01)/log(0.5)) = 7
        assert shots_to_detect(0.5, reliability=0.99) == 7

    def test_shots_to_detect_edge_cases(self):
        from sudoku_nisq.metrics.calculators import shots_to_detect

        assert shots_to_detect(1.0, reliability=0.95) == 1
        assert shots_to_detect(0.0, reliability=0.95) is None

        with pytest.raises(ValueError):
            shots_to_detect(0.5, reliability=1.0)

    def test_calculate_shot_budgets_orders_pessimistic_optimistic(self):
        from sudoku_nisq.metrics.calculators import calculate_shot_budgets

        out = calculate_shot_budgets(0.2, (0.1, 0.3), reliability=0.95)

        assert out["shots_detect_point"] is not None
        assert out["shots_detect_pessimistic"] is not None
        assert out["shots_detect_optimistic"] is not None

        # Lower p -> more required shots
        assert out["shots_detect_pessimistic"] >= out["shots_detect_point"] >= out["shots_detect_optimistic"]
        assert out["reliability"] == pytest.approx(0.95)
