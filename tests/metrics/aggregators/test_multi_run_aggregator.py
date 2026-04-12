"""Tests for MultiRunAggregator."""

import pytest
from datetime import datetime

from sudoku_nisq.metrics.aggregators.multi_run_aggregator import MultiRunAggregator
from sudoku_nisq.metrics.data_models import MetricsResult, AggregatedMetrics


class TestMultiRunAggregator:
    """Test suite for MultiRunAggregator."""
    
    @pytest.fixture
    def sample_results(self):
        """Create sample MetricsResult objects for testing."""
        results = []
        # Create 5 results with known metric values
        for i in range(5):
            result = MetricsResult(
                # Required fields
                p_succ=0.5 + i * 0.05,  # 0.5, 0.55, 0.60, 0.65, 0.70
                p_succ_ci_lower=0.45 + i * 0.05,
                p_succ_ci_upper=0.55 + i * 0.05,
                distinct_valid_solutions=2,
                top_k_valid_mass={1: 0.8, 5: 0.95},
                precision_at_k={1: 0.9, 5: 0.85},
                recall_at_k={1: 0.85, 5: 0.95},
                # Optional fields with values
                valid_odds=2.0 + i,  # 2, 3, 4, 5, 6
                peak_ratio=1.5,
                retention_per_2q=0.95,
                log_loss_per_2q=0.5,
                shots_detect_point=1000,
                # Some fields None to test handling
                snr=None,
                eta_gate=None,
                eta_volume=None,
                eta_shot=None,
            )
            results.append(result)
        
        return results
    
    def test_aggregate_basic(self, sample_results):
        """Test basic aggregation functionality."""
        agg = MultiRunAggregator.aggregate(sample_results)
        
        assert isinstance(agg, AggregatedMetrics)
        assert agg.n_runs == 5
        assert isinstance(agg.timestamp, datetime)
    
    def test_aggregate_p_succ_stats(self, sample_results):
        """Test p_succ aggregation statistics."""
        agg = MultiRunAggregator.aggregate(sample_results)
        
        # p_succ values: 0.5, 0.55, 0.60, 0.65, 0.70
        assert agg.p_succ is not None
        assert agg.p_succ["mean"] == pytest.approx(0.60, abs=1e-10)
        assert agg.p_succ["median"] == pytest.approx(0.60, abs=1e-10)
        # Standard deviation with Bessel's correction
        import math
        expected_std = math.sqrt(sum((x - 0.6)**2 for x in [0.5, 0.55, 0.6, 0.65, 0.7]) / 4)
        assert agg.p_succ["std"] == pytest.approx(expected_std, abs=1e-6)
    
    def test_aggregate_distinct_valid_stats(self, sample_results):
        """Test distinct_valid_solutions aggregation."""
        agg = MultiRunAggregator.aggregate(sample_results)
        
        # distinct_valid_solutions values: all 2 (constant), mapped to distinct_valid in AggregatedMetrics
        assert agg.distinct_valid is not None
        assert agg.distinct_valid["mean"] == pytest.approx(2.0, abs=1e-10)
        assert agg.distinct_valid["median"] == pytest.approx(2.0, abs=1e-10)
        assert agg.distinct_valid["std"] == pytest.approx(0.0, abs=1e-10)  # No variation
    
    def test_aggregate_valid_odds_stats(self, sample_results):
        """Test valid_odds aggregation."""
        agg = MultiRunAggregator.aggregate(sample_results)
        
        # valid_odds values: 2, 3, 4, 5, 6
        assert agg.valid_odds is not None
        assert agg.valid_odds["mean"] == pytest.approx(4.0, abs=1e-10)
        assert agg.valid_odds["median"] == pytest.approx(4.0, abs=1e-10)
    
    def test_aggregate_constant_values(self):
        """Test aggregation when all values are identical."""
        results = []
        for _ in range(3):
            result = MetricsResult(
                p_succ=0.75,
                p_succ_ci_lower=0.70,
                p_succ_ci_upper=0.80,
                distinct_valid_solutions=5,
                top_k_valid_mass={1: 0.8},
                precision_at_k={1: 0.9},
                recall_at_k={1: 0.85},
            )
            results.append(result)
        
        agg = MultiRunAggregator.aggregate(results)
        
        assert agg.p_succ["mean"] == 0.75
        assert agg.p_succ["std"] == 0.0
        assert agg.p_succ["median"] == 0.75
        assert agg.p_succ["iqr"] == 0.0
    
    def test_aggregate_with_none_values(self):
        """Test handling of None values in results."""
        results = []
        for i in range(3):
            result = MetricsResult(
                p_succ=0.5 if i < 2 else None,  # Last one is None
                p_succ_ci_lower=0.45 if i < 2 else None,
                p_succ_ci_upper=0.55 if i < 2 else None,
                distinct_valid_solutions=2,
                top_k_valid_mass={1: 0.8},
                precision_at_k={1: 0.9},
                recall_at_k={1: 0.85},
                valid_odds=None,  # All None
            )
            results.append(result)
        
        agg = MultiRunAggregator.aggregate(results)
        
        # p_succ should aggregate over non-None values
        assert agg.p_succ is not None
        assert agg.p_succ["mean"] == 0.5
        
        # valid_odds all None -> result is None
        assert agg.valid_odds is None
    
    def test_aggregate_single_run(self):
        """Test aggregation with single run."""
        result = MetricsResult(
            p_succ=0.8,
            p_succ_ci_lower=0.75,
            p_succ_ci_upper=0.85,
            distinct_valid_solutions=4,
            top_k_valid_mass={1: 0.9},
            precision_at_k={1: 0.95},
            recall_at_k={1: 0.90},
        )
        
        agg = MultiRunAggregator.aggregate([result])
        
        assert agg.n_runs == 1
        assert agg.p_succ["mean"] == 0.8
        # Single value: std is 0 (or None depending on implementation)
        # IQR should be 0
    
    def test_aggregate_empty_list_raises(self):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot aggregate empty results"):
            MultiRunAggregator.aggregate([])
    
    def test_aggregate_with_notes(self, sample_results):
        """Test aggregation with custom notes."""
        agg = MultiRunAggregator.aggregate(
            sample_results, notes="Test aggregation for benchmark"
        )
        
        assert agg.aggregation_notes == "Test aggregation for benchmark"
    
    def test_aggregate_all_metric_fields(self, sample_results):
        """Test that all metric fields are included in aggregation."""
        agg = MultiRunAggregator.aggregate(sample_results)
        
        # Check that all expected fields exist (even if None)
        expected_fields = [
            "p_succ",
            "distinct_valid",
            "distinct_invalid",
            "top_k_valid_mass",
            "precision_at_k",
            "recall_at_k",
            "mass_precision_at_k",
            "valid_mass_capture_at_k",
            "valid_odds",
            "valid_odds_lower",
            "valid_odds_upper",
            "peak_ratio",
            "peak_gap",
            "retention_per_2q",
            "retention_lower",
            "retention_upper",
            "log_loss",
            "log_loss_lower",
            "log_loss_upper",
            "shots_detect_point",
            "shots_for_precision",
            "snr",
            "eta_gate",
            "eta_volume",
            "eta_shot",
        ]
        
        for field in expected_fields:
            assert hasattr(agg, field)
    
    def test_aggregate_preserves_dict_structure(self, sample_results):
        """Test that aggregated fields have expected dict structure."""
        agg = MultiRunAggregator.aggregate(sample_results)
        
        if agg.p_succ is not None:
            assert "mean" in agg.p_succ
            assert "std" in agg.p_succ
            assert "median" in agg.p_succ
            assert "q1" in agg.p_succ
            assert "q3" in agg.p_succ
            assert "iqr" in agg.p_succ
