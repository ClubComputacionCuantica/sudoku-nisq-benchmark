"""
Tests for statistical metrics calculators.

Tests cover:
- calculate_clopper_pearson_ci(): Exact binomial confidence intervals
- calculate_snr(): Signal-to-noise ratio
- calculate_variability_stats(): Multi-run aggregation statistics
- Edge cases: zero trials, extreme probabilities, empty data
"""

import pytest
import numpy as np

from sudoku_nisq.metrics.calculators import (
    calculate_clopper_pearson_ci,
    calculate_snr,
    calculate_variability_stats
)


class TestCalculateClopperPearsonCI:
    """Tests for calculate_clopper_pearson_ci function."""
    
    def test_basic_confidence_interval(self):
        """Test basic 95% CI calculation."""
        
        # 875 successes out of 1000 trials
        lower, upper = calculate_clopper_pearson_ci(875, 1000, confidence_level=0.95)
        
        # Expected CI around [0.852, 0.894]
        assert lower == pytest.approx(0.852, abs=0.005)
        assert upper == pytest.approx(0.894, abs=0.005)
        assert lower < 0.875 < upper
    
    def test_zero_successes(self):
        """Test CI when no successes observed."""
        
        lower, upper = calculate_clopper_pearson_ci(0, 100, confidence_level=0.95)
        
        # Lower bound should be 0
        assert lower == 0.0
        # Upper bound should be small but positive
        assert 0 < upper < 0.05
    
    def test_all_successes(self):
        """Test CI when all trials successful."""
        
        lower, upper = calculate_clopper_pearson_ci(100, 100, confidence_level=0.95)
        
        # Upper bound should be 1.0
        assert upper == 1.0
        # Lower bound should be high (> 0.95)
        assert lower > 0.95
    
    def test_different_confidence_levels(self):
        """Test with different confidence levels."""
        
        # Same data, different confidence levels
        ci_90 = calculate_clopper_pearson_ci(500, 1000, confidence_level=0.90)
        ci_95 = calculate_clopper_pearson_ci(500, 1000, confidence_level=0.95)
        ci_99 = calculate_clopper_pearson_ci(500, 1000, confidence_level=0.99)
        
        # Wider CI for higher confidence
        assert (ci_90[1] - ci_90[0]) < (ci_95[1] - ci_95[0])
        assert (ci_95[1] - ci_95[0]) < (ci_99[1] - ci_99[0])
    
    def test_zero_trials(self):
        """Test edge case with zero trials."""
        
        lower, upper = calculate_clopper_pearson_ci(0, 0, confidence_level=0.95)
        
        # Undefined case - should return (0, 0)
        assert lower == 0.0
        assert upper == 0.0
    
    def test_small_sample_size(self):
        """Test with very small sample size."""
        
        # 3 successes out of 5 trials
        lower, upper = calculate_clopper_pearson_ci(3, 5, confidence_level=0.95)
        
        # CI should be wide due to small sample
        assert (upper - lower) > 0.4
        assert 0 <= lower < 0.6 < upper <= 1.0
    
    def test_large_sample_size(self):
        """Test with large sample size."""
        
        # 5000 successes out of 10000 trials
        lower, upper = calculate_clopper_pearson_ci(5000, 10000, confidence_level=0.95)
        
        # CI should be narrow due to large sample
        assert (upper - lower) < 0.02
        assert lower < 0.5 < upper


class TestCalculateSNR:
    """Tests for calculate_snr function."""
    
    @pytest.fixture
    def validator_01_10(self):
        """Validator that accepts '01' and '10' as valid."""
        return lambda bs: bs in ["01", "10"]
    
    @pytest.fixture
    def validation_context(self, validator_01_10):
        """Mock ValidationContext."""
        class MockContext:
            def __init__(self, validator):
                self.solution_validator = validator
        return MockContext(validator_01_10)
    
    def test_basic_snr(self, validation_context):
        """Test basic SNR calculation."""
        
        counts = {
            "00": 30,   # Invalid - low
            "01": 450,  # Valid - high
            "10": 500,  # Valid - high
            "11": 20    # Invalid - low
        }
        
        with pytest.warns(DeprecationWarning):
            snr = calculate_snr(counts, validation_context)
        
        # With new formula: valid_mass / invalid_mass
        # Valid (01, 10): 450+500=950 → 0.95
        # Invalid (00, 11): 30+20=50 → 0.05
        # SNR = 0.95/0.05 = 19.0
        assert snr == pytest.approx(19.0)
    
    def test_perfect_snr(self, validation_context):
        """Test SNR when only valid solutions observed."""
        
        counts = {
            "01": 600,
            "10": 400
        }
        
        with pytest.warns(DeprecationWarning):
            snr = calculate_snr(counts, validation_context)
        
        # No invalid solutions - should return inf (perfect discrimination)
        assert snr == float('inf')
    
    def test_zero_snr_no_valid(self, validation_context):
        """Test SNR when no valid solutions observed."""
        
        counts = {
            "00": 600,
            "11": 400
        }
        
        with pytest.warns(DeprecationWarning):
            snr = calculate_snr(counts, validation_context)
        
        # No valid solutions - should return 0.0 (no signal)
        assert snr == 0.0
    
    def test_low_snr(self, validation_context):
        """Test low SNR (noisy results)."""
        
        counts = {
            "00": 260,
            "01": 240,
            "10": 250,
            "11": 250
        }
        
        with pytest.warns(DeprecationWarning):
            snr = calculate_snr(counts, validation_context)
        
        # With new formula: valid_mass / invalid_mass
        # Valid (01, 10): 240+250=490 → 0.49
        # Invalid (00, 11): 260+250=510 → 0.51
        # SNR = 0.49/0.51 ≈ 0.96
        assert snr == pytest.approx(0.96, rel=0.01)
    
    def test_single_invalid_solution(self, validation_context):
        """Test SNR with only one invalid solution."""
        
        counts = {
            "00": 100,
            "01": 500,
            "10": 400
        }
        
        with pytest.warns(DeprecationWarning):
            snr = calculate_snr(counts, validation_context)
        
        # With new formula: valid_mass / invalid_mass
        # Valid (01, 10): 500+400=900 → 0.9
        # Invalid (00): 100 → 0.1
        # SNR = 0.9/0.1 = 9.0
        assert snr == pytest.approx(9.0)


class TestCalculateVariabilityStats:
    """Tests for calculate_variability_stats function."""
    
    def test_basic_variability(self):
        """Test basic variability statistics."""
        
        values = [0.85, 0.87, 0.82, 0.88, 0.86]
        stats = calculate_variability_stats(values)
        
        assert stats["mean"] == pytest.approx(0.856)
        assert stats["std"] == pytest.approx(0.023, abs=0.005)
        assert stats["median"] == pytest.approx(0.86)
        assert stats["q1"] <= stats["median"] <= stats["q3"]
        assert stats["iqr"] == pytest.approx(stats["q3"] - stats["q1"])
    
    def test_single_value(self):
        """Test with single value."""
        
        stats = calculate_variability_stats([0.75])
        
        assert stats["mean"] == 0.75
        assert stats["median"] == 0.75
        assert stats["std"] == 0.0
        assert stats["iqr"] == 0.0
    
    def test_two_values(self):
        """Test with two values."""
        
        stats = calculate_variability_stats([0.7, 0.9])
        
        assert stats["mean"] == 0.8
        assert stats["median"] == 0.8
        assert stats["std"] > 0  # Sample std with Bessel's correction
    
    def test_empty_list(self):
        """Test with empty list."""
        
        calculate_variability_stats([])
        
        # Should return None for all metrics (JSON-safe)\n        assert stats["mean"] is None\n        assert stats["std"] is None\n        assert stats["median"] is None\n        assert stats["q1"] is None\n        assert stats["q3"] is None\n        assert stats["iqr"] is None
    
    def test_identical_values(self):
        """Test with all identical values."""
        
        stats = calculate_variability_stats([0.5, 0.5, 0.5, 0.5])
        
        assert stats["mean"] == 0.5
        assert stats["median"] == 0.5
        assert stats["std"] == 0.0
        assert stats["iqr"] == 0.0
    
    def test_wide_range(self):
        """Test with wide range of values."""
        
        values = [0.1, 0.3, 0.5, 0.7, 0.9]
        stats = calculate_variability_stats(values)
        
        assert stats["mean"] == 0.5
        assert stats["median"] == 0.5
        assert stats["std"] > 0.25  # High variability
        assert stats["iqr"] > 0.3   # Wide IQR
    
    def test_large_sample(self):
        """Test with large sample size."""
        
        # 100 values from normal distribution
        np.random.seed(42)
        values = np.random.normal(loc=0.7, scale=0.1, size=100).tolist()
        
        stats = calculate_variability_stats(values)
        
        assert 0.65 < stats["mean"] < 0.75
        assert 0.05 < stats["std"] < 0.15
        assert stats["median"] is not None
        assert stats["iqr"] > 0


class TestStatisticalMetricsIntegration:
    """Integration tests for statistical metrics."""
    
    def test_ci_contains_true_probability(self):
        """Test that confidence interval contains true probability."""
        
        # Simulate: true p=0.7, observe 700/1000
        true_p = 0.7
        lower, upper = calculate_clopper_pearson_ci(700, 1000, confidence_level=0.95)
        
        # 95% CI should contain true probability
        assert lower < true_p < upper
    
    def test_aggregation_of_ci_bounds(self):
        """Test aggregating CI bounds across multiple runs."""
        
        # Multiple runs with different success rates
        runs = [(700, 1000), (720, 1000), (680, 1000), (710, 1000)]
        
        lower_bounds = []
        upper_bounds = []
        
        for successes, trials in runs:
            lower, upper = calculate_clopper_pearson_ci(successes, trials)
            lower_bounds.append(lower)
            upper_bounds.append(upper)
        
        # Aggregate the bounds
        lower_stats = calculate_variability_stats(lower_bounds)
        upper_stats = calculate_variability_stats(upper_bounds)
        
        assert lower_stats["mean"] < upper_stats["mean"]
        assert lower_stats["std"] >= 0
        assert upper_stats["std"] >= 0
    
    def test_snr_correlation_with_p_succ(self):
        """Test that high SNR correlates with high p_succ."""
        
        def validator(bs):
            return bs in ["01", "10"]
        
        class MockContext:
            def __init__(self):
                self.solution_validator = validator
        
        # High p_succ scenario
        high_counts = {"01": 900, "10": 50, "00": 30, "11": 20}
        with pytest.warns(DeprecationWarning):
            high_snr = calculate_snr(high_counts, MockContext())
        
        # Low p_succ scenario
        low_counts = {"01": 100, "10": 50, "00": 450, "11": 400}
        with pytest.warns(DeprecationWarning):
            low_snr = calculate_snr(low_counts, MockContext())
        
        # High p_succ should have higher SNR
        assert high_snr > low_snr
