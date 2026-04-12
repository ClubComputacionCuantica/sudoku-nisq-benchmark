"""
Tests for ranking metrics calculators.

Tests cover:
- calculate_top_k_valid_mass(): Top-k valid solution probability
- calculate_precision_at_k(): Precision in top-k results
- calculate_recall_at_k(): Coverage of valid solutions in top-k
- Edge cases: empty counts, no valid solutions, k > len(counts)
"""

import pytest

from sudoku_nisq.metrics.calculators import (
    calculate_top_k_valid_mass,
    calculate_precision_at_k,
    calculate_recall_at_k
)


@pytest.fixture
def sample_counts():
    """Provide sample measurement counts."""
    return {
        "00": 500,  # Invalid
        "01": 300,  # Valid
        "10": 150,  # Valid
        "11": 50    # Invalid
    }


@pytest.fixture
def validator_01_10():
    """Validator that accepts only '01' and '10' as valid."""
    return lambda bs: bs in ["01", "10"]


@pytest.fixture
def validation_context(validator_01_10):
    """Mock ValidationContext with validator."""
    class MockContext:
        def __init__(self, validator):
            self.solution_validator = validator
            self.valid_solutions = ["01", "10"]
            self.total_valid_count = 2
    
    return MockContext(validator_01_10)


class TestCalculateTopKValidMass:
    """Tests for calculate_top_k_valid_mass function."""
    
    def test_basic_top_k(self, sample_counts, validation_context):
        """Test basic top-k valid mass calculation."""
        
        k_values = [1, 2, 3, 4]
        result = calculate_top_k_valid_mass(sample_counts, validation_context, k_values)
        
        # Top-1: "00" (500 shots, invalid) = 0%
        assert result[1] == pytest.approx(0.0)
        
        # Top-2: "00" + "01" (800 shots, 300 valid) = 30%
        assert result[2] == pytest.approx(0.3)
        
        # Top-3: "00" + "01" + "10" (950 shots, 450 valid) = 45%
        assert result[3] == pytest.approx(0.45)
        
        # Top-4: All bitstrings (1000 shots, 450 valid) = 45%
        assert result[4] == pytest.approx(0.45)
    
    def test_empty_counts(self, validation_context):
        """Test with empty counts dictionary."""
        
        result = calculate_top_k_valid_mass({}, validation_context, [1, 3, 5])
        
        assert result == {1: 0.0, 3: 0.0, 5: 0.0}
    
    def test_k_exceeds_counts_length(self, sample_counts, validation_context):
        """Test when k is larger than number of bitstrings."""
        
        k_values = [10, 20, 100]
        result = calculate_top_k_valid_mass(sample_counts, validation_context, k_values)
        
        # Should clamp to available bitstrings (4 total)
        assert result[10] == pytest.approx(0.45)
        assert result[20] == pytest.approx(0.45)
        assert result[100] == pytest.approx(0.45)
    
    def test_all_valid(self):
        """Test when all bitstrings are valid."""
        
        counts = {"00": 400, "01": 300, "10": 200, "11": 100}
        def validator(bs):
            return True  # All valid
        
        class MockContext:
            def __init__(self):
                self.solution_validator = validator
        
        result = calculate_top_k_valid_mass(counts, MockContext(), [1, 2, 3])
        
        assert result[1] == pytest.approx(0.4)   # 400/1000
        assert result[2] == pytest.approx(0.7)   # 700/1000
        assert result[3] == pytest.approx(0.9)   # 900/1000
    
    def test_no_valid_in_top_k(self):
        """Test when no valid solutions appear in top-k."""
        
        counts = {"00": 600, "11": 400}  # Both invalid
        def validator(bs):
            return bs in ["01", "10"]  # Only 01 and 10 valid
        
        class MockContext:
            def __init__(self):
                self.solution_validator = validator
        
        result = calculate_top_k_valid_mass(counts, MockContext(), [1, 2])
        
        assert result[1] == 0.0
        assert result[2] == 0.0

    def test_tie_break_is_deterministic(self):
        """Test that tied counts are ordered deterministically (lexicographic bitstring)."""

        # Tie for top count between "01" and "10"; tie-break should pick "01" first.
        counts = {"10": 100, "01": 100, "00": 50}
        def validator(bs):
            return bs == "01"

        class MockContext:
            def __init__(self):
                self.solution_validator = validator

        result = calculate_top_k_valid_mass(counts, MockContext(), [1, 2])

        # Top-1 should contain only "01" (valid), so 100/250 = 0.4
        assert result[1] == pytest.approx(0.4)
        # Top-2 contains "01" + "10"; valid mass still 100/250 = 0.4
        assert result[2] == pytest.approx(0.4)


class TestCalculatePrecisionAtK:
    """Tests for calculate_precision_at_k function."""
    
    def test_basic_precision(self, sample_counts, validation_context):
        """Test basic precision@k calculation."""
        
        k_values = [1, 2, 3, 4]
        result = calculate_precision_at_k(sample_counts, validation_context, k_values)
        
        # Top-1: ["00"] -> 0 valid / 1 = 0.0
        assert result[1] == pytest.approx(0.0)
        
        # Top-2: ["00", "01"] -> 1 valid / 2 = 0.5
        assert result[2] == pytest.approx(0.5)
        
        # Top-3: ["00", "01", "10"] -> 2 valid / 3 = 0.667
        assert result[3] == pytest.approx(0.667, abs=0.001)
        
        # Top-4: ["00", "01", "10", "11"] -> 2 valid / 4 = 0.5
        assert result[4] == pytest.approx(0.5)
    
    def test_perfect_precision(self):
        """Test when all top-k are valid."""
        
        counts = {"01": 400, "10": 300, "00": 200, "11": 100}
        def validator(bs):
            return bs in ["01", "10"]
        
        class MockContext:
            def __init__(self):
                self.solution_validator = validator
        
        result = calculate_precision_at_k(counts, MockContext(), [1, 2])
        
        assert result[1] == 1.0  # Top-1 is "01" (valid)
        assert result[2] == 1.0  # Top-2 are "01" and "10" (both valid)
    
    def test_zero_precision(self):
        """Test when no valid solutions in top-k."""
        
        counts = {"00": 500, "11": 500}
        def validator(bs):
            return bs in ["01", "10"]
        
        class MockContext:
            def __init__(self):
                self.solution_validator = validator
        
        result = calculate_precision_at_k(counts, MockContext(), [1, 2])
        
        assert result[1] == 0.0
        assert result[2] == 0.0
    
    def test_empty_counts(self, validation_context):
        """Test precision with empty counts."""
        
        result = calculate_precision_at_k({}, validation_context, [1, 3])
        
        assert result == {1: 0.0, 3: 0.0}


class TestCalculateRecallAtK:
    """Tests for calculate_recall_at_k function."""
    
    def test_basic_recall(self, sample_counts, validation_context):
        """Test basic recall@k calculation."""
        
        k_values = [1, 2, 3, 4]
        result = calculate_recall_at_k(sample_counts, validation_context, k_values)
        
        # 2 valid solutions observed: "01" and "10"
        # Top-1: ["00"] -> 0 valid found / 2 total = 0.0
        assert result[1] == pytest.approx(0.0)
        
        # Top-2: ["00", "01"] -> 1 valid found / 2 total = 0.5
        assert result[2] == pytest.approx(0.5)
        
        # Top-3: ["00", "01", "10"] -> 2 valid found / 2 total = 1.0
        assert result[3] == pytest.approx(1.0)
        
        # Top-4: All bitstrings -> 2 valid found / 2 total = 1.0
        assert result[4] == pytest.approx(1.0)
    
    def test_perfect_recall(self):
        """Test when all valid solutions appear in top-k."""
        
        counts = {"01": 400, "10": 300, "11": 200}
        def validator(bs):
            return bs in ["01", "10"]
        
        class MockContext:
            def __init__(self):
                self.solution_validator = validator
        
        result = calculate_recall_at_k(counts, MockContext(), [2, 3])
        
        assert result[2] == 1.0  # Both valid solutions in top-2
        assert result[3] == 1.0  # Both valid solutions in top-3
    
    def test_partial_recall(self):
        """Test partial recall scenario."""
        
        # 3 valid solutions observed, but dispersed
        counts = {"00": 500, "01": 250, "11": 150, "10": 100}
        def validator(bs):
            return bs in ["01", "10", "11"]
        
        class MockContext:
            def __init__(self):
                self.solution_validator = validator
        
        result = calculate_recall_at_k(counts, MockContext(), [1, 2, 3])
        
        assert result[1] == pytest.approx(0.0)   # "00" invalid
        assert result[2] == pytest.approx(1/3)   # Only "01" in top-2
        assert result[3] == pytest.approx(2/3)   # "01" and "11" in top-3
    
    def test_no_valid_solutions_observed(self):
        """Test when no valid solutions appear in counts."""
        
        counts = {"00": 600, "11": 400}
        def validator(bs):
            return bs in ["01", "10"]
        
        class MockContext:
            def __init__(self):
                self.solution_validator = validator
        
        result = calculate_recall_at_k(counts, MockContext(), [1, 2])
        
        # Denominator is 0 (no valid solutions), should return 0.0
        assert result[1] == 0.0
        assert result[2] == 0.0
    
    def test_empty_counts(self, validation_context):
        """Test recall with empty counts."""
        
        result = calculate_recall_at_k({}, validation_context, [1, 3, 5])
        
        assert result == {1: 0.0, 3: 0.0, 5: 0.0}


class TestRankingMetricsIntegration:
    """Integration tests combining multiple ranking metrics."""
    
    def test_consistency_precision_recall(self):
        """Test that precision and recall are consistent."""
        
        # Perfect ranking: all valid solutions at top
        counts = {"01": 500, "10": 400, "00": 100}
        def validator(bs):
            return bs in ["01", "10"]
        
        class MockContext:
            def __init__(self):
                self.solution_validator = validator
        
        precision = calculate_precision_at_k(counts, MockContext(), [2])
        recall = calculate_recall_at_k(counts, MockContext(), [2])
        
        # Both should be 1.0 (perfect ranking)
        assert precision[2] == 1.0
        assert recall[2] == 1.0
    
    def test_tradeoff_precision_recall(self):
        """Test precision-recall tradeoff as k increases."""
        
        # Mixed ranking
        counts = {"01": 400, "00": 300, "10": 200, "11": 100}
        def validator(bs):
            return bs in ["01", "10"]
        
        class MockContext:
            def __init__(self):
                self.solution_validator = validator
        
        k_values = [1, 2, 3]
        precision = calculate_precision_at_k(counts, MockContext(), k_values)
        recall = calculate_recall_at_k(counts, MockContext(), k_values)
        
        # Recall should increase monotonically (more valid solutions found)
        assert recall[1] <= recall[2] <= recall[3]
        
        # Precision can vary depending on ranking (not always monotonic)
        # In this case: [01 valid, 00 invalid, 10 valid, 11 invalid]
        # k=1: 1/1=1.0, k=2: 1/2=0.5, k=3: 2/3=0.67 (increases!)
        assert precision[1] == 1.0
        assert precision[2] == pytest.approx(0.5)
        assert precision[3] == pytest.approx(2/3)
