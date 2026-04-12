"""
Tests for success metrics calculators.

Tests cover:
- calculate_p_succ(): Success probability
- calculate_distinct_solutions(): Coverage of valid solutions
"""

import pytest
from typing import Dict

class TestSuccessMetrics:
    """Tests for SuccessMetricsCalculator."""
    
    @pytest.fixture
    def sample_counts(self) -> Dict[str, int]:
        """Provide sample measurement counts."""
        return {
            "00": 500,  # Invalid
            "01": 300,  # Valid
            "10": 150,  # Valid
            "11": 50    # Invalid
        }
    
    @pytest.fixture
    def validator(self):
        """Validator that accepts '01' and '10'."""
        return lambda bs: bs in ["01", "10"]

    def test_calculate_p_succ_basic(self, sample_counts, validator):
        """Test basic success probability calculation."""
        from sudoku_nisq.metrics.calculators import calculate_p_succ
        
        # Total shots = 1000
        # Valid shots = 300 + 150 = 450
        # p_succ = 450 / 1000 = 0.45
        
        result = calculate_p_succ(sample_counts, validator)
        assert result == pytest.approx(0.45)
        
    def test_calculate_p_succ_empty(self, validator):
        """Test p_succ with empty counts."""
        from sudoku_nisq.metrics.calculators import calculate_p_succ
        
        assert calculate_p_succ({}, validator) == 0.0
        
    def test_calculate_p_succ_no_valid(self, sample_counts):
        """Test p_succ when no shots are valid."""
        from sudoku_nisq.metrics.calculators import calculate_p_succ
        
        # Validator accepts nothing
        def validator(bs):
            return False
        
        result = calculate_p_succ(sample_counts, validator)
        assert result == 0.0
        
    def test_calculate_distinct_solutions(self, sample_counts, validator):
        """Test distinct solution counting."""
        from sudoku_nisq.metrics.calculators import calculate_distinct_solutions
        
        # Valid solutions in counts: "01" (300), "10" (150) -> 2 distinct
        result = calculate_distinct_solutions(sample_counts, validator)
        assert result == 2
        
    def test_distinct_solutions_some_missing(self, validator):
        """Test when some valid solutions are not observed."""
        from sudoku_nisq.metrics.calculators import calculate_distinct_solutions
        
        # "10" is valid but missing
        counts = {"00": 500, "01": 500}
        
        result = calculate_distinct_solutions(counts, validator)
        assert result == 1
