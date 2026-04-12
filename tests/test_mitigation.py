"""Tests for error mitigation integration with exact cover solver."""

import pytest
from sudoku_nisq.mitigation.expectation_wrapper import (
    compute_success_expectation,
    compute_bitstring_expectation
)

pytestmark = pytest.mark.unit


class TestExpectationWrapper:
    """Test expectation value computation from measurement counts."""
    
    def test_compute_success_expectation_basic(self):
        """Test basic success probability calculation."""
        counts = {'00': 100, '01': 50, '10': 30, '11': 20}
        def validator(b):
            return b in ['01', '10']  # Only these are valid
        
        result = compute_success_expectation(counts, validator)
        expected = (50 + 30) / 200  # 0.4
        assert abs(result - expected) < 1e-9
    
    def test_compute_success_expectation_all_valid(self):
        """Test when all outcomes are valid."""
        counts = {'00': 50, '01': 50}
        def validator(b):
            return True  # All valid
        
        result = compute_success_expectation(counts, validator)
        assert abs(result - 1.0) < 1e-9
    
    def test_compute_success_expectation_none_valid(self):
        """Test when no outcomes are valid."""
        counts = {'00': 50, '01': 50}
        def validator(b):
            return False  # None valid
        
        result = compute_success_expectation(counts, validator)
        assert abs(result - 0.0) < 1e-9
    
    def test_compute_success_expectation_empty_counts(self):
        """Test with empty counts dictionary."""
        counts = {}
        def validator(b):
            return True
        
        result = compute_success_expectation(counts, validator)
        assert result == 0.0
    
    def test_compute_bitstring_expectation(self):
        """Test probability of specific bitstring."""
        counts = {'00': 100, '01': 50, '10': 30, '11': 20}
        
        result = compute_bitstring_expectation(counts, '01')
        expected = 50 / 200
        assert abs(result - expected) < 1e-9
    
    def test_compute_bitstring_expectation_missing(self):
        """Test bitstring not in counts."""
        counts = {'00': 100, '01': 50}
        
        result = compute_bitstring_expectation(counts, '11')
        assert result == 0.0


class TestExactCoverValidation:
    """Test exact cover solution validation logic."""
    
    def test_validation_logic_concept(self):
        """Conceptual test of exact cover validation.
        
        This validates the logic used in ExactCoverQuantumSolver._is_valid_solution
        without requiring full solver instantiation.
        """
        # Example exact cover problem:
        # Universe: [0, 1, 2, 3]
        # Subsets: S_0={0,1}, S_1={2,3}, S_2={0,2}, S_3={1,3}
        # Valid solution: S_0 and S_1 (bitstring "11")
        
        universe = [0, 1, 2, 3]
        subsets = {
            'S_0': [0, 1],
            'S_1': [2, 3],
            'S_2': [0, 2],
            'S_3': [1, 3],
        }
        
        def validate(bitstring: str) -> bool:
            """Mock validation logic."""
            selected_indices = [i for i, bit in enumerate(bitstring) if bit == '1']
            covered_elements = []
            for idx in selected_indices:
                subset_key = f'S_{idx}'
                if subset_key in subsets:
                    covered_elements.extend(subsets[subset_key])
            
            return (len(covered_elements) == len(set(covered_elements)) and 
                    set(covered_elements) == set(universe))
        
        # Valid solutions
        assert validate('1100') is True  # S_0 and S_1
        assert validate('0011') is True  # S_2 and S_3
        
        # Invalid solutions (overlap or incomplete coverage)
        assert validate('1010') is False  # S_0 and S_2 (overlap at 0)
        assert validate('1000') is False  # Only S_0 (incomplete)
        assert validate('0000') is False  # No subsets selected


class TestMitigationImports:
    """Test that mitigation module imports work correctly."""
    
    def test_import_mitigation_module(self):
        """Test importing the mitigation module."""
        from sudoku_nisq import mitigation
        assert hasattr(mitigation, 'compute_success_expectation')
        assert hasattr(mitigation, 'create_zne_executor')
        assert hasattr(mitigation, 'create_pec_executor')
    
    def test_mitiq_availability(self):
        """Test Mitiq availability detection."""
        from sudoku_nisq.mitigation import executors
        # Should not raise, just checks if module variable exists
        assert hasattr(executors, 'MITIQ_AVAILABLE')
        assert isinstance(executors.MITIQ_AVAILABLE, bool)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
