"""
Success metrics calculator.

Computes basic success probability and coverage metrics from measurement data.
"""

from typing import Dict, Callable


class SuccessMetricsCalculator:
    """Calculate basic success probability and coverage metrics.
    
    These are the most fundamental metrics for evaluating quantum algorithm
    correctness (Section 1 of the benchmarking metrics design).
    """
    
    @staticmethod
    def calculate_p_succ(
        counts: Dict[str, int],
        validator: Callable[[str], bool]
    ) -> float:
        """Calculate success probability (p_succ).
        
        Success probability is the fraction of measured bitstrings that 
        correspond to valid solutions. This is the primary outcome variable
        for quantum algorithm evaluation.
        
        Args:
            counts: Measurement counts dictionary mapping bitstring to count
            validator: Function that returns True if bitstring is a valid solution
            
        Returns:
            Success probability in range [0, 1]
            
        Examples:
            >>> counts = {'00': 500, '11': 500}
            >>> validator = lambda bs: bs == '11'
            >>> SuccessMetricsCalculator.calculate_p_succ(counts, validator)
            0.5
            
            >>> # All invalid
            >>> validator = lambda bs: False
            >>> SuccessMetricsCalculator.calculate_p_succ(counts, validator)
            0.0
        """
        total_shots = sum(counts.values())
        if total_shots == 0:
            return 0.0
        
        valid_shots = sum(
            count for bitstring, count in counts.items()
            if validator(bitstring)
        )
        
        return valid_shots / total_shots
    
    @staticmethod
    def calculate_distinct_solutions(
        counts: Dict[str, int],
        validator: Callable[[str], bool]
    ) -> int:
        """Count number of distinct valid solutions observed (coverage).
        
        Coverage reveals whether the quantum distribution is multimodal or
        collapsed due to noise. Important for search algorithms that may
        have multiple valid solutions.
        
        Args:
            counts: Measurement counts dictionary
            validator: Function that returns True if bitstring is valid
            
        Returns:
            Number of unique valid bitstrings with count > 0
            
        Examples:
            >>> counts = {'00': 10, '11': 20, '01': 5}
            >>> validator = lambda bs: bs in ['00', '11']
            >>> SuccessMetricsCalculator.calculate_distinct_solutions(counts, validator)
            2
            
            >>> # Only one valid solution observed
            >>> validator = lambda bs: bs == '11'
            >>> SuccessMetricsCalculator.calculate_distinct_solutions(counts, validator)
            1
        """
        distinct_count = sum(
            1 for bitstring, count in counts.items()
            if count > 0 and validator(bitstring)
        )
        
        return distinct_count
    
    @staticmethod
    def count_valid_shots(
        counts: Dict[str, int],
        validator: Callable[[str], bool]
    ) -> int:
        """Count total number of valid shots (helper for CI calculation).
        
        Args:
            counts: Measurement counts dictionary
            validator: Function that returns True if bitstring is valid
            
        Returns:
            Total number of shots that produced valid solutions
        """
        return sum(
            count for bitstring, count in counts.items()
            if validator(bitstring)
        )


# Module-level convenience functions for Phase 4 integration
def calculate_p_succ(counts: Dict[str, int], validator: Callable[[str], bool]) -> float:
    """Module-level wrapper for success probability calculation.
    
    See SuccessMetricsCalculator.calculate_p_succ for documentation.
    """
    return SuccessMetricsCalculator.calculate_p_succ(counts, validator)


def calculate_distinct_solutions(counts: Dict[str, int], validator: Callable[[str], bool]) -> int:
    """Module-level wrapper for distinct solutions counting.
    
    See SuccessMetricsCalculator.calculate_distinct_solutions for documentation.
    """
    return SuccessMetricsCalculator.calculate_distinct_solutions(counts, validator)

