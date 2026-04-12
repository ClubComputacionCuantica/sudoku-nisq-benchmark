"""
Ranking metrics calculators for solution quality assessment.

This module computes metrics related to the ranking of valid solutions in the
measurement distribution, helping assess whether valid solutions appear in
top-k most frequent outcomes.
"""

from typing import Dict, List, Any


def calculate_top_k_valid_mass(
    counts: Dict[str, int],
    validation_context: Any,
    k_values: List[int]
) -> Dict[int, float]:
    """
    Calculate cumulative probability mass of valid solutions in top-k outcomes.
    
    Sort all measured bitstrings by frequency (descending), then compute the
    fraction of total shots that landed on valid solutions within the top-k
    most frequent bitstrings for each k.
    
    Args:
        counts: Measurement counts dictionary (bitstring -> count)
        validation_context: Object with `solution_validator(bitstring) -> bool`
        k_values: List of k values to compute (e.g., [1, 3, 5, 10])
    
    Returns:
        Dictionary mapping k -> cumulative probability of valid solutions in top-k.
        Example: {1: 0.65, 3: 0.82, 5: 0.87} means 65% of shots in top-1, 
                 82% in top-3, etc.
    
    Edge Cases:
        - Empty counts: Returns {k: 0.0 for k in k_values}
        - k > len(counts): Clamps to total counts length
        - No valid solutions in top-k: Returns 0.0 for that k
    
    Example:
        >>> counts = {"00": 500, "01": 300, "10": 150, "11": 50}
        >>> validator = lambda bs: bs in ["01", "10"]
        >>> calculate_top_k_valid_mass(counts, ctx, [1, 2, 3])
        {1: 0.0, 2: 0.3, 3: 0.45}  # "00" invalid, "01" valid (30%), "10" adds 15%
    """
    if not counts:
        return {k: 0.0 for k in k_values}
    
    total_shots = sum(counts.values())
    if total_shots == 0:
        return {k: 0.0 for k in k_values}
    
    # Sort bitstrings by frequency (descending, deterministic tie-break)
    sorted_bitstrings = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    
    validator = validation_context.solution_validator
    result = {}
    
    for k in k_values:
        # Clamp k to available bitstrings
        effective_k = min(k, len(sorted_bitstrings))
        
        # Sum probability mass of valid solutions in top-k
        valid_mass = sum(
            count for bitstring, count in sorted_bitstrings[:effective_k]
            if validator(bitstring)
        )
        
        result[k] = valid_mass / total_shots
    
    return result


def calculate_precision_at_k(
    counts: Dict[str, int],
    validation_context: Any,
    k_values: List[int]
) -> Dict[int, float]:
    """
    Calculate precision@k: fraction of top-k bitstrings that are valid.
    
    Among the k most frequent bitstrings, what fraction are valid solutions?
    This measures how "clean" the top results are (low false positive rate).
    
    Args:
        counts: Measurement counts dictionary (bitstring -> count)
        validation_context: Object with `solution_validator(bitstring) -> bool`
        k_values: List of k values to compute (e.g., [1, 3, 5, 10])
    
    Returns:
        Dictionary mapping k -> precision (valid count / k).
        Example: {1: 1.0, 3: 0.67, 5: 0.6} means top-1 is valid, 
                 2 out of top-3 are valid, 3 out of top-5 are valid.
    
    Edge Cases:
        - Empty counts: Returns {k: 0.0 for k in k_values}
        - k > len(counts): Computes precision over available bitstrings
        - All invalid: Returns 0.0
        - All valid: Returns 1.0
    
    Example:
        >>> counts = {"00": 500, "01": 300, "10": 150, "11": 50}
        >>> validator = lambda bs: bs in ["01", "10"]
        >>> calculate_precision_at_k(counts, ctx, [1, 2, 3])
        {1: 0.0, 2: 0.5, 3: 0.67}  # 0/1, 1/2, 2/3 valid in top-k
    """
    if not counts:
        return {k: 0.0 for k in k_values}
    
    # Sort bitstrings by frequency (descending, deterministic tie-break)
    sorted_bitstrings = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    
    validator = validation_context.solution_validator
    result = {}
    
    for k in k_values:
        # Clamp k to available bitstrings
        effective_k = min(k, len(sorted_bitstrings))
        
        if effective_k == 0:
            result[k] = 0.0
            continue
        
        # Count valid solutions in top-k
        valid_count = sum(
            1 for bitstring, _ in sorted_bitstrings[:effective_k]
            if validator(bitstring)
        )
        
        result[k] = valid_count / effective_k
    
    return result


def calculate_recall_at_k(
    counts: Dict[str, int],
    validation_context: Any,
    k_values: List[int]
) -> Dict[int, float]:
    """
    Calculate recall@k: fraction of all valid solutions found in top-k.
    
    Among all valid solutions that appeared in measurement counts, what
    fraction are present in the top-k most frequent bitstrings? This measures
    how well the algorithm concentrates probability on valid solutions.
    
    Args:
        counts: Measurement counts dictionary (bitstring -> count)
        validation_context: Object with `solution_validator(bitstring) -> bool`
        k_values: List of k values to compute (e.g., [1, 3, 5, 10])
    
    Returns:
        Dictionary mapping k -> recall (valid in top-k / total valid observed).
        Example: {1: 0.50, 3: 1.0, 5: 1.0} means 1 of 2 valid solutions in top-1,
                 both valid solutions in top-3 and top-5.
    
    Edge Cases:
        - Empty counts: Returns {k: 0.0 for k in k_values}
        - No valid solutions observed: Returns {k: 0.0 for k in k_values}
        - k encompasses all valid solutions: Returns 1.0
        - k > len(counts): Clamps to available bitstrings
    
    Example:
        >>> counts = {"00": 500, "01": 300, "10": 150, "11": 50}
        >>> validator = lambda bs: bs in ["01", "10"]  # 2 valid solutions
        >>> calculate_recall_at_k(counts, ctx, [1, 2, 3])
        {1: 0.0, 2: 0.5, 3: 1.0}  # 0/2, 1/2, 2/2 valid solutions found
    """
    if not counts:
        return {k: 0.0 for k in k_values}
    
    validator = validation_context.solution_validator
    
    # Count total valid solutions observed (denominator)
    total_valid = sum(
        1 for bitstring in counts.keys()
        if validator(bitstring)
    )
    
    if total_valid == 0:
        return {k: 0.0 for k in k_values}
    
    # Sort bitstrings by frequency (descending, deterministic tie-break)
    sorted_bitstrings = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    
    result = {}
    
    for k in k_values:
        # Clamp k to available bitstrings
        effective_k = min(k, len(sorted_bitstrings))
        
        # Count valid solutions in top-k (numerator)
        valid_in_top_k = sum(
            1 for bitstring, _ in sorted_bitstrings[:effective_k]
            if validator(bitstring)
        )
        
        result[k] = valid_in_top_k / total_valid
    
    return result
