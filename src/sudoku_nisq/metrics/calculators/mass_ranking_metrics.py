"""
Mass-weighted ranking metrics for solution quality assessment.

This module computes ranking metrics based on probability mass rather than
unique bitstring counts, providing more intuitive sampling-based metrics.
"""

from typing import Dict, List, Any


def calculate_mass_precision_at_k(
    counts: Dict[str, int],
    validation_context: Any,
    k_values: List[int]
) -> Dict[int, float]:
    """
    Calculate mass-weighted precision: valid_mass_in_top_k / total_mass_in_top_k.
    
    Measures the fraction of probability mass that is valid within the top-k
    outcomes. Unlike count-based precision (which treats all bitstrings equally),
    this weights by how often each outcome appears.
    
    Args:
        counts: Measurement counts dictionary (bitstring -> count)
        validation_context: Object with `solution_validator(bitstring) -> bool`
        k_values: List of k thresholds to evaluate
    
    Returns:
        Dictionary mapping k -> mass_precision@k.
        Returns 0.0 if top-k set has zero mass.
    
    Edge Cases:
        - Empty counts: Returns {k: 0.0 for k in k_values}
        - k > len(counts): Clamps to available bitstrings
        - All invalid in top-k: Returns 0.0
        - All valid in top-k: Returns 1.0
    
    Interpretation:
        - Answers: "If I sample from top-k most frequent outcomes, what fraction
          of my samples will be valid?"
        - Higher is better (cleaner top-k set)
        - Monotone in neither direction (depends on distribution shape)
        - More intuitive than count-based precision for sampling scenarios
    
    Example:
        >>> counts = {"00": 500, "01": 300, "10": 150, "11": 50}
        >>> validator = lambda bs: bs in ["00", "10"]
        >>> calculate_mass_precision_at_k(counts, ctx, [1, 2, 3])
        {
            1: 1.0,      # Top 1: "00" valid, mass = 500/1000 = 0.5
            2: 0.625,    # Top 2: "00"+"01", valid_mass=500, total=800, = 0.625
            3: 0.684     # Top 3: all three, valid_mass=650, total=950, = 0.684
        }
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
        effective_k = min(k, len(sorted_bitstrings))
        
        if effective_k == 0:
            result[k] = 0.0
            continue
        
        # Calculate total and valid mass in top-k
        top_k_items = sorted_bitstrings[:effective_k]
        total_mass_top_k = sum(count for _, count in top_k_items)
        valid_mass_top_k = sum(
            count for bitstring, count in top_k_items
            if validator(bitstring)
        )
        
        if total_mass_top_k == 0:
            result[k] = 0.0
        else:
            result[k] = valid_mass_top_k / total_mass_top_k
    
    return result


def calculate_valid_mass_capture_at_k(
    counts: Dict[str, int],
    validation_context: Any,
    k_values: List[int]
) -> Dict[int, float]:
    """
    Calculate valid mass capture: valid_mass_in_top_k / total_valid_mass.
    
    Measures what fraction of all valid probability mass is contained in the
    top-k outcomes. This is equivalent to top_k_valid_mass / p_succ and tells
    you how concentrated the valid solutions are.
    
    Args:
        counts: Measurement counts dictionary (bitstring -> count)
        validation_context: Object with `solution_validator(bitstring) -> bool`
        k_values: List of k thresholds to evaluate
    
    Returns:
        Dictionary mapping k -> capture@k.
        Returns 0.0 if no valid solutions observed.
    
    Edge Cases:
        - Empty counts: Returns {k: 0.0 for k in k_values}
        - No valid solutions: Returns {k: 0.0 for k in k_values}
        - k encompasses all valid bitstrings: Returns 1.0
        - k > len(counts): Clamps to available bitstrings
    
    Interpretation:
        - Answers: "How much of the valid probability is in the top-k list?"
        - Monotone increasing in k (more coverage as k grows)
        - High capture at small k: Valid solutions are highly concentrated
        - Low capture even at large k: Valid solutions are widely dispersed
    
    Example:
        >>> counts = {"00": 500, "01": 300, "10": 150, "11": 50}
        >>> validator = lambda bs: bs in ["00", "10"]
        >>> # Total valid mass = 650/1000 = 0.65
        >>> calculate_valid_mass_capture_at_k(counts, ctx, [1, 2, 3])
        {
            1: 0.769,    # Top 1 captures 500/650 = 76.9% of valid mass
            2: 0.769,    # Top 2 adds "01" (invalid), still 500/650
            3: 1.0       # Top 3 captures all valid mass (500+150)/650 = 100%
        }
    """
    if not counts:
        return {k: 0.0 for k in k_values}
    
    total_shots = sum(counts.values())
    if total_shots == 0:
        return {k: 0.0 for k in k_values}
    
    validator = validation_context.solution_validator
    
    # Calculate total valid mass across all outcomes
    total_valid_mass = sum(
        count for bitstring, count in counts.items()
        if validator(bitstring)
    )
    
    if total_valid_mass == 0:
        return {k: 0.0 for k in k_values}
    
    # Sort bitstrings by frequency (descending, deterministic tie-break)
    sorted_bitstrings = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    
    result = {}
    
    for k in k_values:
        effective_k = min(k, len(sorted_bitstrings))
        
        # Calculate valid mass in top-k
        valid_mass_top_k = sum(
            count for bitstring, count in sorted_bitstrings[:effective_k]
            if validator(bitstring)
        )
        
        result[k] = valid_mass_top_k / total_valid_mass
    
    return result
