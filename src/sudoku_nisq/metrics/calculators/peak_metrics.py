"""
Peak-based discrimination metrics for solution quality assessment.

This module computes metrics based on the most probable valid and invalid
outcomes, measuring whether valid solutions stand out from the noise floor.
"""

from typing import Dict, Any, Optional


def calculate_peak_metrics(
    counts: Dict[str, int],
    validation_context: Any
) -> Dict[str, Optional[float]]:
    """
    Calculate peak discrimination metrics comparing best valid vs best invalid.
    
    Measures how much the most probable valid solution stands out from the
    most probable invalid outcome. Unlike mass-based odds, this is sensitive
    to probability distribution shape and concentration.
    
    Args:
        counts: Measurement counts dictionary (bitstring -> count)
        validation_context: Object with `solution_validator(bitstring) -> bool`
    
    Returns:
        Dictionary with keys:
        - p_best_valid: Probability of most frequent valid bitstring
        - p_best_invalid: Probability of most frequent invalid bitstring
        - peak_ratio: p_best_valid / p_best_invalid (None if infinite)
        - peak_gap: p_best_valid - p_best_invalid (absolute difference)
        - peak_ratio_is_infinite: True if no invalid solutions observed
    
    Edge Cases:
        - Empty counts: Returns all 0.0 values, is_infinite=False
        - No valid solutions: p_best_valid=0, ratio=0, gap=negative
        - No invalid solutions: ratio=None + is_infinite=True
        - All zero counts: Returns all 0.0 values
    
    Interpretation:
        - peak_ratio > 1: Best valid exceeds best invalid (good discrimination)
        - peak_ratio < 1: Best invalid exceeds best valid (poor discrimination)
        - peak_gap > 0: Positive separation between valid and invalid peaks
        - Large gap + large ratio: Strong amplitude amplification on valid solutions
    
    Example:
        >>> counts = {"00": 500, "01": 450, "10": 30, "11": 20}
        >>> validator = lambda bs: bs in ["00", "01"]
        >>> calculate_peak_metrics(counts, ctx)
        {
            'p_best_valid': 0.50,
            'p_best_invalid': 0.03,
            'peak_ratio': 16.67,
            'peak_gap': 0.47,
            'peak_ratio_is_infinite': False
        }
    """
    if not counts:
        return {
            "p_best_valid": 0.0,
            "p_best_invalid": 0.0,
            "peak_ratio": 0.0,
            "peak_gap": 0.0,
            "peak_ratio_is_infinite": False
        }
    
    total_shots = sum(counts.values())
    if total_shots == 0:
        return {
            "p_best_valid": 0.0,
            "p_best_invalid": 0.0,
            "peak_ratio": 0.0,
            "peak_gap": 0.0,
            "peak_ratio_is_infinite": False
        }
    
    validator = validation_context.solution_validator
    p_best_valid = 0.0
    p_best_invalid = 0.0
    
    for bitstring, count in counts.items():
        prob = count / total_shots
        if validator(bitstring):
            if prob > p_best_valid:
                p_best_valid = prob
        else:
            if prob > p_best_invalid:
                p_best_invalid = prob
    
    # Calculate ratio with infinity handling
    if p_best_invalid == 0.0:
        if p_best_valid > 0.0:
            peak_ratio = None  # Infinite ratio (JSON-safe)
            is_infinite = True
        else:
            peak_ratio = 0.0  # Both zero
            is_infinite = False
    else:
        peak_ratio = p_best_valid / p_best_invalid
        is_infinite = False
    
    peak_gap = p_best_valid - p_best_invalid
    
    return {
        "p_best_valid": p_best_valid,
        "p_best_invalid": p_best_invalid,
        "peak_ratio": peak_ratio,
        "peak_gap": peak_gap,
        "peak_ratio_is_infinite": is_infinite
    }
