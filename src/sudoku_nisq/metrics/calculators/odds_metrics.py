"""
Odds-based metrics for valid solution probability ratios.

This module replaces the problematic "SNR" metric with statistically
interpretable odds ratios and provides CI transform utilities.
"""

from typing import Optional, Tuple, Callable


def calculate_valid_odds(p_succ: float) -> Optional[float]:
    """
    Calculate odds ratio for valid solutions: valid_odds = p_succ / (1 - p_succ).
    
    Measures the probability ratio of valid to invalid outcomes. This is
    mathematically equivalent to the old "SNR" but with an honest name that
    reflects its interpretation as an odds ratio, not signal-to-noise.
    
    Args:
        p_succ: Success probability (fraction of shots landing on valid solutions)
    
    Returns:
        Odds ratio = p_succ / (1 - p_succ).
        Returns 0.0 if p_succ <= 0 (no valid solutions).
        Returns None if p_succ >= 1 (infinite odds; perfect discrimination).
    
    Edge Cases:
        - p_succ <= 0: Returns 0.0 (no valid solutions observed)
        - p_succ >= 1: Returns None (infinite odds; set odds_is_infinite flag)
        - p_succ = 0.5: Returns 1.0 (equal valid/invalid probability)
    
    Interpretation:
        - odds < 1: More invalid than valid outcomes
        - odds = 1: Equal valid and invalid probability
        - odds > 1: More valid than invalid outcomes
        - odds → ∞: Approaching perfect discrimination
    
    Example:
        >>> calculate_valid_odds(p_succ=0.875)
        7.0  # 87.5% valid, 12.5% invalid, ratio = 0.875/0.125 = 7.0
        
        >>> calculate_valid_odds(p_succ=1.0)
        None  # Perfect discrimination (set odds_is_infinite: true)
    """
    if p_succ <= 0.0:
        return 0.0
    if p_succ >= 1.0:
        return None  # Infinite odds (JSON-safe; flag set separately)
    return p_succ / (1.0 - p_succ)


def transform_ci_monotone(
    ci: Tuple[Optional[float], Optional[float]],
    transform_fn: Callable[[float], float],
    increasing: bool = True
) -> Tuple[Optional[float], Optional[float]]:
    """
    Apply a monotone transform to confidence interval bounds.
    
    For monotone increasing functions, (f(lo), f(hi)) preserves order.
    For monotone decreasing functions, reverses to (f(hi), f(lo)).
    
    Args:
        ci: Confidence interval tuple (lower, upper)
        transform_fn: Monotone function to apply to each bound
        increasing: True if transform_fn is increasing, False if decreasing
    
    Returns:
        Transformed confidence interval (lower, upper).
        Returns (None, None) if input CI contains None.
    
    Example:
        >>> ci = (0.8523, 0.8944)
        >>> transform_ci_monotone(ci, lambda p: p / (1 - p), increasing=True)
        (5.77, 8.47)  # Odds ratio transform
    """
    lo, hi = ci
    if lo is None or hi is None:
        return (None, None)
    
    a, b = transform_fn(lo), transform_fn(hi)
    
    if increasing:
        return (a, b)
    else:
        return (b, a)  # Reverse for decreasing functions


def calculate_valid_odds_with_ci(
    p_succ: float,
    ci: Tuple[Optional[float], Optional[float]]
) -> Tuple[Optional[float], Tuple[Optional[float], Optional[float]], bool]:
    """
    Calculate valid odds with confidence interval propagation.
    
    Applies the odds transform to both point estimate and CI bounds.
    Handles infinite odds case with explicit flag.
    
    Args:
        p_succ: Success probability point estimate
        ci: Confidence interval for p_succ (lower, upper)
    
    Returns:
        Tuple of (odds_point, odds_ci, is_infinite):
        - odds_point: Odds ratio or None if infinite
        - odds_ci: Transformed CI or (None, None)
        - is_infinite: True if odds is infinite (p_succ >= 1)
    
    Example:
        >>> calculate_valid_odds_with_ci(0.875, (0.8523, 0.8944))
        (7.0, (5.77, 8.47), False)
        
        >>> calculate_valid_odds_with_ci(1.0, (0.95, 1.0))
        (None, (None, None), True)
    """
    odds_point = calculate_valid_odds(p_succ)
    is_infinite = (odds_point is None and p_succ > 0.0)
    
    if is_infinite:
        # Can't transform CI if point estimate is infinite
        return (None, (None, None), True)
    
    # Transform CI bounds (odds is monotone increasing in p)
    odds_ci = transform_ci_monotone(
        ci,
        lambda p: p / (1.0 - p) if 0 < p < 1 else (0.0 if p <= 0 else float('inf')),
        increasing=True
    )
    
    return (odds_point, odds_ci, is_infinite)
