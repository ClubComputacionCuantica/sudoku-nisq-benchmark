"""
Statistical metrics calculators for uncertainty quantification.

This module computes statistical measures of solution quality, including
confidence intervals and signal-to-noise ratios.
"""

from typing import Tuple, Dict, Any, Optional
from scipy.stats import beta
import numpy as np  # For variability_stats only


def calculate_clopper_pearson_ci(
    num_successes: int,
    num_trials: int,
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate Clopper-Pearson (exact) confidence interval for binomial proportion.
    
    Uses scipy.stats.beta.ppf to compute the exact confidence interval for
    success probability without normal approximation assumptions. This is the
    gold standard for small sample sizes or extreme probabilities.
    
    Args:
        num_successes: Number of successful outcomes (valid solution measurements)
        num_trials: Total number of trials (total shots)
        confidence_level: Desired confidence level (default 0.95 for 95% CI)
    
    Returns:
        Tuple of (lower_bound, upper_bound) for the confidence interval.
        Both values are in [0, 1].
    
    Edge Cases:
        - num_successes = 0: Returns (0.0, upper_bound) using beta distribution
        - num_successes = num_trials: Returns (lower_bound, 1.0)
        - num_trials = 0: Returns (0.0, 0.0) (undefined)
    
    References:
        Clopper, C.J., and Pearson, E.S. (1934). "The use of confidence or 
        fiducial limits illustrated in the case of the binomial."
    
    Example:
        >>> calculate_clopper_pearson_ci(875, 1000, 0.95)
        (0.8523, 0.8944)  # 95% CI for 87.5% success rate
    """
    # Input validation
    if num_successes < 0 or num_trials < 0:
        raise ValueError(f"Counts must be non-negative: successes={num_successes}, trials={num_trials}")
    if num_successes > num_trials:
        raise ValueError(f"Successes ({num_successes}) cannot exceed trials ({num_trials})")
    if not (0 < confidence_level < 1):
        raise ValueError(f"Confidence level must be in (0, 1), got {confidence_level}")
    
    if num_trials == 0:
        return (0.0, 0.0)  # Undefined proportion - return zeros for JSON compatibility
    
    alpha = 1 - confidence_level
    
    # Lower bound: quantile of Beta(successes, trials - successes + 1)
    if num_successes == 0:
        lower = 0.0
    else:
        lower = beta.ppf(alpha / 2, num_successes, num_trials - num_successes + 1)
    
    # Upper bound: quantile of Beta(successes + 1, trials - successes)
    if num_successes == num_trials:
        upper = 1.0
    else:
        upper = beta.ppf(1 - alpha / 2, num_successes + 1, num_trials - num_successes)
    
    return (lower, upper)


def calculate_snr(
    counts: Dict[str, int],
    validation_context: Any
) -> float:
    """
    **DEPRECATED**: Use calculate_valid_odds() from odds_metrics.py instead.
    
    This function computes valid_mass / invalid_mass but calls it "SNR" which
    is misleading (it's actually an odds ratio, not signal-to-noise). The name
    has been corrected to calculate_valid_odds() for statistical honesty.
    
    This function remains for backward compatibility but will be removed in v0.5.0.
    Please migrate to odds_metrics.calculate_valid_odds(p_succ).
    
    Args:
        counts: Measurement counts dictionary (bitstring -> count)
        validation_context: Object with `solution_validator(bitstring) -> bool`
    
    Returns:
        Same as valid_odds: p_succ / (1 - p_succ).
        Returns 0.0 if no valid solutions, inf if all valid.
    """
    import warnings
    warnings.warn(
        "calculate_snr() is deprecated. Use calculate_valid_odds(p_succ) from "
        "odds_metrics.py instead. This function will be removed in v0.5.0.",
        DeprecationWarning,
        stacklevel=2
    )
    
    if not counts:
        return 0.0
    
    total_shots = sum(counts.values())
    if total_shots == 0:
        return 0.0
    
    # Calculate total probability mass for valid and invalid solutions
    validator = validation_context.solution_validator
    valid_mass = sum(
        count for bitstring, count in counts.items()
        if validator(bitstring)
    ) / total_shots
    
    invalid_mass = 1.0 - valid_mass
    
    # Edge cases
    if valid_mass == 0.0:
        return 0.0  # No signal
    if invalid_mass == 0.0:
        return float('inf')  # Perfect discrimination
    
    return valid_mass / invalid_mass


def calculate_variability_stats(values: list) -> Dict[str, Optional[float]]:
    """
    Calculate variability statistics (mean, std, median, IQR) for multi-run data.
    
    Used for aggregating metrics across multiple experimental runs with the same
    configuration (backend, encoding, opt_level).
    
    Args:
        values: List of numeric values (e.g., p_succ from multiple runs)
    
    Returns:
        Dictionary with keys:
            - 'mean': Arithmetic mean
            - 'std': Sample standard deviation
            - 'median': 50th percentile
            - 'q1': 25th percentile (Q1)
            - 'q3': 75th percentile (Q3)
            - 'iqr': Interquartile range (Q3 - Q1)
    
    Edge Cases:
        - Empty list: Returns None for all metrics
        - Single value: Returns value for mean/median, 0.0 for std/IQR
        - Two values: Uses sample std (Bessel's correction)
    
    Example:
        >>> values = [0.85, 0.87, 0.82, 0.88, 0.86]
        >>> calculate_variability_stats(values)
        {'mean': 0.856, 'std': 0.023, 'median': 0.86, 'q1': 0.85, 'q3': 0.87, 'iqr': 0.02}
    """
    if not values:
        return {
            'mean': None,
            'std': None,
            'median': None,
            'q1': None,
            'q3': None,
            'iqr': None
        }
    
    arr = np.array(values)
    
    mean = np.mean(arr)
    std = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
    median = np.median(arr)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    
    return {
        'mean': float(mean),
        'std': float(std),
        'median': float(median),
        'q1': float(q1),
        'q3': float(q3),
        'iqr': float(iqr)
    }
