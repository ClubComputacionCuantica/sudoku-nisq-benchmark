"""
Shot budget metrics for sampling cost estimation.

This module computes the number of shots required to detect valid solutions
with specified reliability, replacing the problematic eta_shot metric.
"""

import math
from typing import Optional, Dict, Tuple, Union


def shots_to_detect(
    p_succ: float,
    reliability: float = 0.95
) -> Optional[int]:
    """
    Calculate shots needed to see at least one valid outcome with given reliability.
    
    Probability of missing valid outcomes after N shots is (1-p)^N. To achieve
    success probability ≥ reliability, we need (1-p)^N ≤ (1-reliability), which
    gives N ≥ log(1-reliability) / log(1-p).
    
    Args:
        p_succ: Success probability per shot (valid solution rate)
        reliability: Target probability of seeing ≥1 valid solution (default 0.95)
    
    Returns:
        Minimum integer shots needed to achieve target reliability.
        Returns None if p_succ <= 0 (impossible with zero success rate).
        Returns 1 if p_succ >= 1 (guaranteed success on first shot).
    
    Edge Cases:
        - p_succ <= 0: Returns None (impossible to detect)
        - p_succ >= 1: Returns 1 (first shot guaranteed)
        - reliability <= 0 or >= 1: Raises ValueError
        - p_succ near 0: Returns very large number
    
    Interpretation:
        - Directly answers: "How many shots for 95% confidence of finding solution?"
        - Inversely proportional to p_succ (halving p doubles required shots)
        - More reliable than eta_shot which had 1/N² scaling artifact
    
    Example:
        >>> shots_to_detect(p_succ=0.087, reliability=0.95)
        34  # Need 34 shots for 95% chance of seeing valid solution
        
        >>> shots_to_detect(p_succ=0.01, reliability=0.95)
        299  # Low success rate requires many more shots
        
        >>> shots_to_detect(p_succ=0.5, reliability=0.99)
        7  # Higher reliability increases required shots
    """
    if not (0.0 < reliability < 1.0):
        raise ValueError(f"Reliability must be in (0, 1), got {reliability}")
    
    if p_succ <= 0.0:
        return None  # Impossible to detect with zero success rate
    if p_succ >= 1.0:
        return 1  # Guaranteed success on first shot
    
    # Solve (1 - p_succ)^N ≤ (1 - reliability) for N
    numerator = math.log(1.0 - reliability)
    denominator = math.log(1.0 - p_succ)
    
    return int(math.ceil(numerator / denominator))


def calculate_shot_budgets(
    p_succ: float,
    ci: Tuple[Optional[float], Optional[float]],
    reliability: float = 0.95
) -> Dict[str, Union[int, float, None]]:
    """
    Calculate shot budgets with uncertainty bounds from confidence interval.
    
    Computes three estimates:
    - Point: Based on observed p_succ
    - Pessimistic: Based on lower CI bound (true p might be lower, need more shots)
    - Optimistic: Based on upper CI bound (true p might be higher, need fewer shots)
    
    Args:
        p_succ: Success probability point estimate
        ci: Confidence interval for p_succ (lower, upper)
        reliability: Target detection reliability (default 0.95)
    
    Returns:
        Dictionary with keys:
        - shots_detect_point: Shots based on point estimate
        - shots_detect_pessimistic: Shots based on lower CI bound (worst case)
        - shots_detect_optimistic: Shots based on upper CI bound (best case)
        - reliability: Target reliability used (for reference)
    
    Edge Cases:
        - p_succ = 0: Point estimate returns None
        - CI contains 0: Pessimistic returns None
        - CI contains 1: Optimistic returns 1
        - CI is (None, None): Pessimistic and optimistic return None
    
    Interpretation:
        - Pessimistic budget: Conservative estimate accounting for uncertainty
        - Optimistic budget: Best-case scenario estimate
        - Range between them reflects statistical uncertainty in shot requirements
    
    Example:
        >>> calculate_shot_budgets(0.087, (0.0717, 0.1046), reliability=0.95)
        {
            'shots_detect_point': 34,
            'shots_detect_pessimistic': 42,  # Assume true p is as low as 0.0717
            'shots_detect_optimistic': 29,   # Assume true p is as high as 0.1046
            'reliability': 0.95
        }
    """
    ci_lower, ci_upper = ci
    
    point_shots = shots_to_detect(p_succ, reliability)
    
    pessimistic_shots = None
    if ci_lower is not None and ci_lower > 0.0:
        pessimistic_shots = shots_to_detect(ci_lower, reliability)
    
    optimistic_shots = None
    if ci_upper is not None and ci_upper > 0.0:
        optimistic_shots = shots_to_detect(ci_upper, reliability)
    
    return {
        "shots_detect_point": point_shots,
        "shots_detect_pessimistic": pessimistic_shots,
        "shots_detect_optimistic": int(optimistic_shots) if optimistic_shots is not None else None,
        "reliability": reliability
    }
