"""Expectation value wrapper for bitstring-based quantum algorithms.

Converts measurement count distributions into scalar expectation values
suitable for error mitigation techniques like ZNE and PEC.
"""

from typing import Dict, Callable


def compute_success_expectation(
    counts: Dict[str, int], 
    validator: Callable[[str], bool]
) -> float:
    """Compute success probability as an expectation value.
    
    Transforms a measurement count distribution into a scalar expectation
    value by computing the probability that a random measurement yields
    a valid solution. This expectation value can be used with ZNE and PEC.
    
    The expectation is defined as:
        ⟨f⟩ = Σ_b P(b) · f(b)
    where:
        - b is a bitstring measurement outcome
        - P(b) is the probability of measuring b
        - f(b) = 1 if b is valid, 0 otherwise
    
    Args:
        counts (Dict[str, int]): Measurement counts mapping bitstrings 
            to their observed frequencies.
        validator (Callable[[str], bool]): Function that returns True 
            if a bitstring represents a valid solution.
    
    Returns:
        float: Success probability in [0, 1], representing the expectation
            value of the indicator function for valid solutions.
    
    Example:
        >>> counts = {'00': 100, '01': 50, '10': 30, '11': 20}
        >>> validator = lambda b: b in ['01', '10']  # Only these are valid
        >>> compute_success_expectation(counts, validator)
        0.4  # (50 + 30) / 200 = 0.4
    """
    if not counts:
        return 0.0
    
    total_shots = sum(counts.values())
    if total_shots == 0:
        return 0.0
    
    # Compute weighted sum: each valid outcome contributes its probability
    expectation = 0.0
    for bitstring, count in counts.items():
        probability = count / total_shots
        is_valid = 1.0 if validator(bitstring) else 0.0
        expectation += probability * is_valid
    
    return expectation


def compute_bitstring_expectation(
    counts: Dict[str, int],
    target_bitstring: str
) -> float:
    """Compute probability of a specific bitstring as an expectation value.
    
    Returns P(target_bitstring), useful for mitigating individual bitstring
    probabilities when you know the desired solution.
    
    Args:
        counts (Dict[str, int]): Measurement count distribution.
        target_bitstring (str): The specific bitstring to measure probability for.
    
    Returns:
        float: Probability of measuring the target bitstring.
    
    Example:
        >>> counts = {'00': 100, '01': 50, '10': 30, '11': 20}
        >>> compute_bitstring_expectation(counts, '01')
        0.25  # 50 / 200
    """
    if not counts:
        return 0.0
    
    total_shots = sum(counts.values())
    if total_shots == 0:
        return 0.0
    
    return counts.get(target_bitstring, 0) / total_shots
