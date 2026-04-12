"""
Retention-based normalization metrics for hardware resource cost.

This module computes log-loss and geometric mean retention factors that
properly account for multiplicative error accumulation in quantum circuits.
"""

import math
from typing import Optional, Tuple


def calculate_log_loss_per_2q(
    p_succ: float,
    two_qubit_gates: int,
    eps: float = 1e-12
) -> Optional[float]:
    """
    Calculate per-2-qubit-gate log loss: -log(p_succ) / n_gates.
    
    Measures average information loss per entangling gate. This properly
    reflects multiplicative error accumulation (gates compound exponentially)
    unlike the linear p_succ/n_gates ratio. Smaller values indicate better
    noise resilience.
    
    Args:
        p_succ: Success probability (fraction of shots landing on valid solutions)
        two_qubit_gates: Total count of two-qubit gates (CNOT, CZ, etc.)
        eps: Small constant to prevent log(0) (default 1e-12)
    
    Returns:
        Log loss per gate = -log(max(p_succ, eps)) / n_gates.
        Returns None if two_qubit_gates <= 0 (undefined for trivial circuit).
    
    Edge Cases:
        - two_qubit_gates <= 0: Returns None (no entangling gates)
        - p_succ = 0: Uses eps to avoid -inf, yields large positive loss
        - p_succ = 1: Returns 0.0 (perfect success, zero loss)
    
    Interpretation:
        - Smaller is better (less information loss per gate)
        - Enables fair comparison across problem sizes
        - Monotone decreasing in p_succ (unlike eta_gate near p=0)
        - Natural units: nats per gate (use log10 for bits per gate)
    
    Example:
        >>> calculate_log_loss_per_2q(p_succ=0.75, two_qubit_gates=60)
        0.00479  # ~0.5% loss per gate
        
        >>> calculate_log_loss_per_2q(p_succ=0.01, two_qubit_gates=100)
        0.0461  # ~4.6% loss per gate (worse algorithm)
    """
    if two_qubit_gates <= 0:
        return None
    
    p_clamped = min(max(p_succ, eps), 1.0)
    return -math.log(p_clamped) / two_qubit_gates


def calculate_retention_per_2q(
    p_succ: float,
    two_qubit_gates: int,
    eps: float = 1e-12
) -> Optional[float]:
    """
    Calculate per-2-qubit-gate geometric mean retention: p_succ^(1/n_gates).
    
    Measures the effective success probability "per gate" under multiplicative
    error model. Closer to 1.0 is better. This is the exponential inverse of
    log_loss and provides an intuitive "percentage retention per gate" metric.
    
    Args:
        p_succ: Success probability (fraction of shots landing on valid solutions)
        two_qubit_gates: Total count of two-qubit gates (CNOT, CZ, etc.)
        eps: Small constant to prevent log(0) (default 1e-12)
    
    Returns:
        Retention per gate = p_succ^(1/n_gates).
        Returns None if two_qubit_gates <= 0 (undefined for trivial circuit).
    
    Edge Cases:
        - two_qubit_gates <= 0: Returns None (no entangling gates)
        - p_succ = 0: Uses eps to avoid 0^0, yields ~0 retention
        - p_succ = 1: Returns 1.0 (perfect retention per gate)
    
    Interpretation:
        - Closer to 1.0 is better (higher per-gate success retention)
        - 0.99 means ~1% fidelity loss per gate
        - Enables intuitive comparison: "Algorithm A retains 98.5% per gate"
        - Monotone increasing in p_succ (unlike eta_gate)
    
    Example:
        >>> calculate_retention_per_2q(p_succ=0.75, two_qubit_gates=60)
        0.9952  # ~99.5% retention per gate
        
        >>> calculate_retention_per_2q(p_succ=0.01, two_qubit_gates=100)
        0.9549  # ~95.5% retention per gate (worse algorithm)
    """
    if two_qubit_gates <= 0:
        return None
    
    p_clamped = min(max(p_succ, eps), 1.0)
    return p_clamped ** (1.0 / two_qubit_gates)


def calculate_log_loss_per_volume(
    p_succ: float,
    circuit_volume: Optional[int],
    eps: float = 1e-12
) -> Optional[float]:
    """
    Calculate per-volume log loss: -log(p_succ) / circuit_volume.
    
    Similar to log_loss_per_2q but normalized by circuit volume (gate×depth),
    which correlates with coherence time usage. Aligns with Volumetric
    Benchmarking standards.
    
    Args:
        p_succ: Success probability
        circuit_volume: Total circuit volume (gate count × depth)
        eps: Small constant to prevent log(0)
    
    Returns:
        Log loss per volume unit or None if volume unavailable/zero.
    
    Edge Cases:
        - circuit_volume is None: Returns None (SDK doesn't provide volume)
        - circuit_volume <= 0: Returns None (trivial circuit)
        - p_succ = 0: Uses eps to avoid -inf
        - p_succ = 1: Returns 0.0
    
    Example:
        >>> calculate_log_loss_per_volume(p_succ=0.75, circuit_volume=1470)
        0.000196  # ~0.02% loss per volume unit
    """
    if circuit_volume is None or circuit_volume <= 0:
        return None
    
    p_clamped = min(max(p_succ, eps), 1.0)
    return -math.log(p_clamped) / circuit_volume


def calculate_retention_per_volume(
    p_succ: float,
    circuit_volume: Optional[int],
    eps: float = 1e-12
) -> Optional[float]:
    """
    Calculate per-volume geometric mean retention: p_succ^(1/volume).
    
    Measures effective per-volume-unit success retention. Aligns with
    Volumetric Benchmarking methodology.
    
    Args:
        p_succ: Success probability
        circuit_volume: Total circuit volume (gate count × depth)
        eps: Small constant to prevent log(0)
    
    Returns:
        Retention per volume unit or None if volume unavailable/zero.
    
    Edge Cases:
        - circuit_volume is None: Returns None (SDK doesn't provide volume)
        - circuit_volume <= 0: Returns None (trivial circuit)
        - p_succ = 0: Uses eps, yields ~0 retention
        - p_succ = 1: Returns 1.0
    
    Example:
        >>> calculate_retention_per_volume(p_succ=0.75, circuit_volume=1470)
        0.999804  # ~99.98% retention per volume unit
    """
    if circuit_volume is None or circuit_volume <= 0:
        return None
    
    p_clamped = min(max(p_succ, eps), 1.0)
    return p_clamped ** (1.0 / circuit_volume)


def calculate_retention_with_ci(
    p_succ: float,
    ci: Tuple[Optional[float], Optional[float]],
    denominator: int,
    eps: float = 1e-12
) -> Tuple[Optional[float], Tuple[Optional[float], Optional[float]]]:
    """
    Calculate retention metric with confidence interval propagation.
    
    Applies geometric mean transform to both point estimate and CI bounds.
    Retention is monotone increasing in p_succ, so CI order is preserved.
    
    Args:
        p_succ: Success probability point estimate
        ci: Confidence interval for p_succ (lower, upper)
        denominator: Resource count (gates or volume)
        eps: Small constant for numerical stability
    
    Returns:
        Tuple of (retention_point, retention_ci):
        - retention_point: Retention factor or None if denominator invalid
        - retention_ci: Transformed CI or (None, None)
    
    Example:
        >>> calculate_retention_with_ci(0.875, (0.8523, 0.8944), 60)
        (0.9978, (0.9974, 0.9982))
    """
    if denominator <= 0:
        return (None, (None, None))
    
    retention_point = calculate_retention_per_2q(p_succ, denominator, eps)
    
    lo, hi = ci
    if lo is None or hi is None:
        return (retention_point, (None, None))
    
    # Retention is monotone increasing in p_succ
    retention_lo = min(max(lo, eps), 1.0) ** (1.0 / denominator)
    retention_hi = min(max(hi, eps), 1.0) ** (1.0 / denominator)
    
    return (retention_point, (retention_lo, retention_hi))


def calculate_log_loss_with_ci(
    p_succ: float,
    ci: Tuple[Optional[float], Optional[float]],
    denominator: int,
    eps: float = 1e-12
) -> Tuple[Optional[float], Tuple[Optional[float], Optional[float]]]:
    """
    Calculate log loss metric with confidence interval propagation.
    
    Applies log transform to both point estimate and CI bounds. Log loss is
    monotone decreasing in p_succ, so CI order is reversed.
    
    Args:
        p_succ: Success probability point estimate
        ci: Confidence interval for p_succ (lower, upper)
        denominator: Resource count (gates or volume)
        eps: Small constant for numerical stability
    
    Returns:
        Tuple of (log_loss_point, log_loss_ci):
        - log_loss_point: Log loss or None if denominator invalid
        - log_loss_ci: Transformed CI with reversed order (lo, hi)
    
    Example:
        >>> calculate_log_loss_with_ci(0.875, (0.8523, 0.8944), 60)
        (0.00223, (0.00187, 0.00267))
        # Note: CI reversed because log_loss decreases as p increases
    """
    if denominator <= 0:
        return (None, (None, None))
    
    loss_point = calculate_log_loss_per_2q(p_succ, denominator, eps)
    
    lo, hi = ci
    if lo is None or hi is None:
        return (loss_point, (None, None))
    
    # Log loss is monotone decreasing in p_succ
    # Higher p_succ → lower loss, so transform and reverse
    loss_lo = -math.log(min(max(hi, eps), 1.0)) / denominator  # Use hi for lower loss bound
    loss_hi = -math.log(min(max(lo, eps), 1.0)) / denominator  # Use lo for upper loss bound
    
    return (loss_point, (loss_lo, loss_hi))
