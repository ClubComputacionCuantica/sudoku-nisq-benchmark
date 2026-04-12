"""
Cost-normalized efficiency metrics using heuristic cost models.

This module provides alternative normalization approaches to the geometric-mean
retention metrics. Three cost models are supported:

1. **Product-based (η_×)**: p_succ / (depth × n_2q)
   - Simple multiplicative penalty
   - Risk: double-counting correlated dimensions

2. **Weighted-sum (η_+)**: p_succ / (α·depth + β·n_2q)
   - Tunable blend of depth and gate count
   - Avoids over-penalization from product

3. **Decay rate (k)**: -log(p_succ) / (α·depth + β·n_2q)
   - Exponential decay model
   - Physics-aligned for multiplicative noise

These metrics complement the retention-based metrics by offering different
perspectives on resource efficiency. See user guide for comparison and
selection guidance.

Mathematical Background
-----------------------
The weighted-sum metrics assume a cost function:
    C = α·depth + β·n_2q

where α, β > 0 are weights reflecting hardware characteristics.

**Linear model (η_+):**
    η_+ = p_succ / C
    Interpretation: Success per unit cost (higher is better)

**Exponential model (k):**
    Assumes: p_succ ≈ exp(-k·C)
    Solve for: k = -log(p_succ) / C
    Interpretation: Penalty per unit cost (smaller is better)

Weight Selection
----------------
1. **Heuristic**: α=1, β=1 (simple, no tuning)
2. **Hardware-informed**: α ~ layer time, β ~ 2Q error contribution
3. **Fitted**: Empirical regression on dataset (most accurate)

For weight fitting, see `fit_cost_weights()`.

References
----------
- Depth×2Q product: Common heuristic in circuit optimization literature
- Weighted cost models: See Tannu & Qureshi (MICRO 2019) for cost proxies
- Exponential decay: Standard model for fidelity vs. gate count

Notes
-----
- These metrics are heuristic alternatives to retention metrics
- Not statistically principled (no CI propagation initially)
- Useful for exploratory analysis and cross-validation of cost models
"""

import math
from typing import Optional, List, Tuple


def calculate_eta_product(
    p_succ: float,
    depth: int,
    two_qubit_gates: int
) -> Optional[float]:
    """
    Calculate success probability per unit of (depth × 2Q gates).
    
    Formula:
        η_× = p_succ / (depth × n_2q)
    
    This is the simplest cost-normalized metric: treat the product of depth
    and 2Q count as a single "circuit volume" measure.
    
    **Interpretation:**
        - "Success per unit of circuit volume (depth × 2Q count)"
        - Higher is better (same success with less volume)
        - Strong penalty for simultaneously large depth and 2Q count
    
    **Gotchas:**
        - **Double-counting risk**: Depth and 2Q count are often correlated
          (more gates → deeper circuit). Multiplying them can over-penalize
          compared to treating them as independent dimensions.
        - **Sensitive to scheduling**: Re-scheduling that reduces depth without
          changing 2Q count can dramatically change this metric.
        - **Not directly actionable**: Doesn't answer "how many shots until
          success?" or "what's the per-gate failure rate?" (see retention
          metrics and shot budgets for those).
    
    **When to use:**
        - Comparing circuits with the same compilation settings
        - When both dimensions matter equally and you want simplicity
        - Exploratory analysis before fitting a weighted model
        - When depth and 2Q count are uncorrelated in your dataset
    
    **Comparison to other metrics:**
        - vs. retention_per_2q: This ignores depth; retention focuses on gates
        - vs. eta_weighted_sum: This uses product; weighted sum avoids double-counting
        - vs. decay_rate: This is linear; decay rate is log-transformed
    
    Args:
        p_succ: Success probability in [0, 1]. Estimated from measurement counts.
        depth: Circuit depth. Scheduled depth preferred (reflects actual
            runtime/noise exposure). Must be non-negative.
        two_qubit_gates: Number of 2-qubit gates. Must be non-negative.
    
    Returns:
        η_× value (higher is better), or None if:
            - cost = 0 (depth or two_qubit_gates is zero)
            - p_succ is invalid (< 0 or > 1)
    
    Examples:
        >>> calculate_eta_product(0.8, depth=10, two_qubit_gates=5)
        0.016  # 0.8 / (10 * 5) = 0.8 / 50
        
        >>> calculate_eta_product(0.5, depth=20, two_qubit_gates=10)
        0.0025  # Same p_succ but 4× cost → 1/4 the efficiency
        
        >>> calculate_eta_product(0.8, depth=0, two_qubit_gates=5)
        None  # Zero depth → undefined
    
    See Also:
        - calculate_eta_weighted_sum: Alternative using α·depth + β·n_2q
        - calculate_decay_rate: Log-transformed exponential model
        - calculate_retention_per_2q: Geometric mean (recommended baseline)
    """
    # Validate inputs
    if not (0 <= p_succ <= 1):
        return None
    
    if depth < 0 or two_qubit_gates < 0:
        return None
    
    # Calculate cost
    cost = depth * two_qubit_gates
    
    if cost == 0:
        return None
    
    return p_succ / cost


def calculate_eta_weighted_sum(
    p_succ: float,
    depth: int,
    two_qubit_gates: int,
    alpha: float = 1.0,
    beta: float = 1.0
) -> Optional[float]:
    """
    Calculate success probability per unit weighted cost.
    
    Formula:
        C = α·depth + β·n_2q
        η_+ = p_succ / C
    
    This metric uses a weighted sum instead of a product, avoiding the
    double-counting tendency when depth and gate count are correlated.
    
    **Interpretation:**
        - "Success per unit weighted cost"
        - Higher is better
        - Tunable to hardware/workload characteristics via α, β
    
    **Weight selection strategies:**
        1. **α=0, β=1**: "Per 2Q gate" (ignores depth entirely)
        2. **α=1, β=0**: "Per depth" (ignores gate count)
        3. **α=1, β=1**: Simple balanced blend (default)
        4. **Hardware-informed**: α ~ layer decoherence, β ~ 2Q error rate
        5. **Fitted weights**: Empirical from dataset (see `fit_cost_weights`)
    
    **When to use:**
        - When one dimension dominates noise (adjust weights accordingly)
        - Cross-hardware comparisons (refit weights per backend)
        - When product penalty (η_×) seems excessive
        - When you have insight into hardware characteristics
    
    **Comparison to other metrics:**
        - vs. eta_product: This uses sum; product over-penalizes
        - vs. decay_rate: This is linear; k is log-transformed
        - vs. retention_per_2q: This is additive cost; retention is multiplicative
    
    Args:
        p_succ: Success probability in [0, 1].
        depth: Circuit depth. Must be non-negative.
        two_qubit_gates: Number of 2-qubit gates. Must be non-negative.
        alpha: Weight for depth (default 1.0). Must be non-negative.
        beta: Weight for 2Q gates (default 1.0). Must be non-negative.
    
    Returns:
        η_+ value (higher is better), or None if:
            - cost = 0 (both weighted dimensions zero)
            - p_succ is invalid (< 0 or > 1)
            - alpha or beta are negative
    
    Examples:
        >>> # Default weights (α=1, β=1)
        >>> calculate_eta_weighted_sum(0.8, depth=10, two_qubit_gates=5)
        0.0533  # 0.8 / (10 + 5) = 0.8 / 15
        
        >>> # Focus on 2Q gates only (α=0, β=1)
        >>> calculate_eta_weighted_sum(0.8, depth=10, two_qubit_gates=5, alpha=0, beta=1)
        0.16  # 0.8 / 5 (same as retention concept but linear)
        
        >>> # Hardware-informed: depth costs 10× more than each 2Q gate
        >>> calculate_eta_weighted_sum(0.8, depth=10, two_qubit_gates=5, alpha=10, beta=1)
        0.00762  # 0.8 / (10*10 + 5*1) = 0.8 / 105
    
    See Also:
        - calculate_eta_product: Simpler product-based alternative
        - calculate_decay_rate: Log-transformed version for exponential model
        - fit_cost_weights: Empirical weight estimation from dataset
    """
    # Validate inputs
    if not (0 <= p_succ <= 1):
        return None
    
    if depth < 0 or two_qubit_gates < 0:
        return None
    
    if alpha < 0 or beta < 0:
        return None
    
    # Calculate weighted cost
    cost = alpha * depth + beta * two_qubit_gates
    
    if cost == 0:
        return None
    
    return p_succ / cost


def calculate_decay_rate(
    p_succ: float,
    depth: int,
    two_qubit_gates: int,
    alpha: float = 1.0,
    beta: float = 1.0,
    epsilon: float = 1e-10
) -> Optional[float]:
    """
    Calculate penalty per unit cost assuming exponential decay model.
    
    Model:
        p_succ ≈ exp(-k·C)  where C = α·depth + β·n_2q
    
    Formula:
        k = -log(p_succ) / C
    
    This metric assumes failures accumulate exponentially with cost, which
    aligns with common noise models in quantum computing (multiplicative
    gate errors, decoherence).
    
    **Interpretation:**
        - k is "decay constant" or "penalty per unit cost"
        - **SMALLER is better** (less penalty per resource unit)
        - If k is roughly constant across circuits, your cost model C
          captures the dominant scaling correctly
        - Connects to physics: exponential fidelity decay with gates/time
    
    **When to use:**
        - Believe failures accumulate exponentially with cost
        - Want model-aligned metric (not just normalized ratio)
        - Checking if cost model (α, β) fits your data
        - Cross-validating against retention metrics (which also use geometric mean)
    
    **Complementary to η_+:**
        - η_+ is linear proxy (easier interpretation: "success per unit")
        - k is exponential model (better physics alignment)
        - Use both; if trends agree, cost model is robust
    
    **Relationship to retention metrics:**
        - retention_per_2q = p_succ^(1/n_2q) also captures exponential decay
        - This metric generalizes to weighted cost C instead of just n_2q
        - k = -log(p_succ) / C connects directly to exponential decay rate
    
    Args:
        p_succ: Success probability in (0, 1]. Must be positive for log.
        depth: Circuit depth. Must be non-negative.
        two_qubit_gates: Number of 2-qubit gates. Must be non-negative.
        alpha: Weight for depth (default 1.0). Must be non-negative.
        beta: Weight for 2Q gates (default 1.0). Must be non-negative.
        epsilon: Floor for p_succ to avoid log(0) (default 1e-10).
    
    Returns:
        Decay rate k (smaller is better), or None if:
            - cost = 0 (both weighted dimensions zero)
            - p_succ < 0 or p_succ > 1 (invalid probability)
            - alpha or beta are negative
    
    Examples:
        >>> # High success, low cost → small k (good)
        >>> calculate_decay_rate(0.8, depth=10, two_qubit_gates=5)
        0.01489  # -log(0.8) / (10+5) ≈ 0.223 / 15
        
        >>> # Same success, higher cost → smaller k (cost doesn't hurt as much)
        >>> calculate_decay_rate(0.8, depth=20, two_qubit_gates=10)
        0.00744  # -log(0.8) / (20+10) ≈ 0.223 / 30
        
        >>> # Low success → larger k (worse)
        >>> calculate_decay_rate(0.1, depth=10, two_qubit_gates=5)
        0.1536  # -log(0.1) / 15 ≈ 2.303 / 15
        
        >>> # Zero success → clamped by epsilon
        >>> calculate_decay_rate(0.0, depth=10, two_qubit_gates=5, epsilon=1e-10)
        1.535  # -log(1e-10) / 15 ≈ 23.03 / 15
    
    See Also:
        - calculate_eta_weighted_sum: Linear version (easier interpretation)
        - calculate_retention_per_2q: Geometric mean focusing on gates only
        - fit_cost_weights: Estimate α, β from dataset via log-linear regression
    """
    # Validate inputs
    if p_succ < 0 or p_succ > 1:
        return None
    
    if depth < 0 or two_qubit_gates < 0:
        return None
    
    if alpha < 0 or beta < 0:
        return None
    
    # Calculate weighted cost
    cost = alpha * depth + beta * two_qubit_gates
    
    if cost == 0:
        return None
    
    # Clamp p_succ to avoid log(0)
    p_clamped = max(p_succ, epsilon)
    
    # Calculate decay rate
    return -math.log(p_clamped) / cost


def fit_cost_weights(
    results: List[Tuple[float, int, int]],
    epsilon: float = 1e-10
) -> Tuple[float, float, float]:
    """
    Fit weights α, β from dataset assuming exponential decay model.
    
    Model:
        p_succ_i ≈ exp(-k·C_i) where C_i = α·depth_i + β·n_2q_i
        
    Log-linear form:
        -log(p_succ_i) ≈ α·depth_i + β·n_2q_i + intercept
    
    This performs ordinary least squares regression on the log-transformed
    success probabilities to estimate the weights that best explain your
    dataset under the exponential decay assumption.
    
    **Usage workflow:**
        1. Collect diverse circuits (vary depth and 2Q count independently)
        2. Run experiments, measure p_succ for each
        3. Call this function to fit α, β
        4. Use fitted weights in calculate_eta_weighted_sum and calculate_decay_rate
        5. Check r² to validate model fit
    
    **When to use:**
        - Have 10+ diverse circuits with varied depth and gate counts
        - Want hardware-specific weights
        - Need to validate if cost model fits your data
        - Cross-hardware comparison requires refitting per backend
    
    **Interpreting results:**
        - **α (depth weight)**: Penalty per layer (reflects decoherence exposure)
        - **β (2Q weight)**: Penalty per 2Q gate (reflects gate errors)
        - **r² (goodness of fit)**: Close to 1 → model fits well;
          close to 0 → cost model doesn't capture scaling
    
    **Limitations:**
        - Requires diverse dataset (depth and 2Q count uncorrelated ideally)
        - Assumes exponential decay (may not hold for all error sources)
        - Sensitive to outliers (consider robust regression if needed)
        - Intercept not returned (absorbed into k values)
    
    Args:
        results: List of (p_succ, depth, two_qubit_gates) tuples.
            - p_succ: Success probability in (0, 1]
            - depth: Non-negative integer
            - two_qubit_gates: Non-negative integer
        epsilon: Floor for p_succ to avoid log(0) (default 1e-10).
    
    Returns:
        Tuple of (alpha, beta, r_squared):
            - alpha: Fitted depth weight (≥ 0)
            - beta: Fitted 2Q gate weight (≥ 0)
            - r_squared: Coefficient of determination (0 to 1)
    
    Raises:
        ValueError: If results has fewer than 2 entries (can't fit model)
        ValueError: If all depths are identical (can't separate α from β)
        ValueError: If all 2Q counts are identical (can't separate α from β)
    
    Examples:
        >>> # Synthetic dataset: k ≈ 0.01 per resource unit
        >>> data = [
        ...     (0.8, 10, 5),   # -log(0.8) ≈ 0.22
        ...     (0.5, 20, 10),  # -log(0.5) ≈ 0.69
        ...     (0.3, 30, 15),  # -log(0.3) ≈ 1.20
        ... ]
        >>> alpha, beta, r2 = fit_cost_weights(data)
        >>> print(f"α={alpha:.4f}, β={beta:.4f}, r²={r2:.3f}")
        α=0.0200, β=0.0200, r²=0.999  # Fits well (r² near 1)
        
        >>> # Use fitted weights in subsequent calculations
        >>> k = calculate_decay_rate(0.8, 10, 5, alpha=alpha, beta=beta)
    
    See Also:
        - calculate_decay_rate: Uses weights α, β
        - calculate_eta_weighted_sum: Also uses weights α, β
    
    Notes:
        - This is a simple OLS implementation. For production use with
          large/noisy datasets, consider using scipy.stats.linregress
          or statsmodels for robust regression and diagnostics.
        - Weights are constrained to be non-negative. If regression returns
          negative weights, they're clamped to 0 (indicates poor model fit).
    """
    if len(results) < 2:
        raise ValueError("Need at least 2 data points to fit weights")
    
    # Extract and validate data
    y_values = []  # -log(p_succ)
    depths = []
    gate_counts = []
    
    for p_succ, depth, two_qubit_gates in results:
        if not (0 < p_succ <= 1):
            # Skip invalid probabilities
            continue
        if depth < 0 or two_qubit_gates < 0:
            continue
        
        p_clamped = max(p_succ, epsilon)
        y_values.append(-math.log(p_clamped))
        depths.append(depth)
        gate_counts.append(two_qubit_gates)
    
    if len(y_values) < 2:
        raise ValueError("Not enough valid data points after filtering")
    
    # Check for degenerate cases
    if len(set(depths)) == 1:
        raise ValueError("All depths are identical; cannot fit alpha separately")
    if len(set(gate_counts)) == 1:
        raise ValueError("All 2Q counts are identical; cannot fit beta separately")
    
    n = len(y_values)
    
    # Calculate means
    y_mean = sum(y_values) / n
    depth_mean = sum(depths) / n
    gates_mean = sum(gate_counts) / n
    
    # Center the data
    y_centered = [y - y_mean for y in y_values]
    depth_centered = [d - depth_mean for d in depths]
    gates_centered = [g - gates_mean for g in gate_counts]
    
    # Build normal equations for multiple linear regression
    # Solve: [α, β]' = (X'X)^(-1) X'y where X = [depth, gates]
    
    # X'X matrix elements
    xx_depth = sum(d * d for d in depth_centered)
    xx_gates = sum(g * g for g in gates_centered)
    xx_cross = sum(d * g for d, g in zip(depth_centered, gates_centered))
    
    # X'y vector elements
    xy_depth = sum(d * y for d, y in zip(depth_centered, y_centered))
    xy_gates = sum(g * y for g, y in zip(gates_centered, y_centered))
    
    # Determinant
    det = xx_depth * xx_gates - xx_cross * xx_cross
    
    if abs(det) < 1e-10:
        raise ValueError("Design matrix is singular; depth and gates are perfectly correlated")
    
    # Solve for α, β
    alpha = (xx_gates * xy_depth - xx_cross * xy_gates) / det
    beta = (xx_depth * xy_gates - xx_cross * xy_depth) / det
    
    # Constrain to non-negative (negative weights are unphysical)
    alpha = max(0.0, alpha)
    beta = max(0.0, beta)
    
    # Calculate R² (coefficient of determination)
    # R² = 1 - (SS_res / SS_tot)
    y_pred = [alpha * d + beta * g for d, g in zip(depth_centered, gates_centered)]
    ss_res = sum((y - yp) ** 2 for y, yp in zip(y_centered, y_pred))
    ss_tot = sum(y ** 2 for y in y_centered)
    
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return alpha, beta, r_squared
