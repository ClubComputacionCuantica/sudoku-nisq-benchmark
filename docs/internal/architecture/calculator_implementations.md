# Calculator Function Implementations

**Implementation Files:**
- `src/sudoku_nisq/metrics/calculators/efficiency_metrics.py`
- `src/sudoku_nisq/metrics/calculators/statistical_metrics.py`
- `src/sudoku_nisq/metrics/calculators/ranking_metrics.py`

---

## Overview

These functions auto-compute evaluation and normalization metrics from quantum execution results.

## 1. Efficiency Metrics (Stage 7 Normalization)

### 1.1 Gate Efficiency: `calculate_eta_gate()`

**Purpose:** Normalize success probability by two-qubit gate count (dominant error source in NISQ devices).

**Formula:**
```
η_gate = p_succ / n_gates    (if n_gates > 0)
η_gate = None                (if n_gates = 0, undefined)
```

**Implementation:**
```python
def calculate_eta_gate(p_succ: float, two_qubit_gates: int) -> Optional[float]:
    if two_qubit_gates == 0:
        return None  # Undefined efficiency for zero gates
    return p_succ / two_qubit_gates
```

**Edge Cases:**
- `two_qubit_gates = 0` → Returns `None` (efficiency is undefined, not worst-case)
- `p_succ = 0` → Returns `0.0` (complete failure)
- `p_succ = 1` → Returns `1 / two_qubit_gates` (perfect success normalized by gate count)

**Interpretation:** Higher values indicate better algorithm performance per gate operation. Enables fair comparison across different problem sizes and encoding strategies. `None` indicates the metric is not meaningful for the given circuit (e.g., trivial circuits with no entanglement).

**Example:**
```python
calculate_eta_gate(p_succ=0.75, two_qubit_gates=60)
# Returns: 0.0125 (1.25% success per gate)

calculate_eta_gate(p_succ=0.8, two_qubit_gates=0)
# Returns: None (undefined efficiency)
```

---

### 1.2 Volume Efficiency: `calculate_eta_volume()`

**Purpose:** Normalize success probability by circuit volume (gate-depth product), which correlates with parallelization and coherence time usage.

**Formula:**
```
η_volume = p_succ / circuit_volume    (if circuit_volume > 0)
η_volume = None                       (if circuit_volume = 0 or None, undefined)
```

**Implementation:**
```python
def calculate_eta_volume(p_succ: float, circuit_volume: Optional[int]) -> Optional[float]:
    if circuit_volume is None:
        return None  # Not all SDKs provide volume
    if circuit_volume == 0:
        return None  # Undefined efficiency for zero volume
    return p_succ / circuit_volume
```

**Edge Cases:**
- `circuit_volume is None` → Returns `None` (not all SDKs provide volume; propagate missing data)
- `circuit_volume = 0` → Returns `None` (efficiency is undefined for trivial circuits)
- `p_succ = 0` → Returns `0.0` (complete failure)

**Interpretation:** Aligns with Volumetric Benchmarking standards. Lower volume (more parallel execution) yields higher efficiency. Useful for comparing depth vs. width trade-offs.

**Example:**
```python
calculate_eta_volume(p_succ=0.75, circuit_volume=1470)
# Returns: 0.00051 (0.051% success per volume unit)

calculate_eta_volume(p_succ=0.5, circuit_volume=0)
# Returns: None (undefined efficiency)
```

---

### 1.3 Shot Efficiency: `calculate_eta_shot()`

**Purpose:** Normalize success probability by measurement shot budget, representing sampling effort.

**Formula:**
```
η_shot = p_succ / N_shots
```

**⚠️ WARNING:** This metric has a conceptual flaw. Since `p_succ` is computed from the same `N_shots`, the formula creates 1/N² scaling: doubling shots halves the metric even with constant success rates. Consider dropping or redefining this metric. See technical review for details.

**Implementation:**
```python
def calculate_eta_shot(p_succ: float, shots: int) -> float:
    if shots == 0:
        return 0.0
    return p_succ / shots
```

**Edge Cases:**
- `shots = 0` → Returns `0.0` (no measurements taken)
- `p_succ = 0` → Returns `0.0` (no valid solutions found)
- `p_succ = 1` → Returns `1 / shots` (every shot was valid)

**Interpretation:** Originally intended as cost-efficiency metric, but the 1/N² dependency on shot count makes interpretation problematic. Not recommended for production benchmarks without redefinition.

**Example:**
```python
calculate_eta_shot(p_succ=0.75, shots=1024)
# Returns: 0.000732 (0.073% success per shot)
# ⚠️ Doubling shots to 2048 would halve this value to ~0.000366
```

---

## 2. Statistical Metrics (Stage 6 Evaluation)

### 2.1 Clopper-Pearson Confidence Interval: `calculate_clopper_pearson_ci()`

**Purpose:** Compute exact binomial confidence interval for success probability without normal approximation assumptions. Gold standard for small sample sizes or extreme probabilities.

**Theoretical Basis:**  
Uses the beta distribution relationship to binomial proportions:
```
Lower bound: Beta(α/2; k, n-k+1)
Upper bound: Beta(1-α/2; k+1, n-k)
```
where `k` = num_successes, `n` = num_trials, `α` = 1 - confidence_level.

**Implementation:**
```python
def calculate_clopper_pearson_ci(
    num_successes: int,
    num_trials: int,
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    # Input validation (v1.1 addition)
    if num_successes < 0 or num_trials < 0:
        raise ValueError("Counts cannot be negative")
    if num_successes > num_trials:
        raise ValueError("Successes cannot exceed trials")
    if not (0 < confidence_level < 1):
        raise ValueError("Confidence level must be in (0, 1)")
    
    if num_trials == 0:
        return (0.0, 0.0)
    
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
```

**Dependencies:** `scipy.stats.beta.ppf` for quantile function.

**Edge Cases:**
- `num_successes = 0` → Returns `(0.0, upper_bound)` using beta distribution
- `num_successes = num_trials` → Returns `(lower_bound, 1.0)` (all trials succeeded)
- `num_trials = 0` → Returns `(0.0, 0.0)` (undefined proportion)

**Interpretation:** Provides rigorous uncertainty quantification. The true success probability lies within the interval with the specified confidence level (default 95%).

**Example:**
```python
calculate_clopper_pearson_ci(875, 1000, 0.95)
# Returns: (0.8523, 0.8944) — 95% CI for 87.5% success rate
```

**Reference:** Clopper, C.J., and Pearson, E.S. (1934). "The use of confidence or fiducial limits illustrated in the case of the binomial." *Biometrika* 26(4): 404-413.

---

### 2.2 Signal-to-Noise Ratio: `calculate_snr()`

**Purpose:** Measure how well valid solutions stand out from the noise floor of invalid outcomes.

**Formula (v1.1 corrected):**
```
SNR = valid_mass / invalid_mass

where:
  valid_mass = sum of all valid solution probabilities
  invalid_mass = sum of all invalid solution probabilities
```

**Implementation:**
```python
def calculate_snr(
    counts: Dict[str, int],
    validation_context: Any
) -> float:
    if not counts:
        return 0.0
    
    total_shots = sum(counts.values())
    if total_shots == 0:
        return 0.0
    
    # Compute valid and invalid probability masses
    valid_mass = 0.0
    invalid_mass = 0.0
    
    validator = validation_context.solution_validator
    for bitstring, count in counts.items():
        prob = count / total_shots
        if validator(bitstring):
            valid_mass += prob
        else:
            invalid_mass += prob
    
    # Edge cases
    if invalid_mass == 0.0:
        if valid_mass > 0.0:
            return None  # Perfect discrimination (JSON-safe; see metadata flag)
        else:
            return 0.0  # Empty result
    
    return valid_mass / invalid_mass
```

**Dependencies:** None (pure Python).

**Edge Cases:**
- No valid solutions → `valid_mass = 0` → Returns `0.0` (no signal)
- All solutions valid → `invalid_mass = 0` → Returns `None` (perfect discrimination; metadata flag `snr_is_infinite: true` set separately)
**Example:**
```python
counts = {"00": 500, "01": 450, "10": 30, "11": 20}
validator = lambda bs: bs in ["00", "01"]
calculate_snr(counts, ctx)
# Returns: 19.0 
# (valid_mass = 0.95, invalid_mass = 0.05, SNR = 0.95/0.05 = 19.0)
```

**Reference:** Formula corrected in v1.1 to match documentation examples and intuitive interpretation. Original implementation used mean(valid)/std(invalid) which produced inconsistent results.

---

### 2.3 Variability Statistics: `calculate_variability_stats()`

**Purpose:** Aggregate metrics across multiple experimental runs with the same configuration.

**Output:** Dictionary with comprehensive statistical summary:
- `mean`: Arithmetic mean
- `std`: Sample standard deviation (Bessel correction)
- `median`: 50th percentile
- `q1`: 25th percentile
- `q3`: 75th percentile
- `iqr`: Interquartile range (Q3 - Q1)

**Implementation:**
```python
def calculate_variability_stats(values: list) -> Dict[str, Optional[float]]:
    if not values:
        # Return None for JSON safety (v1.1 change)
        return {
            'mean': None,
            'std': None,
            'median': None,
            'q1': None,
            'q3': None,
            'iqr': None
        }
    
    arr = np.array(values)
    
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    median = float(np.median(arr))
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
```

**Dependencies:** `numpy` for all statistical operations.

**Edge Cases:**
- Empty list → Returns `None` for all metrics (JSON-safe missing data representation)
- Single value → Returns value for mean/median, `0.0` for std/IQR (no variance)
- Two values → Uses sample std with Bessel correction (`ddof=1`)

**Interpretation:** Enables rigorous multi-run analysis. IQR is robust to outliers compared to std. Use median for skewed distributions.

**Example:**
```python
values = [0.85, 0.87, 0.82, 0.88, 0.86]
calculate_variability_stats(values)
# Returns: {'mean': 0.856, 'std': 0.023, 'median': 0.86, 
#           'q1': 0.85, 'q3': 0.87, 'iqr': 0.02}
```

---

## 3. Ranking Metrics (Stage 6 Evaluation)

### 3.1 Top-K Valid Mass: `calculate_top_k_valid_mass()`

**Purpose:** Compute cumulative probability mass of valid solutions within the top-k most frequent bitstrings.

**Formula:**
```
mass_k = Σ(count_i) / total_shots  for valid bitstrings in top-k
```

**Implementation:**
```python
def calculate_top_k_valid_mass(
    counts: Dict[str, int],
    validation_context: Any,
    k_values: List[int]
) -> Dict[int, float]:
    if not counts:
        return {k: 0.0 for k in k_values}
    
    total_shots = sum(counts.values())
    if total_shots == 0:
        return {k: 0.0 for k in k_values}
    
    # Sort bitstrings by frequency (descending)
    sorted_bitstrings = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
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
```

**Edge Cases:**
- Empty counts → Returns `{k: 0.0 for k in k_values}`
- `k > len(counts)` → Clamps to total available bitstrings
- No valid solutions in top-k → Returns `0.0` for that k

**Interpretation:** Measures how well the quantum algorithm concentrates probability on valid solutions. High mass in small k indicates effective amplitude amplification.

**Example:**
```python
counts = {"00": 500, "01": 300, "10": 150, "11": 50}
validator = lambda bs: bs in ["01", "10"]
calculate_top_k_valid_mass(counts, ctx, [1, 2, 3])
# Returns: {1: 0.0, 2: 0.3, 3: 0.45}
# "00" invalid, "01" valid (30%), "10" adds 15%
```

---

### 3.2 Precision at K: `calculate_precision_at_k()`

**Purpose:** Measure the fraction of top-k bitstrings that are valid solutions (low false positive rate).

**Formula:**
```
precision@k = |valid in top-k| / k
```

**Implementation:**
```python
def calculate_precision_at_k(
    counts: Dict[str, int],
    validation_context: Any,
    k_values: List[int]
) -> Dict[int, float]:
    if not counts:
        return {k: 0.0 for k in k_values}
    
    # Sort bitstrings by frequency (descending)
    sorted_bitstrings = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
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
```

**Edge Cases:**
- Empty counts → Returns `{k: 0.0 for k in k_values}`
- `k > len(counts)` → Uses available bitstrings as denominator
- All invalid → Returns `0.0`
- All valid → Returns `1.0`

**Interpretation:** High precision means the top results are "clean" (mostly valid). Useful for applications where users only check the most frequent outcomes.

**Important:** Precision is **not monotonic** in k. If invalid solutions appear in the ranking, precision can increase when more valid solutions are added to the top-k set.

**Example:**
```python
counts = {"01": 400, "00": 300, "10": 200, "11": 100}
validator = lambda bs: bs in ["01", "10"]
calculate_precision_at_k(counts, ctx, [1, 2, 3])
# Returns: {1: 1.0, 2: 0.5, 3: 0.67}
# k=1: 1/1 valid, k=2: 1/2 valid, k=3: 2/3 valid (increases!)
```

---

### 3.3 Recall at K: `calculate_recall_at_k()`

**Purpose:** Measure the fraction of all valid solutions (that appeared in measurements) found within the top-k bitstrings.

**Formula:**
```
recall@k = |valid in top-k| / |all valid observed|
```

**Implementation:**
```python
def calculate_recall_at_k(
    counts: Dict[str, int],
    validation_context: Any,
    k_values: List[int]
) -> Dict[int, float]:
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
    
    # Sort bitstrings by frequency (descending)
    sorted_bitstrings = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
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
```

**Edge Cases:**
- Empty counts → Returns `{k: 0.0 for k in k_values}`
- No valid solutions observed → Returns `{k: 0.0 for k in k_values}` (denominator is 0)
- `k` encompasses all valid solutions → Returns `1.0`
- `k > len(counts)` → Clamps to available bitstrings

**Interpretation:** High recall means the algorithm is discovering a diverse set of valid solutions, not just one or two. Recall is **monotonic increasing** in k (more solutions found as k grows).

**Example:**
```python
counts = {"00": 500, "01": 250, "11": 150, "10": 100}
validator = lambda bs: bs in ["01", "10", "11"]  # 3 valid solutions
calculate_recall_at_k(counts, ctx, [1, 2, 3])
# Returns: {1: 0.0, 2: 1/3≈0.33, 3: 2/3≈0.67}
# k=1: "00" invalid, k=2: "01" found (1/3), k=3: "01"+"11" found (2/3)
```

---

## 4. Integration with MetricsMetadataManager

All implemented calculators are invoked by `MetricsMetadataManager.record()` to auto-compute Stage 6-7 metrics when `SUDOKU_NISQ_NEW_METADATA=1`:

```python
# Stage 6: Evaluation metrics
k_values = [1, 3, 5, 10]  # Configurable ranking depths
p_succ = calculate_p_succ(counts, validation_context.solution_validator)
ci_lower, ci_upper = calculate_clopper_pearson_ci(num_successes, shots)
distinct = calculate_distinct_solutions(counts, validation_context.solution_validator)
snr = calculate_snr(counts, validation_context)
top_k = calculate_top_k_valid_mass(counts, validation_context, k_values)
precision = calculate_precision_at_k(counts, validation_context, k_values)
recall = calculate_recall_at_k(counts, validation_context, k_values)

# Stage 7: Normalization metrics
eta_gate = calculate_eta_gate(p_succ, two_qubit_gates)
eta_volume = calculate_eta_volume(p_succ, circuit_volume)
eta_shot = calculate_eta_shot(p_succ, shots)
```

Results are persisted to `.quantum_solver_cache/{puzzle_hash}/stage_6_7_metrics.json` with full provenance chain linking to Stage 5 execution records.

---

## 5. Tie-Breaking and Determinism

**Ranking metrics sorting:** All ranking metrics (`calculate_top_k_valid_mass`, `calculate_precision_at_k`, `calculate_recall_at_k`) sort bitstrings by count descending. For production use, consider deterministic tie-breaking:

```python
sorted(counts.items(), key=lambda x: (-x[1], x[0]))  # Break ties lexicographically
```

Current implementation uses `key=lambda x: x[1]` only, which may produce nondeterministic ordering when multiple bitstrings have identical counts (depends on dict iteration order, Python version, platform). This is acceptable for aggregate statistics but may affect reproducibility of specific top-k lists across runs.

**Recommendation:** Add deterministic tie-breaking if exact bitstring rankings are persisted or compared across experiments.

---

## 5. Cost-Normalized Efficiency Metrics

**Implementation File:** `src/sudoku_nisq/metrics/calculators/cost_metrics.py`

**Purpose:** Alternative normalization approaches using heuristic cost models. These complement the retention-based metrics by offering different perspectives on resource efficiency.

### 5.1 Product-Based Normalization: `calculate_eta_product()`

**Formula:**
```
η_× = p_succ / (depth × n_2q)    (if depth × n_2q > 0)
η_× = None                         (otherwise)
```

**Implementation:**
```python
def calculate_eta_product(
    p_succ: float,
    depth: int,
    two_qubit_gates: int
) -> Optional[float]:
    if not (0 <= p_succ <= 1):
        return None
    if depth < 0 or two_qubit_gates < 0:
        return None
    cost = depth * two_qubit_gates
    if cost == 0:
        return None
    return p_succ / cost
```

**Edge Cases:**
- `depth = 0` or `two_qubit_gates = 0` → `None` (undefined efficiency)
- `p_succ < 0` or `p_succ > 1` → `None` (invalid probability)
- `depth < 0` or `two_qubit_gates < 0` → `None` (invalid inputs)

**Interpretation:** Treats depth × 2Q count as a single "circuit volume" measure. Simple but risks double-counting when depth and gate count are correlated. Higher is better.

**Gotchas:**
- **Double-counting risk:** Depth and 2Q count often correlated (more gates → deeper circuit)
- **Scheduling sensitivity:** Parallel scheduling changes depth without changing gate count
- Not directly actionable (doesn't answer "how many shots?" or "what's per-gate failure?")

---

### 5.2 Weighted-Sum Normalization: `calculate_eta_weighted_sum()`

**Formula:**
```
C = α·depth + β·n_2q
η_+ = p_succ / C    (if C > 0)
η_+ = None          (otherwise)
```

**Implementation:**
```python
def calculate_eta_weighted_sum(
    p_succ: float,
    depth: int,
    two_qubit_gates: int,
    alpha: float = 1.0,
    beta: float = 1.0
) -> Optional[float]:
    if not (0 <= p_succ <= 1):
        return None
    if depth < 0 or two_qubit_gates < 0:
        return None
    if alpha < 0 or beta < 0:
        return None
    cost = alpha * depth + beta * two_qubit_gates
    if cost == 0:
        return None
    return p_succ / cost
```

**Edge Cases:**
- `cost = 0` (both weighted terms zero) → `None`
- Invalid inputs (negative values, p_succ out of range) → `None`

**Weight Selection:**
- **α=0, β=1:** "Per 2Q gate" (ignores depth)
- **α=1, β=0:** "Per depth" (ignores gates)
- **α=1, β=1:** Simple balanced blend (default)
- **Hardware-informed:** α ~ layer time, β ~ 2Q error
- **Fitted:** Empirical from dataset (see `fit_cost_weights`)

**Interpretation:** Avoids over-penalization from product. Tunable to hardware characteristics. Higher is better.

---

### 5.3 Decay Rate: `calculate_decay_rate()`

**Formula:**

Assumes exponential decay model:
```
p_succ ≈ exp(-k·C)  where C = α·depth + β·n_2q

Solve for k:
k = -log(p_succ) / C    (if C > 0 and p_succ > 0)
k = None                (otherwise)
```

**Implementation:**
```python
def calculate_decay_rate(
    p_succ: float,
    depth: int,
    two_qubit_gates: int,
    alpha: float = 1.0,
    beta: float = 1.0,
    epsilon: float = 1e-10
) -> Optional[float]:
    if p_succ < 0 or p_succ > 1:
        return None
    if depth < 0 or two_qubit_gates < 0:
        return None
    if alpha < 0 or beta < 0:
        return None
    cost = alpha * depth + beta * two_qubit_gates
    if cost == 0:
        return None
    p_clamped = max(p_succ, epsilon)
    return -math.log(p_clamped) / cost
```

**Edge Cases:**
- `p_succ = 0` → Clamped to `epsilon` (avoids log(0))
- `cost = 0` → `None` (undefined)
- Invalid inputs → `None`

**Interpretation:** k is "decay constant" — **SMALLER is better** (less penalty per resource unit). Connects to physics: exponential fidelity decay. If k is constant across circuits, the cost model C captures dominant scaling.

**Relationship to retention:** Both use exponential models. Retention = p^(1/n) focuses on gates only; decay rate generalizes to weighted cost C.

---

### 5.4 Weight Fitting: `fit_cost_weights()`

**Purpose:** Empirically estimate α, β from dataset assuming exponential decay model.

**Formula:**

Log-linear regression:
```
-log(p_succ_i) ≈ α·depth_i + β·n_2q_i + intercept
```

Solves normal equations for multiple linear regression:
```
[α, β]ᵀ = (XᵀX)⁻¹ Xᵀy
```

**Implementation:**
```python
def fit_cost_weights(
    results: List[Tuple[float, int, int]],
    epsilon: float = 1e-10
) -> Tuple[float, float, float]:
    # Extract data, center, solve normal equations
    # Returns (alpha, beta, r_squared)
    # Constrains α, β ≥ 0 (negative weights unphysical)
```

**Edge Cases:**
- `len(results) < 2` → Raises `ValueError` (can't fit model)
- All depths identical → Raises `ValueError` (can't separate α from β)
- All gate counts identical → Raises `ValueError` (same issue)
- Determinant near zero → Raises `ValueError` (perfect correlation)

**Interpretation:**
- **α:** Penalty per layer (decoherence exposure)
- **β:** Penalty per 2Q gate (gate errors)
- **R² near 1:** Model fits well
- **R² near 0:** Cost model doesn't capture scaling

**Requirements:**
- 10+ diverse circuits (vary depth and gates independently)
- Exponential decay assumption holds
- Low outlier sensitivity (OLS implementation; consider robust regression for production)

---

## 6. Implementation Notes

### 6.1 Design Decisions

1. **Zero-division handling:** Return `None` for undefined denominators (zero gates/volume) rather than raising exceptions or returning `0.0`. Justification: Division by zero indicates degenerate cases where efficiency is undefined, not zero efficiency. `None` preserves the distinction between "unmeasurable" and "measured as zero". For shot efficiency, return `0.0` as a safe default since shots=0 means no work performed.

2. **None propagation:** `calculate_eta_volume()` returns `None` when circuit_volume is unavailable, rather than skipping or using a default. Justification: Preserves information about missing data vs. zero values.

3. **Bessel correction:** All standard deviation calculations use `ddof=1` (sample std, not population std). Justification: Multi-run aggregations are samples from an underlying distribution, not the full population.

4. **ValidationContext pattern:** Calculators expect an object with a `solution_validator` attribute (callable). Justification: Matches `ValidationContext` dataclass in `metrics/data_models.py` while allowing test fixtures flexibility.

5. **Beta distribution for CI:** Use `scipy.stats.beta.ppf` rather than normal approximation. Justification: Exact method works for all sample sizes and probabilities; normal approximation fails for small n or extreme p.

### 6.2 Known Limitations

1. **SNR interpretation:** The ratio `valid_mass / invalid_mass` measures signal strength but doesn't account for distribution shape. An SNR of 19.0 means 95% valid vs. 5% invalid, which is excellent discrimination. However, this tells nothing about how probability is distributed *within* valid solutions (concentrated on one solution vs. spread across many).

2. **SNR infinity edge case:** When all measured solutions are valid (`invalid_mass = 0`), the function returns `None` with an accompanying `snr_is_infinite: true` flag in Stage 6 metadata. This avoids JSON serialization issues with `float('inf')` while preserving the mathematical meaning of perfect discrimination. Downstream analysis should interpret `snr=None` + `snr_is_infinite=true` as unbounded signal-to-noise.

3. **Precision non-monotonicity:** Precision@k can increase with k if more valid solutions are added faster than invalid ones. This is expected behavior but may be counterintuitive.

4. **Recall denominator:** Uses *observed* valid count (bitstrings with non-zero counts), not total problem solutions. Justification: Cannot infer unobserved solutions from measurement data alone.

---

## Appendix: Formula Quick Reference

| Metric | Formula | Returns | Edge Case Handling |
|--------|---------|---------|-------------------|
| `η_gate` | `p_succ / n_gates` | `float \| None` | gates=0 → None |
| `η_volume` | `p_succ / circuit_volume` | `float \| None` | None → None, 0 → None |
| `η_shot` | `p_succ / shots` | `float` | shots=0 → 0.0 |
| `η_×` | `p_succ / (depth × n_2q)` | `float \| None` | cost=0 → None |
| `η_+` | `p_succ / (α·depth + β·n_2q)` | `float \| None` | cost=0 → None, α,β<0 → None |
| `k` (decay) | `-log(p_succ) / (α·depth + β·n_2q)` | `float \| None` | cost=0 → None, p≤0 → None |
| `CI` | `Beta(α/2; k, n-k+1), Beta(1-α/2; k+1, n-k)` | `(float, float) \| (None, None)` | n=0 → (None, None) |
| `SNR` | `valid_mass / invalid_mass` | `float \| None` | invalid=0 → None (w/ flag) |
| `variability` | `mean, std, median, q1, q3, iqr` | `Dict[str, float \| None]` | empty → None |
| `top_k_mass` | `Σ(valid_counts in top-k) / total` | `Dict[int, float]` | empty → 0.0 |
| `precision@k` | `\|valid in top-k\| / k` | `Dict[int, float]` | empty → 0.0 |
| `recall@k` | `\|valid in top-k\| / \|all valid\|` | `Dict[int, float]` | no valid → 0.0 |

**Note:** CI now returns `(None, None)` for n=0 (updated from legacy `(0.0, 0.0)`).

---

**End of Document**
