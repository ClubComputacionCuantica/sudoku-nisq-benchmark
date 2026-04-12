"""Stages 6-7: Metrics metadata manager.

Auto-computes and stores evaluation (Stage 6) and normalization (Stage 7) metrics.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
from datetime import datetime
import logging

from sudoku_nisq.metadata.base import StageMetadataManager
from sudoku_nisq.metadata.config import MetadataConfig

if TYPE_CHECKING:
    pass
from sudoku_nisq.metrics.calculators import (
    calculate_p_succ,
    calculate_distinct_solutions,
    count_valid_shots,
    calculate_top_k_valid_mass,
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_clopper_pearson_ci,
    calculate_variability_stats,
    # New improved metrics
    calculate_valid_odds_with_ci,
    calculate_peak_metrics,
    calculate_mass_precision_at_k,
    calculate_valid_mass_capture_at_k,
    calculate_retention_with_ci,
    calculate_log_loss_with_ci,
    calculate_shot_budgets,
    # Cost-normalized metrics (heuristic alternatives)
    calculate_eta_product,
    calculate_eta_weighted_sum,
    calculate_decay_rate
)

logger = logging.getLogger(__name__)


class MetricsMetadataManager(StageMetadataManager):
    """Manages Stages 6-7: Evaluation + Normalization metrics.
    
    Auto-computes metrics from execution results:
    
    Stage 6 (Evaluation - α, σ):
    - Success probability (p_succ) with confidence intervals
    - Distinct valid solutions found
    - Valid odds (p_succ / (1-p_succ)) with confidence intervals
    - Peak discrimination metrics (p_best_valid, p_best_invalid, peak_ratio, peak_gap)
    - Count-based ranking metrics (top-k, precision@k, recall@k)
    - Mass-weighted ranking metrics (mass_precision@k, valid_mass_capture@k)
    
    Stage 7 (Normalization - τ):
    - Retention-based metrics (log_loss_per_2q, retention_per_2q, per_volume variants)
    - Shot budget metrics (shots_detect with confidence bounds)
    - Cost-normalized metrics (eta_product, eta_weighted_sum, decay_rate)
    
    Legacy metrics (deprecated):
    - SNR (replaced by valid_odds)
    - eta_gate, eta_volume, eta_shot (replaced by retention metrics and shot budgets)
    
    Also computes multi-run aggregations (mean, std, median, IQR).
    
    Storage: {puzzle_hash}/stage_6_7_metrics.json
    Structure: {run_id: {stage_6_evaluation: {...}, stage_7_normalization: {...}}}
    """
    
    @property
    def stage_number(self) -> int:
        return 6  # Covers both Stage 6 and Stage 7
    
    def __init__(self, cache_base: Path, puzzle_hash: str):
        """Initialize metrics metadata manager.
        
        :param cache_base: Base cache directory (e.g., .quantum_solver_cache)
        :param puzzle_hash: Puzzle identifier (determines storage subdirectory)
        """
        self.cache_base = Path(cache_base)
        self.puzzle_hash = puzzle_hash
        self._storage_path = self.cache_base / puzzle_hash / MetadataConfig.STAGE_6_7_METRICS
    
    @property
    def storage_path(self) -> Path:
        return self._storage_path
    
    def record(  # type: ignore[override]
        self,
        run_id: str,
        counts: Dict[str, int],
        validation_context: Optional[Any] = None,
        shots: Optional[int] = None,
        two_qubit_gates: Optional[int] = None,
        circuit_volume: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Auto-compute and store metrics for execution.
        
        Computes Stage 6 (evaluation) and Stage 7 (normalization) metrics from
        execution results and circuit resources. If validation_context is not
        provided, only computes efficiency metrics (Stage 7).
        
        :param run_id: Execution run ID from Stage 5 (UUID)
        :param counts: Measurement counts dict (bitstring -> count)
        :param validation_context: Validation context with solution_validator
        :param shots: Total number of shots (derived from counts if not provided)
        :param two_qubit_gates: Number of two-qubit gates (required for eta_gate)
        :param circuit_volume: Circuit volume (optional, for eta_volume)
        :param kwargs: Additional parameters (reserved for future use)
        :returns: Dictionary with computed metrics (both Stage 6 and Stage 7)
        :raises ValueError: If required parameters missing
        
        Example:
            >>> manager = MetricsMetadataManager(cache_base, puzzle_hash)
            >>> metrics = manager.record(
            ...     run_id="abc123",
            ...     counts={"00": 500, "11": 500},
            ...     validation_context=ctx,
            ...     two_qubit_gates=60
            ... )
        """
        # Validate required parameters
        if not run_id:
            raise ValueError("run_id is required")
        if not counts:
            raise ValueError("counts dictionary is required")
        
        # Derive shots from counts if not provided
        if shots is None:
            shots = sum(counts.values())
        
        metrics: Dict[str, Any] = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "stage_6_evaluation": {},
            "stage_7_normalization": {}
        }
        
        # Stage 6: Evaluation metrics (requires validation context)
        if validation_context is not None:
            try:
                # Success probability
                p_succ = calculate_p_succ(counts, validation_context.solution_validator)
                metrics["stage_6_evaluation"]["p_succ"] = p_succ
                
                # Confidence intervals
                num_successes = count_valid_shots(counts, validation_context.solution_validator)
                ci_lower, ci_upper = calculate_clopper_pearson_ci(num_successes, shots)
                metrics["stage_6_evaluation"]["p_succ_ci_lower"] = ci_lower
                metrics["stage_6_evaluation"]["p_succ_ci_upper"] = ci_upper
                
                # Distinct valid solutions
                distinct = calculate_distinct_solutions(counts, validation_context.solution_validator)
                metrics["stage_6_evaluation"]["distinct_valid_solutions"] = distinct
                
                # Odds-based metrics (replaces SNR)
                odds, odds_ci, is_infinite = calculate_valid_odds_with_ci(p_succ, (ci_lower, ci_upper))
                metrics["stage_6_evaluation"]["valid_odds"] = odds
                metrics["stage_6_evaluation"]["valid_odds_ci_lower"] = odds_ci[0]
                metrics["stage_6_evaluation"]["valid_odds_ci_upper"] = odds_ci[1]
                metrics["stage_6_evaluation"]["valid_odds_is_infinite"] = is_infinite
                
                # Peak-based discrimination metrics
                peak_metrics = calculate_peak_metrics(counts, validation_context)
                metrics["stage_6_evaluation"]["p_best_valid"] = peak_metrics["p_best_valid"]
                metrics["stage_6_evaluation"]["p_best_invalid"] = peak_metrics["p_best_invalid"]
                metrics["stage_6_evaluation"]["peak_ratio"] = peak_metrics["peak_ratio"]
                metrics["stage_6_evaluation"]["peak_gap"] = peak_metrics["peak_gap"]
                metrics["stage_6_evaluation"]["peak_ratio_is_infinite"] = peak_metrics["peak_ratio_is_infinite"]
                
                # Count-based ranking metrics (existing)
                k_values = [1, 3, 5, 10]
                top_k = calculate_top_k_valid_mass(counts, validation_context, k_values)
                precision = calculate_precision_at_k(counts, validation_context, k_values)
                recall = calculate_recall_at_k(counts, validation_context, k_values)
                
                metrics["stage_6_evaluation"]["top_k_valid_mass"] = top_k
                metrics["stage_6_evaluation"]["precision_at_k"] = precision
                metrics["stage_6_evaluation"]["recall_at_k"] = recall
                
                # Mass-weighted ranking metrics (improved)
                mass_precision = calculate_mass_precision_at_k(counts, validation_context, k_values)
                valid_capture = calculate_valid_mass_capture_at_k(counts, validation_context, k_values)
                metrics["stage_6_evaluation"]["mass_precision_at_k"] = mass_precision
                metrics["stage_6_evaluation"]["valid_mass_capture_at_k"] = valid_capture
                
            except Exception as e:
                logger.warning(f"Failed to compute Stage 6 metrics: {e}")
                metrics["stage_6_evaluation"]["error"] = str(e)
        else:
            logger.info("No validation_context provided; skipping Stage 6 evaluation metrics")
        
        # Stage 7: Normalization metrics (requires circuit resources)
        if validation_context is not None and "p_succ" in metrics["stage_6_evaluation"]:
            p_succ = metrics["stage_6_evaluation"]["p_succ"]
            ci_lower = metrics["stage_6_evaluation"].get("p_succ_ci_lower")
            ci_upper = metrics["stage_6_evaluation"].get("p_succ_ci_upper")
            ci = (ci_lower, ci_upper)
            
            # Retention-based normalization (replaces eta_gate/eta_volume)
            if two_qubit_gates is not None:
                try:
                    # Log loss per 2-qubit gate
                    loss, loss_ci = calculate_log_loss_with_ci(p_succ, ci, two_qubit_gates)
                    metrics["stage_7_normalization"]["log_loss_per_2q"] = loss
                    metrics["stage_7_normalization"]["log_loss_per_2q_ci_lower"] = loss_ci[0]
                    metrics["stage_7_normalization"]["log_loss_per_2q_ci_upper"] = loss_ci[1]
                    
                    # Retention per 2-qubit gate
                    retention, retention_ci = calculate_retention_with_ci(p_succ, ci, two_qubit_gates)
                    metrics["stage_7_normalization"]["retention_per_2q"] = retention
                    metrics["stage_7_normalization"]["retention_per_2q_ci_lower"] = retention_ci[0]
                    metrics["stage_7_normalization"]["retention_per_2q_ci_upper"] = retention_ci[1]
                except Exception as e:
                    logger.warning(f"Failed to compute retention metrics (2q): {e}")
            else:
                logger.info("two_qubit_gates not provided; skipping retention metrics")
            
            # Volume-based retention
            if circuit_volume is not None:
                try:
                    from sudoku_nisq.metrics.calculators.retention_metrics import (
                        calculate_log_loss_per_volume,
                        calculate_retention_per_volume
                    )
                    
                    # Log loss per volume
                    loss = calculate_log_loss_per_volume(p_succ, circuit_volume)
                    metrics["stage_7_normalization"]["log_loss_per_volume"] = loss
                    
                    # Retention per volume
                    retention = calculate_retention_per_volume(p_succ, circuit_volume)
                    metrics["stage_7_normalization"]["retention_per_volume"] = retention
                    
                    # CI propagation for volume metrics
                    if ci_lower is not None and ci_upper is not None:
                        loss_lo = calculate_log_loss_per_volume(ci_upper, circuit_volume)  # reversed
                        loss_hi = calculate_log_loss_per_volume(ci_lower, circuit_volume)
                        metrics["stage_7_normalization"]["log_loss_per_volume_ci_lower"] = loss_lo
                        metrics["stage_7_normalization"]["log_loss_per_volume_ci_upper"] = loss_hi
                        
                        retention_lo = calculate_retention_per_volume(ci_lower, circuit_volume)
                        retention_hi = calculate_retention_per_volume(ci_upper, circuit_volume)
                        metrics["stage_7_normalization"]["retention_per_volume_ci_lower"] = retention_lo
                        metrics["stage_7_normalization"]["retention_per_volume_ci_upper"] = retention_hi
                except Exception as e:
                    logger.warning(f"Failed to compute retention metrics (volume): {e}")
            
            # Shot budget metrics (replaces eta_shot)
            try:
                shot_budgets = calculate_shot_budgets(p_succ, ci, reliability=0.95)
                metrics["stage_7_normalization"]["shots_detect_point"] = shot_budgets["shots_detect_point"]
                metrics["stage_7_normalization"]["shots_detect_pessimistic"] = shot_budgets["shots_detect_pessimistic"]
                metrics["stage_7_normalization"]["shots_detect_optimistic"] = shot_budgets["shots_detect_optimistic"]
                metrics["stage_7_normalization"]["shot_budget_reliability"] = shot_budgets["reliability"]
            except Exception as e:
                logger.warning(f"Failed to compute shot budgets: {e}")
            
            # Cost-normalized metrics (heuristic alternatives to retention)
            # Extract depth from kwargs if available
            depth = kwargs.get("depth") or kwargs.get("circuit_depth")
            
            if depth is not None and two_qubit_gates is not None:
                try:
                    # Use default weights α=1, β=1 (users can recompute externally with custom weights)
                    alpha, beta = 1.0, 1.0
                    
                    # Product-based normalization
                    eta_prod = calculate_eta_product(p_succ, depth, two_qubit_gates)
                    metrics["stage_7_normalization"]["eta_product"] = eta_prod
                    
                    # Weighted-sum normalization
                    eta_wsum = calculate_eta_weighted_sum(p_succ, depth, two_qubit_gates, alpha, beta)
                    metrics["stage_7_normalization"]["eta_weighted_sum"] = eta_wsum
                    
                    # Decay rate (exponential model)
                    k = calculate_decay_rate(p_succ, depth, two_qubit_gates, alpha, beta)
                    metrics["stage_7_normalization"]["decay_rate"] = k
                    
                    # Store weights for reproducibility
                    metrics["stage_7_normalization"]["cost_alpha"] = alpha
                    metrics["stage_7_normalization"]["cost_beta"] = beta
                    
                    logger.debug(f"Cost metrics: η_×={eta_prod}, η_+={eta_wsum}, k={k}")
                except Exception as e:
                    logger.warning(f"Failed to compute cost-normalized metrics: {e}")
            else:
                if depth is None:
                    logger.debug("depth not provided; skipping cost-normalized metrics")
                if two_qubit_gates is None:
                    logger.debug("two_qubit_gates not provided; skipping cost-normalized metrics")
        
        # Load existing metrics data
        existing_data: Dict[str, Any] = {}
        if self.storage_path.exists():
            try:
                existing_data = self._load_json()
            except Exception as e:
                logger.warning(f"Failed to load existing metrics: {e}")
        
        # Add new metrics
        existing_data[run_id] = metrics
        
        # Save atomically
        self._save_json(existing_data)
        
        logger.info(f"Recorded metrics for run_id={run_id}")
        return metrics
    
    def query(
        self,
        run_id: Optional[str] = None,
        backend_alias: Optional[str] = None,
        encoding: Optional[str] = None,
        **filters
    ) -> Union[List[Dict], Dict, None]:
        """Query metrics by run_id or filter by backend/encoding.
        
        Returns single run metrics if run_id specified, or list of metrics
        matching the filter criteria.
        
        :param run_id: Specific run ID to retrieve (returns single dict)
        :param backend_alias: Filter by backend alias (returns list)
        :param encoding: Filter by encoding type (returns list)
        :param filters: Additional filter criteria (reserved for future use)
        :returns: Single metrics dict, list of dicts, or None if not found
        
        Example:
            >>> # Get specific run
            >>> metrics = manager.query(run_id="abc123")
            >>> 
            >>> # Get all runs for a backend
            >>> runs = manager.query(backend_alias="aer_simulator")
        """
        if not self.storage_path.exists():
            return None if run_id else []
        
        try:
            data = self._load_json()
        except Exception as e:
            logger.warning(f"Failed to load metrics data: {e}")
            return None if run_id else []
        
        # Single run query
        if run_id:
            return data.get(run_id)
        
        # Filter by criteria (returns all if no filters)
        results = []
        for rid, metrics in data.items():
            # Skip aggregated entries
            if rid.startswith("aggregated_"):
                continue
            
            # Apply filters (currently no backend/encoding stored in metrics)
            # This would require cross-referencing with Stage 5 execution data
            # For Phase 4, we return all non-aggregated entries
            results.append(metrics)
        
        return results
    
    def compute_aggregated(
        self,
        run_ids: List[str],
        aggregation_key: Optional[str] = None,
        **filters
    ) -> Optional[Dict]:
        """Compute multi-run aggregation statistics.
        
        Aggregates metrics across multiple runs, computing mean, std, median,
        and IQR for key metrics like p_succ, SNR, and efficiency measures.
        
        :param run_ids: List of run IDs to aggregate
        :param aggregation_key: Optional key for storing aggregation (e.g., "backend_optlevel")
        :param filters: Additional filter criteria (reserved for future use)
        :returns: Dict with aggregated statistics or None if insufficient data
        :raises ValueError: If run_ids list is empty
        
        Example:
            >>> manager = MetricsMetadataManager(cache_base, puzzle_hash)
            >>> agg = manager.compute_aggregated(
            ...     run_ids=["run1", "run2", "run3"],
            ...     aggregation_key="aer_opt1"
            ... )
            >>> print(agg["p_succ_mean"])  # 0.856
        """
        if not run_ids:
            raise ValueError("run_ids list cannot be empty")
        
        if not self.storage_path.exists():
            logger.warning("No metrics file found; cannot compute aggregation")
            return None
        
        try:
            data = self._load_json()
        except Exception as e:
            logger.warning(f"Failed to load metrics data: {e}")
            return None
        
        # Collect metrics from specified runs.
        # Keep this to scalar metrics only (dict-valued @k metrics are not aggregated here).
        p_succ_values: List[float] = []
        valid_odds_values: List[float] = []
        peak_ratio_values: List[float] = []
        peak_gap_values: List[float] = []

        retention_per_2q_values: List[float] = []
        log_loss_per_2q_values: List[float] = []
        retention_per_volume_values: List[float] = []
        log_loss_per_volume_values: List[float] = []
        shots_detect_point_values: List[int] = []

        # Legacy keys (kept for backward compatibility with older cached metrics files)
        snr_values: List[float] = []
        eta_gate_values: List[float] = []
        eta_volume_values: List[float] = []
        eta_shot_values: List[float] = []

        # Cost-normalized metrics (optional)
        eta_product_values: List[float] = []
        eta_weighted_sum_values: List[float] = []
        decay_rate_values: List[float] = []
        
        for run_id in run_ids:
            if run_id not in data:
                logger.warning(f"Run ID {run_id} not found in metrics data")
                continue
            
            metrics = data[run_id]
            stage_6 = metrics.get("stage_6_evaluation", {})
            stage_7 = metrics.get("stage_7_normalization", {})
            
            if "p_succ" in stage_6:
                p_succ_values.append(stage_6["p_succ"])

            # Current Stage 6 scalar metrics
            if "valid_odds" in stage_6 and stage_6.get("valid_odds") is not None:
                valid_odds_values.append(stage_6["valid_odds"])
            if "peak_ratio" in stage_6 and stage_6.get("peak_ratio") is not None:
                peak_ratio_values.append(stage_6["peak_ratio"])
            if "peak_gap" in stage_6 and stage_6.get("peak_gap") is not None:
                peak_gap_values.append(stage_6["peak_gap"])

            # Current Stage 7 scalar metrics
            if "retention_per_2q" in stage_7 and stage_7.get("retention_per_2q") is not None:
                retention_per_2q_values.append(stage_7["retention_per_2q"])
            if "log_loss_per_2q" in stage_7 and stage_7.get("log_loss_per_2q") is not None:
                log_loss_per_2q_values.append(stage_7["log_loss_per_2q"])
            if "retention_per_volume" in stage_7 and stage_7.get("retention_per_volume") is not None:
                retention_per_volume_values.append(stage_7["retention_per_volume"])
            if "log_loss_per_volume" in stage_7 and stage_7.get("log_loss_per_volume") is not None:
                log_loss_per_volume_values.append(stage_7["log_loss_per_volume"])
            if "shots_detect_point" in stage_7 and stage_7.get("shots_detect_point") is not None:
                shots_detect_point_values.append(stage_7["shots_detect_point"])

            # Optional cost metrics
            if "eta_product" in stage_7 and stage_7.get("eta_product") is not None:
                eta_product_values.append(stage_7["eta_product"])
            if "eta_weighted_sum" in stage_7 and stage_7.get("eta_weighted_sum") is not None:
                eta_weighted_sum_values.append(stage_7["eta_weighted_sum"])
            if "decay_rate" in stage_7 and stage_7.get("decay_rate") is not None:
                decay_rate_values.append(stage_7["decay_rate"])

            # Legacy keys (older cached files)
            if "snr" in stage_6 and stage_6.get("snr") is not None:
                snr_values.append(stage_6["snr"])
            if "eta_gate" in stage_7 and stage_7.get("eta_gate") is not None:
                eta_gate_values.append(stage_7["eta_gate"])
            if "eta_volume" in stage_7 and stage_7.get("eta_volume") is not None:
                eta_volume_values.append(stage_7["eta_volume"])
            if "eta_shot" in stage_7 and stage_7.get("eta_shot") is not None:
                eta_shot_values.append(stage_7["eta_shot"])
        
        if not p_succ_values:
            logger.warning("No valid p_succ values found for aggregation")
            return None
        
        # Compute variability statistics
        aggregated = {
            "aggregation_type": "multi_run",
            "num_runs": len(p_succ_values),
            "run_ids": run_ids,
            "timestamp": datetime.now().isoformat()
        }
        
        # P_succ statistics
        if p_succ_values:
            stats = calculate_variability_stats(p_succ_values)
            aggregated["p_succ_mean"] = stats["mean"]
            aggregated["p_succ_std"] = stats["std"]
            aggregated["p_succ_median"] = stats["median"]
            aggregated["p_succ_q1"] = stats["q1"]
            aggregated["p_succ_q3"] = stats["q3"]
            aggregated["p_succ_iqr"] = stats["iqr"]

        # Stage 6 scalar statistics (current)
        if valid_odds_values:
            stats = calculate_variability_stats(valid_odds_values)
            aggregated["valid_odds_mean"] = stats["mean"]
            aggregated["valid_odds_std"] = stats["std"]
            aggregated["valid_odds_median"] = stats["median"]
            aggregated["valid_odds_q1"] = stats["q1"]
            aggregated["valid_odds_q3"] = stats["q3"]
            aggregated["valid_odds_iqr"] = stats["iqr"]

        if peak_ratio_values:
            stats = calculate_variability_stats(peak_ratio_values)
            aggregated["peak_ratio_mean"] = stats["mean"]
            aggregated["peak_ratio_std"] = stats["std"]

        if peak_gap_values:
            stats = calculate_variability_stats(peak_gap_values)
            aggregated["peak_gap_mean"] = stats["mean"]
            aggregated["peak_gap_std"] = stats["std"]

        # Stage 7 scalar statistics (current)
        if retention_per_2q_values:
            stats = calculate_variability_stats(retention_per_2q_values)
            aggregated["retention_per_2q_mean"] = stats["mean"]
            aggregated["retention_per_2q_std"] = stats["std"]

        if log_loss_per_2q_values:
            stats = calculate_variability_stats(log_loss_per_2q_values)
            aggregated["log_loss_per_2q_mean"] = stats["mean"]
            aggregated["log_loss_per_2q_std"] = stats["std"]

        if retention_per_volume_values:
            stats = calculate_variability_stats(retention_per_volume_values)
            aggregated["retention_per_volume_mean"] = stats["mean"]
            aggregated["retention_per_volume_std"] = stats["std"]

        if log_loss_per_volume_values:
            stats = calculate_variability_stats(log_loss_per_volume_values)
            aggregated["log_loss_per_volume_mean"] = stats["mean"]
            aggregated["log_loss_per_volume_std"] = stats["std"]

        if shots_detect_point_values:
            # shots_detect_* are ints; variability_stats will cast to float.
            stats = calculate_variability_stats(shots_detect_point_values)  # type: ignore[arg-type]
            aggregated["shots_detect_point_mean"] = stats["mean"]
            aggregated["shots_detect_point_std"] = stats["std"]
        
        # SNR statistics
        if snr_values:
            stats = calculate_variability_stats(snr_values)
            aggregated["snr_mean"] = stats["mean"]
            aggregated["snr_std"] = stats["std"]
        
        # Efficiency statistics
        if eta_gate_values:
            stats = calculate_variability_stats(eta_gate_values)
            aggregated["eta_gate_mean"] = stats["mean"]
            aggregated["eta_gate_std"] = stats["std"]
        
        if eta_volume_values:
            stats = calculate_variability_stats(eta_volume_values)
            aggregated["eta_volume_mean"] = stats["mean"]
            aggregated["eta_volume_std"] = stats["std"]
        
        if eta_shot_values:
            stats = calculate_variability_stats(eta_shot_values)
            aggregated["eta_shot_mean"] = stats["mean"]
            aggregated["eta_shot_std"] = stats["std"]

        # Cost metrics (optional)
        if eta_product_values:
            stats = calculate_variability_stats(eta_product_values)
            aggregated["eta_product_mean"] = stats["mean"]
            aggregated["eta_product_std"] = stats["std"]

        if eta_weighted_sum_values:
            stats = calculate_variability_stats(eta_weighted_sum_values)
            aggregated["eta_weighted_sum_mean"] = stats["mean"]
            aggregated["eta_weighted_sum_std"] = stats["std"]

        if decay_rate_values:
            stats = calculate_variability_stats(decay_rate_values)
            aggregated["decay_rate_mean"] = stats["mean"]
            aggregated["decay_rate_std"] = stats["std"]
        
        # Store aggregated results
        if aggregation_key:
            key = f"aggregated_{aggregation_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            data[key] = aggregated
            self._save_json(data)
            logger.info(f"Stored aggregated metrics under key: {key}")
        
        return aggregated
