"""Multi-run metrics aggregation for benchmarking analysis."""

from typing import List, Optional
from datetime import datetime
import logging

from sudoku_nisq.metrics.data_models import MetricsResult, AggregatedMetrics
from sudoku_nisq.metrics.calculators.statistical_metrics import (
    calculate_variability_stats,
)

logger = logging.getLogger(__name__)


class MultiRunAggregator:
    """Aggregates metrics across multiple benchmark runs.
    
    Computes statistical aggregations (mean, std, median, Q1, Q3, IQR) for
    each numeric metric field in MetricsResult objects. Used for multi-run
    benchmarking to capture variability and assess statistical significance.
    
    Usage:
        # Run multiple experiments
        results = []
        for seed in range(5):
            result = run_experiment(seed=seed)
            results.append(result)
        
        # Aggregate
        aggregated = MultiRunAggregator.aggregate(results)
        print(f"Mean p_succ: {aggregated.p_succ['mean']:.4f}")
        print(f"Std p_succ: {aggregated.p_succ['std']:.4f}")
    
    Notes:
        - Uses Bessel's correction for standard deviation (n-1)
        - Returns None for metrics with insufficient data
        - Handles missing values gracefully (skips None entries)
        - Fields with all None values result in None for all stats
    """
    
    # Metric fields to aggregate (from MetricsResult)
    METRIC_FIELDS = [
        # Success metrics
        "p_succ",
        "p_succ_ci_lower",
        "p_succ_ci_upper",
        "distinct_valid_solutions",  # Maps to distinct_valid in AggregatedMetrics
        # Ranking metrics (dict-valued)
        "top_k_valid_mass",
        "precision_at_k",
        "recall_at_k",
        "mass_precision_at_k",
        "valid_mass_capture_at_k",
        # Odds metrics
        "valid_odds",
        "valid_odds_ci_lower",
        "valid_odds_ci_upper",
        # Peak metrics
        "peak_ratio",
        "peak_gap",
        "p_best_valid",
        "p_best_invalid",
        # Retention metrics
        "retention_per_2q",
        "retention_per_2q_ci_lower",
        "retention_per_2q_ci_upper",
        "retention_per_volume",
        "retention_per_volume_ci_lower",
        "retention_per_volume_ci_upper",
        # Log loss
        "log_loss_per_2q",
        "log_loss_per_2q_ci_lower",
        "log_loss_per_2q_ci_upper",
        "log_loss_per_volume",
        "log_loss_per_volume_ci_lower",
        "log_loss_per_volume_ci_upper",
        # Shot budget
        "shots_detect_point",
        "shots_detect_pessimistic",
        "shots_detect_optimistic",
        # Cost-normalized (heuristic)
        "eta_product",
        "eta_weighted_sum",
        "decay_rate",
        # Deprecated (include for backward compat)
        "snr",
        "eta_gate",
        "eta_volume",
        "eta_shot",
    ]
    
    # Map from MetricsResult field names to AggregatedMetrics field names
    FIELD_NAME_MAP = {
        "distinct_valid_solutions": "distinct_valid",
        "valid_odds_ci_lower": "valid_odds_lower",
        "valid_odds_ci_upper": "valid_odds_upper",
        "log_loss_per_2q": "log_loss",
        "retention_per_2q_ci_lower": "retention_lower",
        "retention_per_2q_ci_upper": "retention_upper",
        "log_loss_per_2q_ci_lower": "log_loss_lower",
        "log_loss_per_2q_ci_upper": "log_loss_upper",
        "retention_per_volume_ci_lower": "retention_volume_lower",
        "retention_per_volume_ci_upper": "retention_volume_upper",
        "log_loss_per_volume": "log_loss_per_volume",
        "log_loss_per_volume_ci_lower": "log_loss_volume_lower",
        "log_loss_per_volume_ci_upper": "log_loss_volume_upper",
    }
    
    @classmethod
    def aggregate(
        cls,
        results: List[MetricsResult],
        notes: Optional[str] = None,
    ) -> AggregatedMetrics:
        """Aggregate metrics across multiple benchmark runs.
        
        Args:
            results: List of MetricsResult objects from multiple runs
            notes: Optional notes about the aggregation process
        
        Returns:
            AggregatedMetrics with statistical aggregations for each metric
        
        Raises:
            ValueError: If results list is empty
        
        Example:
            >>> results = [run_1_metrics, run_2_metrics, run_3_metrics]
            >>> agg = MultiRunAggregator.aggregate(results)
            >>> print(f"p_succ: {agg.p_succ['mean']:.4f} ± {agg.p_succ['std']:.4f}")
        """
        if not results:
            raise ValueError("Cannot aggregate empty results list")
        
        n_runs = len(results)
        logger.info(f"Aggregating {n_runs} benchmark runs")
        
        # Dictionary to hold aggregated metrics
        aggregated_fields = {}
        
        # Iterate over each metric field
        for field_name in cls.METRIC_FIELDS:
            # Collect values from all results (skip None)
            values = []
            for result in results:
                value = getattr(result, field_name, None)
                if value is not None:
                    values.append(value)
            
            # Compute statistics
            if values:
                try:
                    # Handle dict-valued metrics (e.g., per-k metrics)
                    if isinstance(values[0], dict):
                        per_key_stats = {}
                        all_keys = set()
                        for value_dict in values:
                            all_keys.update(value_dict.keys())

                        for key in sorted(all_keys, key=str):
                            key_values = [
                                v.get(key)
                                for v in values
                                if isinstance(v, dict) and v.get(key) is not None
                            ]
                            per_key_stats[key] = (
                                calculate_variability_stats(key_values)
                                if key_values
                                else None
                            )

                        mapped_name = cls.FIELD_NAME_MAP.get(field_name, field_name)
                        aggregated_fields[mapped_name] = per_key_stats
                    else:
                        stats = calculate_variability_stats(values)
                        # Use mapped name if it exists, otherwise use original
                        mapped_name = cls.FIELD_NAME_MAP.get(field_name, field_name)
                        aggregated_fields[mapped_name] = stats
                except Exception as e:
                    logger.warning(
                        f"Failed to compute stats for {field_name}: {e}"
                    )
                    mapped_name = cls.FIELD_NAME_MAP.get(field_name, field_name)
                    aggregated_fields[mapped_name] = None  # type: ignore[assignment]
            else:
                # No valid values for this field
                mapped_name = cls.FIELD_NAME_MAP.get(field_name, field_name)
                aggregated_fields[mapped_name] = None  # type: ignore[assignment]
        
        # Create AggregatedMetrics object
        return AggregatedMetrics(
            n_runs=n_runs,
            timestamp=datetime.utcnow(),
            aggregation_notes=notes,
            # Success metrics
            p_succ=aggregated_fields.get("p_succ"),
            distinct_valid=aggregated_fields.get("distinct_valid"),
            # Ranking metrics
            top_k_valid_mass=aggregated_fields.get("top_k_valid_mass"),
            precision_at_k=aggregated_fields.get("precision_at_k"),
            recall_at_k=aggregated_fields.get("recall_at_k"),
            mass_precision_at_k=aggregated_fields.get("mass_precision_at_k"),
            valid_mass_capture_at_k=aggregated_fields.get("valid_mass_capture_at_k"),
            # Odds metrics
            valid_odds=aggregated_fields.get("valid_odds"),
            valid_odds_lower=aggregated_fields.get("valid_odds_lower"),
            valid_odds_upper=aggregated_fields.get("valid_odds_upper"),
            # Peak metrics
            peak_ratio=aggregated_fields.get("peak_ratio"),
            peak_gap=aggregated_fields.get("peak_gap"),
            p_best_valid=aggregated_fields.get("p_best_valid"),
            p_best_invalid=aggregated_fields.get("p_best_invalid"),
            # Retention metrics
            retention_per_2q=aggregated_fields.get("retention_per_2q"),
            retention_lower=aggregated_fields.get("retention_lower"),
            retention_upper=aggregated_fields.get("retention_upper"),
            retention_per_volume=aggregated_fields.get("retention_per_volume"),
            retention_volume_lower=aggregated_fields.get("retention_volume_lower"),
            retention_volume_upper=aggregated_fields.get("retention_volume_upper"),
            # Log loss
            log_loss=aggregated_fields.get("log_loss"),
            log_loss_lower=aggregated_fields.get("log_loss_lower"),
            log_loss_upper=aggregated_fields.get("log_loss_upper"),
            log_loss_per_volume=aggregated_fields.get("log_loss_per_volume"),
            log_loss_volume_lower=aggregated_fields.get("log_loss_volume_lower"),
            log_loss_volume_upper=aggregated_fields.get("log_loss_volume_upper"),
            # Shot budget
            shots_detect_point=aggregated_fields.get("shots_detect_point"),
            shots_detect_pessimistic=aggregated_fields.get("shots_detect_pessimistic"),
            shots_detect_optimistic=aggregated_fields.get("shots_detect_optimistic"),
            # Deprecated
            snr=aggregated_fields.get("snr"),
            eta_gate=aggregated_fields.get("eta_gate"),
            eta_volume=aggregated_fields.get("eta_volume"),
            eta_shot=aggregated_fields.get("eta_shot"),
        )
