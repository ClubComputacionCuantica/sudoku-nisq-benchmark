"""Calculators package - individual metric computation functions."""

# Success metrics (existing)
from .success_metrics import (
    calculate_p_succ,
    calculate_distinct_solutions,
    count_valid_shots
)

# Ranking metrics (Phase 4)
from .ranking_metrics import (
    calculate_top_k_valid_mass,
    calculate_precision_at_k,
    calculate_recall_at_k
)

# Mass-weighted ranking metrics (improved)
from .mass_ranking_metrics import (
    calculate_mass_precision_at_k,
    calculate_valid_mass_capture_at_k
)

# Statistical metrics (Phase 4)
from .statistical_metrics import (
    calculate_clopper_pearson_ci,
    calculate_snr,  # Deprecated, use calculate_valid_odds
    calculate_variability_stats
)

# Odds-based discrimination metrics (replaces SNR)
from .odds_metrics import (
    calculate_valid_odds,
    calculate_valid_odds_with_ci,
    transform_ci_monotone
)

# Peak-based discrimination metrics
from .peak_metrics import (
    calculate_peak_metrics
)

# Retention-based normalization metrics (improved efficiency)
from .retention_metrics import (
    calculate_log_loss_per_2q,
    calculate_retention_per_2q,
    calculate_log_loss_per_volume,
    calculate_retention_per_volume,
    calculate_retention_with_ci,
    calculate_log_loss_with_ci
)

# Shot budget metrics (replaces eta_shot)
from .shot_budget_metrics import (
    shots_to_detect,
    calculate_shot_budgets
)

# Cost-normalized metrics (heuristic alternatives)
from .cost_metrics import (
    calculate_eta_product,
    calculate_eta_weighted_sum,
    calculate_decay_rate,
    fit_cost_weights
)

__all__ = [
    # Success metrics
    "calculate_p_succ",
    "calculate_distinct_solutions",
    "count_valid_shots",
    # Count-based ranking metrics
    "calculate_top_k_valid_mass",
    "calculate_precision_at_k",
    "calculate_recall_at_k",
    # Mass-weighted ranking metrics
    "calculate_mass_precision_at_k",
    "calculate_valid_mass_capture_at_k",
    # Statistical metrics
    "calculate_clopper_pearson_ci",
    "calculate_snr",  # Deprecated
    "calculate_variability_stats",
    # Odds metrics
    "calculate_valid_odds",
    "calculate_valid_odds_with_ci",
    "transform_ci_monotone",
    # Peak metrics
    "calculate_peak_metrics",
    # Retention-based normalization
    "calculate_log_loss_per_2q",
    "calculate_retention_per_2q",
    "calculate_log_loss_per_volume",
    "calculate_retention_per_volume",
    "calculate_retention_with_ci",
    "calculate_log_loss_with_ci",
    # Shot budget metrics
    "shots_to_detect",
    "calculate_shot_budgets",
    # Cost-normalized metrics
    "calculate_eta_product",
    "calculate_eta_weighted_sum",
    "calculate_decay_rate",
    "fit_cost_weights",
]
