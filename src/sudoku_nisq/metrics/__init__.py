"""
Metrics calculation package for sudoku-nisq benchmarking.

This package provides modular, composable metrics for evaluating quantum
algorithm performance according to modern benchmarking standards.

Key components:
- data_models: Data structures for results and metadata
- calculators: Individual metric calculation functions
- collectors: Provider-specific metadata collectors
- aggregators: Multi-run aggregation
- reporters: Export and visualization
- benchmarking: High-level benchmark orchestration
"""

# Version following the architecture design
__version__ = "0.1.0-alpha"

# Core data models
from .data_models import (
    ExecutionResult,
    HardwareMetadata,
    CompilationMetadata,
    ValidationContext,
    MetricsResult,
)

__all__ = [
    "ExecutionResult",
    "HardwareMetadata",
    "CompilationMetadata",
    "ValidationContext",
    "MetricsResult",
]

# Calculators will be imported as they're implemented
# from .calculators import (
#     SuccessMetricsCalculator,
#     RankingMetricsCalculator,
#     StatisticalMetricsCalculator,
#     EfficiencyMetricsCalculator,
#     VariabilityMetricsCalculator,
# )

# TODO: Add remaining imports as modules are implemented
