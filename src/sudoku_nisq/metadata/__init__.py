"""Stage-specific metadata management system.

This module provides a modular architecture for tracking the 7-stage NISQ benchmark pipeline:

Stage 1: Instance Selection (𝓘, μ) → InstanceMetadataManager
Stage 2a: IR Construction → LogicalIRMetadataManager  
Stage 2b: IR Policy (𝖢_IR) → IRPolicyMetadataManager
Stage 3: Compilation (𝖢) → CompilationMetadataManager
Stage 4: Executable (Pulse) → ExecutableMetadataManager (minimal)
Stage 5: Execution (Runtime) → ExecutionMetadataManager
Stage 6-7: Evaluation + Normalization → MetricsMetadataManager

Each manager handles stage-specific data persistence, auto-extraction from domain objects,
and provides query interfaces for provenance tracking.

Usage:
    from sudoku_nisq.metadata import BenchmarkSession
    
    session = BenchmarkSession(puzzle_hash="abc123")
    session.execute_run(puzzle, backend="aer_simulator", shots=1024)
"""

from sudoku_nisq.metadata.base import StageMetadataManager
from sudoku_nisq.metadata.config import MetadataConfig
from sudoku_nisq.metadata.instance import InstanceMetadataManager
from sudoku_nisq.metadata.logical_ir import LogicalIRMetadataManager
from sudoku_nisq.metadata.ir_policy import IRPolicyMetadataManager
from sudoku_nisq.metadata.compilation import CompilationMetadataManager
from sudoku_nisq.metadata.executable import ExecutableMetadataManager
from sudoku_nisq.metadata.execution import ExecutionMetadataManager
from sudoku_nisq.metadata.metrics import MetricsMetadataManager
from sudoku_nisq.metadata.benchmark_session import BenchmarkSession

__all__ = [
    "StageMetadataManager",
    "MetadataConfig",
    "InstanceMetadataManager",
    "LogicalIRMetadataManager",
    "IRPolicyMetadataManager",
    "CompilationMetadataManager",
    "ExecutableMetadataManager",
    "ExecutionMetadataManager",
    "MetricsMetadataManager",
    "BenchmarkSession",
]
