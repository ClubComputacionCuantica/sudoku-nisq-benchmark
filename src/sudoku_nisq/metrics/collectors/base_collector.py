"""Abstract base class for metadata collectors in the benchmarking pipeline.

This module defines the MetadataCollector interface for collecting comprehensive
metadata from quantum backends, circuits, and execution results. Collectors are
stateful classes designed for post-execution analysis in benchmarking workflows.

Note: This is separate from src/sudoku_nisq/metadata/collectors/ which provides
lightweight function-based Stage 5 hardware snapshots during execution.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from sudoku_nisq.metrics.data_models import (
    HardwareMetadata,
    CompilationMetadata,
    ExecutionResult,
)


class MetadataCollector(ABC):
    """Abstract base class for collecting benchmarking metadata.
    
    Collectors are responsible for extracting comprehensive metadata from quantum
    backends, circuits, and execution results for benchmarking analysis. Each
    collector implementation handles a specific quantum SDK/provider.
    
    Key Distinction:
        - Stage 5 collectors (src/sudoku_nisq/metadata/collectors/): Lightweight
          function-based hardware snapshots captured during execution
        - Benchmarking collectors (this module): Stateful class-based extractors
          for comprehensive post-execution analysis including hardware, compilation,
          execution, and circuit volume metadata
    
    Usage:
        collector = QiskitMetadataCollector()
        
        # Collect hardware calibration data
        hw_metadata = collector.collect_hardware_metadata(backend)
        
        # Extract compilation metadata
        comp_metadata = collector.extract_compilation_metadata(
            circuit, compiled_circuit, backend
        )
        
        # Extract execution results
        exec_result = collector.extract_execution_result(
            result, backend, compiled_circuit
        )
        
        # Calculate circuit volume
        volume = collector.calculate_circuit_volume(compiled_circuit)
    """
    
    @abstractmethod
    def collect_hardware_metadata(self, backend: Any) -> HardwareMetadata:
        """Collect hardware calibration and characterization metadata.
        
        Args:
            backend: Quantum backend object (provider-specific type)
        
        Returns:
            HardwareMetadata dataclass with calibration data, error rates,
            coherence times, and backend characterization
        
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        pass
    
    @abstractmethod
    def extract_compilation_metadata(
        self,
        circuit: Any,
        compiled_circuit: Any,
        backend: Optional[Any] = None,
    ) -> CompilationMetadata:
        """Extract metadata from circuit compilation/transpilation.
        
        Analyzes the compilation process to extract:
        - Initial and final qubit layouts
        - Pre/post compilation gate counts and depth
        - Optimization level and routing metrics (SWAP counts)
        - SDK versions and compilation parameters
        
        Args:
            circuit: Original circuit before compilation (SDK-specific type)
            compiled_circuit: Transpiled circuit (SDK-specific type)
            backend: Optional backend used for compilation (for constraint info)
        
        Returns:
            CompilationMetadata dataclass with layout, gate counts, depth,
            optimization level, and compilation parameters
        
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        pass
    
    @abstractmethod
    def extract_execution_result(
        self,
        result: Any,
        backend: Any,
        compiled_circuit: Any,
        job: Optional[Any] = None,
    ) -> ExecutionResult:
        """Extract execution metadata from result object.
        
        Normalizes provider-specific result objects into a common ExecutionResult
        format. Handles multiple result formats (legacy Result, SamplerV2, etc.).
        
        Args:
            result: Execution result object (provider-specific type)
            backend: Backend used for execution
            compiled_circuit: Compiled circuit that was executed
            job: Optional job object for extracting job_id (V2 primitives)
        
        Returns:
            ExecutionResult dataclass with counts, shots, timing, backend info,
            circuit metrics, and optional job_id/run_id/metadata fields
        
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        pass
    
    @abstractmethod
    def calculate_circuit_volume(self, circuit: Any) -> Optional[int]:
        """Calculate circuit volume (sum of active gates per layer).
        
        Circuit volume is computed by analyzing the DAG layer structure:
        - Convert circuit to DAG representation
        - For each layer (depth-1 slice), count active gates
        - Sum gate counts across all layers
        - Exclude barriers, measurements, and directives
        
        For circuits with unknown control flow (while loops with unknown iteration
        counts), this returns None to avoid misleading approximations.
        
        Args:
            circuit: Compiled circuit (SDK-specific type)
        
        Returns:
            Circuit volume (int) if computable, None if circuit has unknown
            control flow or volume cannot be determined
        
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        pass
