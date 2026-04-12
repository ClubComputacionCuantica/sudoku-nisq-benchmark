"""Stage 2b: IR Policy metadata manager.

Tracks solver configuration and SDK versions that affect IR representation.
"""

from pathlib import Path
from typing import Dict, List, Union
from datetime import datetime, timezone
from sudoku_nisq.metadata.base import StageMetadataManager
from sudoku_nisq.metadata.config import MetadataConfig


class IRPolicyMetadataManager(StageMetadataManager):
    """Manages Stage 2b: IR Policy (𝖢_IR) metadata.
    
    Tracks solver configuration that affects circuit IR:
    - Solver options (decompose_cnz, track_memory, etc.)
    - SDK versions (pytket 1.31.1, qiskit 1.0.0, etc.)
    - Gate decomposition policies
    
    Storage: {puzzle_hash}/stage_2b_ir_policy.json
    Structure: {solver_name: {encoding: {sdk_version, solver_options, ...}}}
    """
    
    @property
    def stage_number(self) -> int:
        return 2  # Stage 2b
    
    def __init__(self, cache_base: Path, puzzle_hash: str):
        """Initialize IR policy metadata manager.
        
        :param cache_base: Base cache directory (e.g., .quantum_solver_cache)
        :param puzzle_hash: Puzzle identifier (determines storage subdirectory)
        """
        self.cache_base = Path(cache_base)
        self.puzzle_hash = puzzle_hash
        self._storage_path = self.cache_base / puzzle_hash / MetadataConfig.STAGE_2B_IR_POLICY
    
    @property
    def storage_path(self) -> Path:
        return self._storage_path
    
    def record(self, **kwargs) -> str:
        """Record IR policy configuration for solver/encoding pair.
        
        :param solver_name: Solver class name (e.g., "ExactCoverQuantumSolver")
        :param encoding: Encoding type (e.g., "simple", "pattern")
        :param sdk: SDK used for circuit construction ("pytket", "qiskit", "braket")
        :param sdk_version: SDK version string (e.g., "1.31.1")
        :param solver_options: Dict of solver-specific options (decompose_cnz, track_memory, etc.)
        :param gate_decomposition_policy: Gate decomposition rules (optional)
        :param timestamp: ISO timestamp (auto-generated if not provided)
        :returns: Empty string (policy metadata doesn't have unique ID)
        
        Example::
        
            manager = IRPolicyMetadataManager(cache_base, puzzle_hash)
            manager.record(
                solver_name="ExactCoverQuantumSolver",
                encoding="simple",
                sdk="pytket",
                sdk_version="1.31.1",
                solver_options={"decompose_cnz": True, "track_memory": False}
            )
        """
        solver_name = kwargs.get("solver_name")
        encoding = kwargs.get("encoding")
        
        if not solver_name:
            raise ValueError("solver_name is required for IR policy recording")
        if not encoding:
            raise ValueError("encoding is required for IR policy recording")
        
        # Load existing policy or create new
        policy_data = self._load_json() or {}
        
        # Initialize solver entry if not exists
        if solver_name not in policy_data:
            policy_data[solver_name] = {}
        
        # Build policy entry for this encoding
        policy_entry = {
            "sdk": kwargs.get("sdk"),
            "sdk_version": kwargs.get("sdk_version"),
            "solver_options": kwargs.get("solver_options", {}),
            "gate_decomposition_policy": kwargs.get("gate_decomposition_policy"),
            "timestamp": kwargs.get("timestamp", datetime.now(timezone.utc).isoformat()),
        }
        
        # Store under solver_name → encoding
        policy_data[solver_name][encoding] = policy_entry
        self._save_json(policy_data)
        
        return ""  # Policy metadata doesn't generate unique ID
    
    def query(self, **filters) -> Union[List[Dict], Dict, None]:
        """Query IR policy by solver/encoding.
        
        :param solver_name: Solver class name (required for filtering)
        :param encoding: Encoding type (optional, returns all encodings if not specified)
        :returns: Policy dict for specified solver/encoding, or None if not found
        
        Example::
        
            # Get policy for specific solver/encoding
            policy = manager.query(
                solver_name="ExactCoverQuantumSolver",
                encoding="simple"
            )
            
            # Get all encodings for a solver
            all_policies = manager.query(solver_name="ExactCoverQuantumSolver")
        """
        solver_name = filters.get("solver_name")
        if not solver_name:
            raise ValueError("solver_name is required for IR policy query")
        
        policy_data = self._load_json() or {}
        
        # Check if solver exists
        if solver_name not in policy_data:
            return None
        
        # If encoding specified, return specific policy
        encoding = filters.get("encoding")
        if encoding:
            return policy_data[solver_name].get(encoding)
        
        # Otherwise return all encodings for this solver
        return policy_data[solver_name]
