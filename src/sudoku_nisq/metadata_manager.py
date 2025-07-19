import os
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping
import warnings

class MetadataManager:
    """
    Manages JSON persistence for experiment metadata.

    Provides atomic operations for storing puzzle metadata and circuit resources.
    """
    __slots__ = (
        "cache_base",
        "puzzle_hash",
        "sort_keys",
        "metadata_path",
        "_data",
        "_dirty",
    )
    
    def __init__(
        self,
        cache_base: Path,
        puzzle_hash: str,
        *,
        sort_keys: bool = True,
    ):
        # Base folder (e.g. ".quantum_solver_cache")
        self.cache_base    = cache_base
        self.puzzle_hash   = puzzle_hash
        self.sort_keys     = sort_keys

        # Full path to metadata.json for this puzzle
        self.metadata_path = self.cache_base / self.puzzle_hash / "metadata.json"

        # In-memory store & dirty flag
        self._data: dict[str, Any] | None = None
        self._dirty = False

    def load(self) -> dict[str, Any]:
        """Lazily load (or initialize) the metadata dict.

        If the metadata file exists, it is read into memory. If the file is
        corrupt, a warning is issued, and the metadata is reset to an empty
        dictionary. If the file does not exist, an empty dictionary is initialized.
        """
        if self._data is None:
            if self.metadata_path.exists():
                try:
                    self._data = json.loads(self.metadata_path.read_text())
                except json.JSONDecodeError:
                    warnings.warn(
                        f"{self.metadata_path!s} is corrupt; resetting metadata to empty",
                        UserWarning,
                        stacklevel=2
                    )
                    self._data = {}
            else:
                self._data = {}
        return self._data

    def save(self) -> None:
        if not self._dirty:
            return

        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        dir_ = self.metadata_path.parent

        # Create a .tmp file, write+fsync, then rename it in place
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".tmp",
            dir=dir_, delete=False, encoding="utf-8"
        ) as tf:
            json.dump(self._data, tf, indent=2, sort_keys=self.sort_keys)
            tf.flush()
            os.fsync(tf.fileno())
            tmp_name = tf.name

        os.replace(tmp_name, self.metadata_path)
        self._dirty = False
        
    def unload(self) -> None:
        """
        Drop the in-memory metadata tree to free RAM.
        Next call to .load() will re-read from disk.
        """
        self._data = None
        self._dirty = False

    def ensure_puzzle_fields(
        self,
        *,
        size: int,
        num_missing_cells: int,
        board: list[list[int]],
    ) -> None:
        """Set (or overwrite) the puzzle‐level fields if they differ."""
        md = self.load()
        # Convert the board (matrix) to a string representation
        board_str = json.dumps(board)
        fields = {
            "puzzle_hash":       self.puzzle_hash,
            "size":              size,
            "num_missing_cells": num_missing_cells,
            "board":             board_str,  # Save as string
        }
        for k, v in fields.items():
            if md.get(k) != v:
                md[k] = v
                self._dirty = True

    def set_main_circuit_resources(
        self,
        solver_name: str,
        encoding: str,
        resources: Mapping[str, int],
    ) -> None:
        """
        Set solver's main_circuit_resources:
          {
            "n_qubits": …,
            "n_gates": …,
            "n_mcx_gates": …,
            "depth": …
          }
        """
        md = self.load()
        # Get (or create) the solver section
        solver_section = md.setdefault("solvers", {}) \
                        .setdefault(solver_name, {})
        # Get (or create) the encoding sub-section
        encoding_section = solver_section.setdefault("encodings", {}) \
                                        .setdefault(encoding, {})
        # Now write the main_circuit_resources if they’ve changed
        if encoding_section.get("main_circuit_resources") != resources:
            encoding_section["main_circuit_resources"] = dict(resources)
            self._dirty = True

    def set_backend_resources(
        self,
        solver_name: str,
        encoding: str,
        backend_alias: str,
        opt_level: int,
        resources: Mapping[str, int | str],
    ) -> None:
        """
        Set solver→backend→opt_level resources:
          {
            "n_qubits": …,
            "n_gates": …,
            "depth": …
          }
        """
        md = self.load()
        solvers = md.setdefault("solvers", {})
        sol_md  = solvers.setdefault(solver_name, {})
        encs    = sol_md.setdefault("encodings", {})
        enc_md  = encs.setdefault(encoding, {})
        backends = enc_md.setdefault("backends", {})
        be = backends.setdefault(backend_alias, {})

        lvl = str(opt_level)
        if be.get(lvl) != resources:
            be[lvl] = dict(resources)
            self._dirty = True

    def remove_solver(self, solver_name: str) -> None:
        """Remove all entries for a given solver."""
        md = self.load()
        solvers = md.get("solvers", {})
        if solver_name in solvers:
            solvers.pop(solver_name)
            self._dirty = True

    def get_solver_data(self, solver_name: str) -> dict[str, Any] | None:
        """Retrieve the raw metadata dict for one solver, or None."""
        md = self.load()
        return md.get("solvers", {}).get(solver_name)

    def get_resource_summary(self) -> dict[str, Any]:
        """
        Get a user-friendly summary of resource data.
        
        Returns:
            dict: Summary of puzzle and solver resource information
        """
        md = self.load()
        
        summary = {
            "puzzle_info": {
                "hash": md.get("puzzle_hash"),
                "size": md.get("size"),
                "num_missing_cells": md.get("num_missing_cells")
            },
            "solvers": {}
        }
        
        solvers = md.get("solvers", {})
        for solver_name, solver_data in solvers.items():
            summary["solvers"][solver_name] = {}
            # Type assertion to help mypy understand this is a dict
            solver_summary = summary["solvers"][solver_name]
            assert isinstance(solver_summary, dict)
            encodings = solver_data.get("encodings", {})
            
            for encoding_name, encoding_data in encodings.items():
                encoding_summary = {
                    "main_circuit": encoding_data.get("main_circuit_resources", {}),
                    "backends": {}
                }
                
                backends = encoding_data.get("backends", {})
                for backend_alias, backend_data in backends.items():
                    encoding_summary["backends"][backend_alias] = backend_data
                
                solver_summary[encoding_name] = encoding_summary
        
        return summary
