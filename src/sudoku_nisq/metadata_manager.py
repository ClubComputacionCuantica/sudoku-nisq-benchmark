import os
import json
import csv
import tempfile
from pathlib import Path
from typing import Any, Mapping

class MetadataManager:
    """
    Manages JSON and CSV persistence for experiment metadata.
    
    Provides atomic operations for storing puzzle metadata, circuit resources,
    and automatic CSV logging for bulk experiments.
    """
    # CSV column order - consistent across all exports
    _CSV_FIELDNAMES = [
        "puzzle_hash", "size", "num_missing_cells",
        "solver_name", "encoding", "backend_alias", "opt_level",
        "main_n_qubits", "main_n_gates", "main_n_mcx_gates", "main_depth",
        "backend_n_qubits", "backend_n_gates", "backend_depth",
        "error"
    ]
    
    def __init__(
        self,
        cache_base: Path,
        puzzle_hash: str,
        *,
        sort_keys: bool = True,
        log_csv_path: Path | None = None,
    ):
        # Base folder (e.g. ".quantum_solver_cache")
        self.cache_base    = cache_base
        self.puzzle_hash   = puzzle_hash
        self.sort_keys     = sort_keys
        self.log_csv_path  = log_csv_path

        # Full path to metadata.json for this puzzle
        self.metadata_path = self.cache_base / self.puzzle_hash / "metadata.json"

        # In-memory store & dirty flag
        self._data: dict[str, Any] | None = None
        self._dirty = False

    def load(self) -> dict[str, Any]:
        """Lazily load (or initialize) the metadata dict."""
        if self._data is None:
            if self.metadata_path.exists():
                try:
                    self._data = json.loads(self.metadata_path.read_text())
                except json.JSONDecodeError:
                    # Corrupt file → start fresh
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
        
    def ensure_puzzle_fields(
        self,
        *,
        size: int,
        num_missing_cells: int,
        board: list[list[int]],
    ) -> None:
        """Set (or overwrite) the puzzle‐level fields if they differ."""
        md = self.load()
        fields = {
            "puzzle_hash":       self.puzzle_hash,
            "size":              size,
            "num_missing_cells": num_missing_cells,
            "board":             board,
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
        resources: Mapping[str, int | str],  # Allow error strings
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
            
            # Log to CSV if configured
            if self.log_csv_path:
                self._log_backend_resource_row(
                    solver_name, encoding, backend_alias, opt_level, resources
                )

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

    def _append_csv_row(self, row_data: dict[str, Any]) -> None:
        """Append a single row to the CSV log file if configured."""
        if not self.log_csv_path:
            return
            
        # Ensure parent directory exists
        self.log_csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Ensure row_data has all required fields (fill missing with None)
        complete_row = {field: row_data.get(field) for field in self._CSV_FIELDNAMES}
        
        # Check if file exists to determine if we need headers
        file_exists = self.log_csv_path.exists()
        
        with open(self.log_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self._CSV_FIELDNAMES)
            
            # Write header if this is a new file
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(complete_row)

    def _log_backend_resource_row(
        self,
        solver_name: str,
        encoding: str,
        backend_alias: str,
        opt_level: int,
        backend_resources: Mapping[str, int],
    ) -> None:
        """Create and append a single CSV row for this backend resource update."""
        md = self.load()
        
        # Get main circuit resources
        main_resources = (md.get("solvers", {})
                           .get(solver_name, {})
                           .get("encodings", {})
                           .get(encoding, {})
                           .get("main_circuit_resources", {}))
        
        # Create the row with consistent field order
        row_data = {
            "puzzle_hash": self.puzzle_hash,
            "size": md.get("size"),
            "num_missing_cells": md.get("num_missing_cells"),
            "solver_name": solver_name,
            "encoding": encoding,
            "backend_alias": backend_alias,
            "opt_level": opt_level,
            "main_n_qubits": main_resources.get("n_qubits"),
            "main_n_gates": main_resources.get("n_gates"),
            "main_n_mcx_gates": main_resources.get("n_mcx_gates"),
            "main_depth": main_resources.get("depth"),
            "backend_n_qubits": backend_resources.get("n_qubits"),
            "backend_n_gates": backend_resources.get("n_gates"),
            "backend_depth": backend_resources.get("depth"),
            "error": backend_resources.get("error")
        }
        
        self._append_csv_row(row_data)

    def _log_main_circuit_csv_row(self, solver_name: str, encoding: str, main_resources: Mapping[str, int]):
        """Log main circuit data as CSV row (no backend info)."""
        md = self.load()
        row_data = {
            "puzzle_hash": self.puzzle_hash,
            "size": md.get("size"),
            "num_missing_cells": md.get("num_missing_cells"),
            "solver_name": solver_name,
            "encoding": encoding,
            "backend_alias": None,  # Main circuit has no backend
            "opt_level": None,
            "main_n_qubits": main_resources.get("n_qubits"),
            "main_n_gates": main_resources.get("n_gates"),
            "main_n_mcx_gates": main_resources.get("n_mcx_gates"),
            "main_depth": main_resources.get("depth"),
            "backend_n_qubits": None,
            "backend_n_gates": None,
            "backend_depth": None,
            "error": None
        }
        self._append_csv_row(row_data)

    def export_full_metadata_csv(self, csv_path: Path) -> None:
        """
        Export all metadata for this puzzle as flattened CSV rows.
        This creates multiple rows (one per solver+encoding+backend+opt_level combination).
        
        Uses the same column order as the incremental CSV logging via _CSV_FIELDNAMES
        to ensure consistency across all CSV exports.
        """
        md = self.load()
        if not md:
            return
            
        rows = []
        base_fields = {
            "puzzle_hash": md.get("puzzle_hash"),
            "size": md.get("size"),
            "num_missing_cells": md.get("num_missing_cells"),
        }
        
        solvers = md.get("solvers", {})
        for solver_name, solver_data in solvers.items():
            encodings = solver_data.get("encodings", {})
            
            for encoding_name, encoding_data in encodings.items():
                main_resources = encoding_data.get("main_circuit_resources", {})
                backends = encoding_data.get("backends", {})
                
                if not backends:
                    # No backend data, just create a row for main circuit
                    row = {
                        **base_fields,
                        "solver_name": solver_name,
                        "encoding": encoding_name,
                        "backend_alias": None,
                        "opt_level": None,
                        "main_n_qubits": main_resources.get("n_qubits"),
                        "main_n_gates": main_resources.get("n_gates"),
                        "main_n_mcx_gates": main_resources.get("n_mcx_gates"),
                        "main_depth": main_resources.get("depth"),
                        "backend_n_qubits": None,
                        "backend_n_gates": None,
                        "backend_depth": None,
                    }
                    rows.append(row)
                else:
                    # Create rows for each backend+opt_level combination
                    for backend_alias, backend_data in backends.items():
                        for opt_level_str, opt_resources in backend_data.items():
                            try:
                                opt_level = int(opt_level_str)
                            except ValueError:
                                opt_level = opt_level_str
                                
                            row = {
                                **base_fields,
                                "solver_name": solver_name,
                                "encoding": encoding_name,
                                "backend_alias": backend_alias,
                                "opt_level": opt_level,
                                "main_n_qubits": main_resources.get("n_qubits"),
                                "main_n_gates": main_resources.get("n_gates"),
                                "main_n_mcx_gates": main_resources.get("n_mcx_gates"),
                                "main_depth": main_resources.get("depth"),
                                "backend_n_qubits": opt_resources.get("n_qubits"),
                                "backend_n_gates": opt_resources.get("n_gates"),
                                "backend_depth": opt_resources.get("depth"),
                            }
                            rows.append(row)
        
        if not rows:
            return
            
        # Write all rows to CSV
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self._CSV_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)

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
            encodings = solver_data.get("encodings", {})
            
            for encoding_name, encoding_data in encodings.items():
                encoding_summary = {
                    "main_circuit": encoding_data.get("main_circuit_resources", {}),
                    "backends": {}
                }
                
                backends = encoding_data.get("backends", {})
                for backend_alias, backend_data in backends.items():
                    encoding_summary["backends"][backend_alias] = backend_data
                
                summary["solvers"][solver_name][encoding_name] = encoding_summary
        
        return summary
