import json
import pandas as pd
from typing import List, Optional, Any, Dict
import os
from sudoku_nisq import Sudoku

def generate_and_profile_puzzles(
    num_puzzles: int,
    num_missing_cells: int,
    solvers: List[str],
    optimisation_levels: Optional[List[int]] = None,
    include_transpiled: bool = True,
    save_to: str = "csv",
    filename: str = "sudoku_profiles.csv",
    verbose: bool = False
) -> pd.DataFrame:
    if optimisation_levels is None:
        optimisation_levels = [0, 1, 2, 3]

    # Load existing data if file exists, to prevent duplicate work
    existing_ids = set()
    if os.path.exists(filename):
        try:
            df_existing = pd.read_csv(filename)
            existing_ids = set(df_existing["puzzle_id"].unique())
        except Exception:
            if verbose:
                print("Warning: Failed to load existing file. It will be ignored.")

    records: List[Dict[str, Any]] = []

    for pid in range(num_puzzles):
        if pid in existing_ids:
            if verbose:
                print(f"Skipping puzzle {pid}, already processed.")
            continue

        try:
            s = Sudoku(num_missing_cells=num_missing_cells)
            mat = getattr(s.puzzle, "board", s.puzzle)
            puzzle_str = json.dumps(mat.tolist() if hasattr(mat, "tolist") else mat)
        except Exception as e:
            for key in solvers:
                records.append({
                    "puzzle_id": pid,
                    "solver": key,
                    "resource_type": "theoretical",
                    "n_qubits": None,
                    "MCX_gates": None,
                    "n_gates": None,
                    "depth": None,
                    "optimisation_level": None,
                    "backend": None,
                    "error": f"Puzzle generation failed: {str(e)}",
                    "puzzle": None,
                })
            continue

        for key in solvers:
            try:
                getattr(s, f"init_{key}")()
                solver = getattr(s, key)
                solver.main_circuit = solver.get_circuit()
            except Exception as e:
                records.append({
                    "puzzle_id": pid,
                    "solver": key,
                    "resource_type": "theoretical",
                    "n_qubits": None,
                    "MCX_gates": None,
                    "n_gates": None,
                    "depth": None,
                    "optimisation_level": None,
                    "backend": None,
                    "error": f"Solver init failed: {str(e)}",
                    "puzzle": puzzle_str,
                })
                continue

            # Theoretical Resources
            try:
                tr = solver.find_resources()
            except Exception as e:
                tr = {"n_qubits": None, "MCX_gates": None, "n_gates": None, "error": str(e)}

            records.append({
                "puzzle_id": pid,
                "solver": key,
                "resource_type": "theoretical",
                "n_qubits": tr.get("n_qubits"),
                "MCX_gates": tr.get("MCX_gates"),
                "n_gates": tr.get("n_gates"),
                "depth": None,
                "optimisation_level": None,
                "backend": None,
                "error": tr.get("error", None),
                "puzzle": puzzle_str,
            })

            # Transpiled Resources (Optional)
            if include_transpiled:
                try:
                    tx_list = solver.find_transpiled_resources(optimisation_levels=optimisation_levels)
                except Exception as e:
                    tx_list = [
                        {"optimisation_level": lvl, "backend": getattr(solver, "current_backend", None),
                         "n_qubits": None, "n_gates": None, "depth": None, "error": str(e)}
                        for lvl in optimisation_levels
                    ]

                for meta in tx_list:
                    records.append({
                        "puzzle_id": pid,
                        "solver": key,
                        "resource_type": "transpiled",
                        "n_qubits": meta.get("n_qubits"),
                        "MCX_gates": None,
                        "n_gates": meta.get("n_gates"),
                        "depth": meta.get("depth"),
                        "optimisation_level": meta.get("optimisation_level"),
                        "backend": meta.get("backend"),
                        "error": meta.get("error"),
                        "puzzle": puzzle_str,
                    })

        if verbose:
            print(f"  → Done puzzle {pid+1}/{num_puzzles}")

    df = pd.DataFrame(records)

    if save_to.lower() == "csv":
        # Append mode
        if os.path.exists(filename):
            df_existing = pd.read_csv(filename)
            df = pd.concat([df_existing, df], ignore_index=True)
        df.to_csv(filename, index=False)
        if verbose:
            print(f"Saved to {filename}")
    else:
        raise ValueError(f"Unsupported save_to: {save_to}")

    return df