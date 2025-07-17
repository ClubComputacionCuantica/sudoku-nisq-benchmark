# type: ignore

import csv
import time
import json
import gc
import psutil
import argparse
from pathlib import Path
from typing import Dict, Any, List

from sudoku_nisq.q_sudoku import QSudoku
from sudoku_nisq.quantum_solver import QuantumSolver
from sudoku_nisq.backends import BackendManager
from sudoku_nisq.exact_cover_solver import ExactCoverQuantumSolver

# ─── CONFIG ─────────────────────────────────────────────────────

api_token = "MgUA1d64SPwFrqy-C1FnFYUY7lQG4B2F1k0xie5bUcW5"
instance = "crn:v1:bluemix:public:quantum-computing:us-east:a/53bccd1b6f1943a486285adb9d2dfa3f:c8244fef-9afe-428a-8e1a-08a6a03e61e9::"

BackendManager.authenticate_ibm(
    api_token=api_token,
    instance=instance
    )

# Map each QuantumSolver subclass to a list of encodings you want to test:
SOLVERS_CONFIG: Dict[type[QuantumSolver], List[str]] = {
    ExactCoverQuantumSolver: ["simple", "pattern"],
}

BACKENDS: List[str] = ['ibm_brisbane', 'ibm_fez', 'ibm_sherbrooke', 'ibm_torino', 'ibm_marrakesh', 'ibm_kingston']

for backend in BACKENDS:
    BackendManager.add_ibm_device(device=backend)

OPT_LEVELS: List[int]     = [0, 1, 2, 3]  # Optimization levels to test
SUBGRID_SIZES: List[int]  = [2]       # tests 4×4, 9×9
MISSING_CELLS: List[int]  = list(range(2, 16))  # Number of missing cells to test
SAMPLES_PER_COMBO: int    = 3 # How many samples to take per (size, missing) combo

CSV_PATH          = Path(f"transpile_feasibility_{SUBGRID_SIZES}_{SAMPLES_PER_COMBO}.csv")
JSON_REPORT_PATH  = Path(f"feasibility_report_{SUBGRID_SIZES}_{SAMPLES_PER_COMBO}.json")
CACHE_BASE        = Path("./cache")
CANONICALIZE      = False
CACHE_MAIN        = True
CACHE_TRANSPILED  = True

# Memory monitoring configuration
MEMORY_THRESHOLD  = 0.85  # Stop if memory usage exceeds 85%
MEMORY_CHECK_PAUSE = 5    # Seconds to wait when memory is high

def memory_is_safe(threshold: float = MEMORY_THRESHOLD) -> bool:
    """Check if current memory usage is below the threshold."""
    mem = psutil.virtual_memory()
    return mem.percent < threshold * 100

def wait_for_memory_clearance(threshold: float = MEMORY_THRESHOLD, max_wait_time: int = 300) -> bool:
    """
    Wait for memory usage to drop below threshold.
    Returns True if memory cleared, False if max_wait_time exceeded.
    """
    waited = 0
    while not memory_is_safe(threshold):
        if waited >= max_wait_time:
            print(f"Memory still high after {max_wait_time}s, continuing anyway...")
            return False
        
        mem = psutil.virtual_memory()
        print(f"High memory usage ({mem.percent:.1f}%), waiting {MEMORY_CHECK_PAUSE}s...")
        time.sleep(MEMORY_CHECK_PAUSE)
        waited += MEMORY_CHECK_PAUSE
        gc.collect()  # Force garbage collection
    return True

# What columns we record per transpile run
CSV_HEADER = [
    "puzzle_hash",
    "size",
    "missing",
    "solver",
    "encoding",
    "backend",
    "opt_level",
    "transpile_time_s",
    "success",
    "error",
]

def init_csv(path: Path) -> None:
    """Create CSV with header if it doesn't exist yet."""
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)

def write_csv_row(path: Path, row: Dict[str, Any]) -> None:
    """Append one row (matching CSV_HEADER) to our CSV file."""
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([row.get(col) for col in CSV_HEADER])

def generate_report(csv_path: Path) -> Dict[str, Any]:
    """
    Read the CSV at csv_path and compute:
      - For each (solver,encoding,backend,opt_level,size,missing):
          * average transpile_time_s, min/max times, success_fraction
      - For each (solver,encoding,backend,opt_level,size):
          * max safe missing cells (where success_fraction == 1.0)
    Returns a dict with keys: "detailed_stats", "max_safe_missing_cells", "time_analysis".
    """
    # data_map[(solver,encoding,backend,opt,size)] → { missing: [ (time, success), ... ] }
    data_map: Dict[tuple[str,str,str,int,int], Dict[int, List[tuple[float,bool]]]] = {}

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            solver   = r["solver"]
            encoding = r["encoding"]
            backend  = r["backend"]
            opt      = int(r["opt_level"])
            size     = int(r["size"])
            missing  = int(r["missing"])
            success  = r["success"].lower() in ("true", "1")
            t_str    = r["transpile_time_s"]
            t_val    = float(t_str) if t_str not in (None, "", "None") else None

            key = (solver, encoding, backend, opt, size)
            missing_map = data_map.setdefault(key, {})
            missing_map.setdefault(missing, []).append((t_val, success))

    detailed_stats: Dict[str, Any] = {}
    max_safe_missing: Dict[str, Any] = {}
    time_analysis: Dict[str, Any] = {}

    for (solver, enc, be, opt, size), missing_map in data_map.items():
        # Navigate into nested dicts for detailed stats
        det_solver = detailed_stats.setdefault(solver, {})
        det_enc    = det_solver.setdefault(enc, {})
        det_be     = det_enc.setdefault(be, {})
        det_opt    = det_be.setdefault(str(opt), {})
        det_opt[str(size)] = {}

        # Navigate for max safe missing cells
        safe_solver = max_safe_missing.setdefault(solver, {})
        safe_enc    = safe_solver.setdefault(enc, {})
        safe_be     = safe_enc.setdefault(be, {})
        safe_opt    = safe_be.setdefault(str(opt), {})

        # Navigate for time analysis
        time_solver = time_analysis.setdefault(solver, {})
        time_enc    = time_solver.setdefault(enc, {})
        time_be     = time_enc.setdefault(be, {})
        time_opt    = time_be.setdefault(str(opt), {})
        time_opt[str(size)] = {}

        max_safe_missing_cells: int | None = None
        all_times = []
        
        for missing in sorted(missing_map):
            entries   = missing_map[missing]
            times     = [t for t,ok in entries if t is not None and ok]  # Only successful times
            all_times.extend(times)
            succ_frac = sum(1 for _,ok in entries if ok) / len(entries)

            avg_t = sum(times)/len(times) if times else None
            min_t = min(times) if times else None
            max_t = max(times) if times else None

            det_opt[str(size)][str(missing)] = {
                "average_time_s": avg_t,
                "min_time_s": min_t,
                "max_time_s": max_t,
                "success_fraction": succ_frac,
                "total_attempts": len(entries),
                "successful_attempts": sum(1 for _,ok in entries if ok)
            }

            if succ_frac == 1.0:
                max_safe_missing_cells = missing

        safe_opt[str(size)] = max_safe_missing_cells
        
        # Overall time analysis for this configuration
        if all_times:
            time_opt[str(size)] = {
                "overall_average_time_s": sum(all_times)/len(all_times),
                "overall_min_time_s": min(all_times),
                "overall_max_time_s": max(all_times),
                "total_successful_transpilations": len(all_times)
            }

    return {
        "detailed_stats": detailed_stats,
        "max_safe_missing_cells": max_safe_missing,
        "time_analysis": time_analysis
    }

def generate_and_save_report(csv_path: Path = CSV_PATH, json_path: Path = JSON_REPORT_PATH) -> None:
    """Generate feasibility report from CSV and save to JSON."""
    if not csv_path.exists():
        print(f"Error: CSV file '{csv_path}' not found!")
        print("Run the script without --report-only first to generate data.")
        return
    
    print(f"Generating report from {csv_path}...")
    report = generate_report(csv_path)
    
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    print(f"Report saved to {json_path}")
    
    # Print a quick summary
    print("\n=== FEASIBILITY SUMMARY ===")
    for solver, solver_data in report["max_safe_missing_cells"].items():
        print(f"\nSolver: {solver}")
        for encoding, encoding_data in solver_data.items():
            print(f"  Encoding: {encoding}")
            for backend, backend_data in encoding_data.items():
                print(f"    Backend: {backend}")
                for opt_level, opt_data in backend_data.items():
                    print(f"      Opt {opt_level}:")
                    for size, max_missing in opt_data.items():
                        grid_size = int(size) * int(size)  # subgrid_size^2 = total grid size
                        if max_missing is not None:
                            print(f"        {grid_size}x{grid_size} grid: Max {max_missing} missing cells")
                        else:
                            print(f"        {grid_size}x{grid_size} grid: No successful transpilations")

    print("\n=== TRANSPILATION TIME ANALYSIS ===")
    for solver, solver_data in report["time_analysis"].items():
        print(f"\nSolver: {solver}")
        for encoding, encoding_data in solver_data.items():
            print(f"  Encoding: {encoding}")
            for backend, backend_data in encoding_data.items():
                print(f"    Backend: {backend}")
                for opt_level, opt_data in backend_data.items():
                    print(f"      Opt {opt_level}:")
                    for size, time_stats in opt_data.items():
                        grid_size = int(size) * int(size)  # subgrid_size^2 = total grid size
                        avg_time = time_stats["overall_average_time_s"]
                        min_time = time_stats["overall_min_time_s"]
                        max_time = time_stats["overall_max_time_s"]
                        count = time_stats["total_successful_transpilations"]
                        print(f"        {grid_size}x{grid_size} puzzle: {avg_time:.2f}s avg (min: {min_time:.2f}s, max: {max_time:.2f}s, {count} successes)")

# ─── MAIN WORKFLOW ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Sudoku NISQ Transpilation Feasibility Test")
    parser.add_argument("--report-only", action="store_true", 
                       help="Generate report from existing CSV without running benchmarks")
    args = parser.parse_args()
    
    if args.report_only:
        generate_and_save_report()
        return
    
    # Initial memory report
    mem = psutil.virtual_memory()
    print(f"Starting script with {mem.percent:.1f}% memory usage ({mem.used/1024**3:.1f}GB used, {mem.available/1024**3:.1f}GB available)")
    print(f"Memory threshold set to {MEMORY_THRESHOLD*100}%")
    
    # 1) Validate that backends exist
    for alias in BACKENDS:
        try:
            BackendManager.get(alias)
        except ValueError as e:
            available = BackendManager.aliases()
            raise ValueError(f"Backend '{alias}' not found. Available: {available}") from e

    # 2) Prepare CSV
    init_csv(CSV_PATH)

    # 3) Sweep puzzles
    for size in SUBGRID_SIZES:
        for missing in MISSING_CELLS:
            for sample_idx in range(SAMPLES_PER_COMBO):
                # Check memory before generating new puzzle
                if not memory_is_safe():
                    print(f"Memory usage high before generating puzzle (size={size}, missing={missing}, sample={sample_idx})")
                    if not wait_for_memory_clearance():
                        print("Memory still high, but continuing with puzzle generation...")

                # 3.1 Generate puzzle
                qs = QSudoku.generate(
                    subgrid_size=size,
                    num_missing_cells=missing,
                    canonicalize=CANONICALIZE,
                    cache_base=CACHE_BASE,
                    csv_path=None  # disable QSudoku's own CSV logging
                )
                puzzle_hash = qs.get_hash()

                # 3.2 Attach backends once
                for alias in BACKENDS:
                    qs.attach_backend(alias)

                # 3.3 Per solver / encoding
                for SolverCls, encodings in SOLVERS_CONFIG.items():
                    for enc in encodings:
                        qs.set_solver(SolverCls, encoding=enc, store_transpiled=CACHE_TRANSPILED)

                        # 3.3.a Build circuit
                        # Check memory before building circuit
                        if not memory_is_safe():
                            if not wait_for_memory_clearance():
                                print("Skipping circuit build due to persistent high memory usage")
                                # Record failure for all backends/opts
                                for alias in BACKENDS:
                                    for opt in OPT_LEVELS:
                                        write_csv_row(CSV_PATH, {
                                            "puzzle_hash":      puzzle_hash,
                                            "size":             size,
                                            "missing":          missing,
                                            "solver":           SolverCls.__name__,
                                            "encoding":         enc,
                                            "backend":          alias,
                                            "opt_level":        opt,
                                            "transpile_time_s": None,
                                            "success":          False,
                                            "error":            "memory_limit_exceeded"
                                        })
                                qs.drop_solver()
                                continue

                        try:
                            qs.build_circuit()
                        except Exception as build_err:
                            # If build fails, record a failure row for every backend+opt
                            for alias in BACKENDS:
                                for opt in OPT_LEVELS:
                                    write_csv_row(CSV_PATH, {
                                        "puzzle_hash":      puzzle_hash,
                                        "size":             size,
                                        "missing":          missing,
                                        "solver":           SolverCls.__name__,
                                        "encoding":         enc,
                                        "backend":          alias,
                                        "opt_level":        opt,
                                        "transpile_time_s": None,
                                        "success":          False,
                                        "error":            f"build_error: {build_err}"
                                    })
                            qs.drop_solver()
                            continue

                        # 3.3.b Transpile × backend × opt_level
                        for alias in BACKENDS:
                            for opt in OPT_LEVELS:
                                # Check memory before transpilation
                                if not memory_is_safe():
                                    if not wait_for_memory_clearance():
                                        print(f"Skipping transpile for {alias} opt={opt} due to high memory")
                                        write_csv_row(CSV_PATH, {
                                            "puzzle_hash":      puzzle_hash,
                                            "size":             size,
                                            "missing":          missing,
                                            "solver":           SolverCls.__name__,
                                            "encoding":         enc,
                                            "backend":          alias,
                                            "opt_level":        opt,
                                            "transpile_time_s": None,
                                            "success":          False,
                                            "error":            "memory_limit_exceeded"
                                        })
                                        continue

                                start = time.perf_counter()
                                err: str | None = None
                                try:
                                    qs.transpile(alias, [opt])
                                except Exception as e:
                                    err = str(e)
                                elapsed = time.perf_counter() - start

                                write_csv_row(CSV_PATH, {
                                    "puzzle_hash":      puzzle_hash,
                                    "size":             size,
                                    "missing":          missing,
                                    "solver":           SolverCls.__name__,
                                    "encoding":         enc,
                                    "backend":          alias,
                                    "opt_level":        opt,
                                    "transpile_time_s": elapsed,
                                    "success":          (err is None),
                                    "error":            err
                                })

                        qs.drop_solver()

                # 3.4 Cleanup memory
                del qs
                gc.collect()
                
                # Report memory usage periodically
                if sample_idx == 0:  # Report at start of each missing cells combination
                    mem = psutil.virtual_memory()
                    print(f"Memory usage after size={size}, missing={missing}: {mem.percent:.1f}% ({mem.used/1024**3:.1f}GB used)")

    # 4) Post‐process CSV → JSON report
    generate_and_save_report()

if __name__ == "__main__":
    main()