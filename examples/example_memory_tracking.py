"""Example demonstrating memory tracking during circuit construction (advanced/dev feature).

Memory tracking is disabled by default for production use. This example shows how
to enable it for development, profiling, and debugging purposes.
"""

from sudoku_nisq import QSudoku, ExactCoverQuantumSolver

def main():
    print("=" * 70)
    print("Memory Tracking Example (Advanced/Dev Feature)")
    print("=" * 70)
    print("\nNote: Memory tracking is disabled by default.")
    print("Enable with track_memory=True for development/profiling.")
    
    # Create a simple 2x2 Sudoku puzzle
    puzzle = QSudoku.generate(size=4, num_missing_cells=2)
    
    # Enable memory tracking (advanced feature, disabled by default)
    puzzle.set_solver(ExactCoverQuantumSolver, encoding='simple', track_memory=True)
    
    print(f"\nProblem size: {puzzle._solver.s_size} subsets, {puzzle._solver.u_size} universe elements")
    
    # Build circuit (memory tracking happens if enabled)
    print("\nBuilding circuit with memory tracking enabled...")
    puzzle.build_circuit()
    
    # Get memory usage statistics
    memory_usage = puzzle._solver.get_memory_usage()
    
    if memory_usage:
        print("\n" + "=" * 70)
        print("Memory Usage Statistics")
        print("=" * 70)
        print(f"Initial memory:  {memory_usage['initial_mb']:.2f} MB")
        print(f"Peak memory:     {memory_usage['peak_mb']:.2f} MB")
        print(f"Current memory:  {memory_usage['current_mb']:.2f} MB")
        print(f"Delta:           {memory_usage['delta_mb']:.2f} MB")
        
        print("\nDetailed snapshots:")
        for label, mem_mb in memory_usage['snapshots'].items():
            print(f"  {label:20s}: {mem_mb:.2f} MB")
    else:
        print("\n✗ Memory tracking was not enabled")
        print("Tip: Use track_memory=True when setting solver")
    
    # Also show gate counts for comparison
    gate_counts = puzzle._solver.get_gate_counts()
    if gate_counts:
        print("\n" + "=" * 70)
        print("Gate Counts (Always Available)")
        print("=" * 70)
        for gate_name, count in sorted(gate_counts.items()):
            print(f"  {gate_name:10s}: {count}")
    
    print("\n" + "=" * 70)
    print("Use Cases for Memory Tracking:")
    print("  - Profiling circuit construction for large problem sizes")
    print("  - Identifying memory bottlenecks during development")
    print("  - Debugging out-of-memory issues")
    print("  - Performance optimization and resource planning")
    print("=" * 70)

if __name__ == "__main__":
    main()