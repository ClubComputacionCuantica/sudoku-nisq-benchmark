"""Error Mitigation Comparison Example.

Demonstrates Zero Noise Extrapolation (ZNE) with the Sudoku NISQ solver
using different backend types. Shows how mitigation works identically across
native Qiskit backends and pytket backends.

This example compares:
1. Raw (unmitigated) execution
2. ZNE-mitigated execution

Backends demonstrated:
- Aer simulator (native Qiskit)
- Aer simulator with noise model (native Qiskit)

For IBM Quantum or Quantinuum hardware, replace the backend alias and ensure
you have appropriate credentials configured.

Requirements:
    pip install mitiq qiskit qiskit-aer qiskit-ibm-runtime
"""

from sudoku_nisq import QSudoku
from sudoku_nisq.solvers.exact_cover_solver import ExactCoverQuantumSolver
from sudoku_nisq.backends import BackendManager

# Optional: For noise modeling
try:
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    NOISE_AVAILABLE = True
except ImportError:
    NOISE_AVAILABLE = False
    print("Note: qiskit-aer not available. Install with: pip install qiskit-aer")


def create_simple_noise_model():
    """Create a simple depolarizing noise model for demonstration."""
    if not NOISE_AVAILABLE:
        return None
    
    noise_model = NoiseModel()
    
    # Add depolarizing error to single-qubit gates
    error_1q = depolarizing_error(0.001, 1)  # 0.1% error rate
    noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'x', 'h'])
    
    # Add depolarizing error to two-qubit gates
    error_2q = depolarizing_error(0.01, 2)  # 1% error rate
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz'])
    
    return noise_model


def run_comparison_single_backend(backend_alias: str, shots: int = 4096, use_noise: bool = False):
    """Run comparison on a single backend.
    
    Args:
        backend_alias: Backend identifier (e.g., 'aer', 'ibm_brisbane', 'H1-1')
        shots: Number of measurement shots per execution
        use_noise: If True and backend is Aer, apply a simple noise model
    """
    print(f"\n{'='*70}")
    print(f"Backend: {backend_alias}")
    print(f"Shots: {shots}")
    if use_noise and backend_alias == 'aer':
        print("Noise model: Simple depolarizing noise")
    print(f"{'='*70}\n")
    
    # 1. Setup puzzle (2x2 for quick demo)
    # TODO: Add seed parameter for reproducibility (Phase 5 implementation)
    puzzle = QSudoku.generate(size=2, num_missing_cells=2, subgrid_size=1)
    puzzle.set_solver(ExactCoverQuantumSolver, encoding='simple')
    
    print("Puzzle configuration:")
    print("  Size: 2x2")
    print("  Missing cells: 2")
    print("  Encoding: simple")
    
    # 2. Check resource requirements
    resources = puzzle._solver.resource_estimation()
    print("\nResource requirements:")
    print(f"  Qubits: {resources['n_qubits']}")
    print(f"  Gates: {resources['n_gates']}")
    
    # 3. Get backend
    manager = BackendManager.inst()
    
    # Special handling for Aer with noise
    if backend_alias == 'aer' and use_noise:
        noise_model = create_simple_noise_model()
        if noise_model is None:
            print("\nSkipping noisy simulation (qiskit-aer not available)")
            return
        # Initialize Aer with noise model
        backend_alias = manager.init_aer(device="statevector", noise_model=noise_model, alias="aer_noisy")
        backend = manager.get(backend_alias)
        # Note: Noise model application happens in run_aer() method
        print(f"\nBackend type: {type(backend).__name__}")
    else:
        # Initialize Aer backend if needed
        if backend_alias == 'aer':
            backend_alias = manager.init_aer(device="statevector", alias="aer")
        backend = manager.get(backend_alias)
        print(f"\nBackend type: {type(backend).__name__}")
    
    # Detect SDK type
    backend_module = getattr(backend, '__module__', '')
    is_qiskit = 'qiskit' in backend_module
    is_pytket = hasattr(backend, 'process_circuit')
    
    if is_qiskit:
        print("SDK: Native Qiskit (no conversion needed for mitigation)")
    elif is_pytket:
        print("SDK: PyTKET (Qiskit→pytket conversion in mitigation)")
    
    # 4. Build circuit
    puzzle.build_circuit(sdk="qiskit")
    print("\nCircuit built (Qiskit format)")
    
    # 5. Run WITHOUT mitigation
    print("\n" + "-"*70)
    print("Running WITHOUT mitigation...")
    print("-"*70)
    
    if 'aer_noisy' in backend_alias:
        # Backend already has noise model configured
        result_raw = puzzle.run_aer(shots=shots)
    elif backend_alias == 'aer':
        result_raw = puzzle.run_aer(shots=shots)
    else:
        result_raw = puzzle.run(backend_alias, shots=shots)
    
    formatted_raw = puzzle.format_result(result_raw)
    raw_success_rate = formatted_raw['success_rate']
    
    print(f"Raw success rate: {raw_success_rate:.2%}")
    if formatted_raw.get('solutions') and len(formatted_raw['solutions']) > 0:
        top_sol = formatted_raw['solutions'][0]
        count = top_sol.get('count', top_sol.get('counts', 'N/A'))
        print(f"Top solution counts: {count}")
    
    # 6. Run WITH ZNE mitigation
    print("\n" + "-"*70)
    print("Running WITH ZNE mitigation...")
    print("-"*70)
    
    try:
        if 'aer' in backend_alias:
            # For Aer, we need to use the lower-level run() method with use_zne
            # since run_aer doesn't support mitigation parameters yet
            result_zne = puzzle._solver.run(
                backend=backend,
                backend_alias=backend_alias,
                shots=shots,
                use_zne=True
            )
        else:
            result_zne = puzzle._solver.run(
                backend=backend,
                backend_alias=backend_alias,
                shots=shots,
                use_zne=True
            )
        
        formatted_zne = puzzle.format_result(result_zne)
        zne_success_rate = formatted_zne.get('success_rate', 0)
        
        # Access mitigated success probability
        if hasattr(result_zne, 'mitigated_success_prob'):
            mitigated_prob = result_zne.mitigated_success_prob
            print(f"Mitigated success probability: {mitigated_prob:.2%}")
        else:
            print("Warning: Mitigated probability not attached to result")
            mitigated_prob = None
        
        print(f"ZNE success rate (from counts): {zne_success_rate:.2%}")
        if formatted_zne.get('solutions') and len(formatted_zne['solutions']) > 0:
            top_sol = formatted_zne['solutions'][0]
            count = top_sol.get('count', top_sol.get('counts', 'N/A'))
            print(f"Top solution counts: {count}")
        
        # 7. Compare results
        print("\n" + "="*70)
        print("COMPARISON")
        print("="*70)
        print(f"Raw success rate:       {raw_success_rate:.4%}")
        if mitigated_prob is not None:
            print(f"Mitigated probability:  {mitigated_prob:.4%}")
            improvement = mitigated_prob - raw_success_rate
            improvement_pct = (improvement / raw_success_rate * 100) if raw_success_rate > 0 else 0
            print(f"Absolute improvement:   {improvement:+.4%}")
            print(f"Relative improvement:   {improvement_pct:+.2f}%")
        else:
            print("Mitigated probability:  Not available")
        
        print("\nNote: ZNE effectiveness depends on:")
        print("  - Circuit depth and noise levels")
        print("  - Scale factors used (default: [1, 3, 5])")
        print("  - Extrapolation method (default: Richardson)")
        print("  - For very low noise, improvement may be minimal")
        
    except ImportError as e:
        print(f"\nError: Mitiq not installed. {e}")
        print("Install with: pip install mitiq")
    except Exception as e:
        print(f"\nError during mitigation: {e}")
        import traceback
        traceback.print_exc()


def run_multi_backend_comparison():
    """Compare mitigation across multiple backend types."""
    print("\n" + "="*70)
    print("MULTI-BACKEND MITIGATION COMPARISON")
    print("="*70)
    
    # Aer without noise (baseline)
    run_comparison_single_backend('aer', shots=4096, use_noise=False)
    
    # Aer with noise (shows mitigation benefit)
    if NOISE_AVAILABLE:
        run_comparison_single_backend('aer', shots=4096, use_noise=True)
    
    # Uncomment to test with IBM Quantum (requires credentials)
    # run_comparison_single_backend('ibm_brisbane', shots=4096)
    
    # Uncomment to test with Quantinuum (requires credentials)
    # run_comparison_single_backend('H1-1', shots=4096)
    
    # NOTE: AWS Braket is NOT supported for error mitigation
    # (Braket SDK is incompatible with Mitiq's Qiskit/Cirq-only interface)


def main():
    """Run error mitigation demonstration."""
    print("""
==================================================================
         Error Mitigation Comparison for Sudoku NISQ
==================================================================
  This example demonstrates Zero Noise Extrapolation (ZNE)
  with different backend types (native Qiskit and pytket).

  The mitigation executor automatically detects backend SDK
  type and handles circuit conversions transparently.
==================================================================
    """)
    
    print("\n[NOTE] This example requires Mitiq library for error mitigation.")
    print("If not installed, run: pip install mitiq\n")
    
    # Single backend demo (Aer with noise)
    if NOISE_AVAILABLE:
        print("\nRunning demonstration with noisy Aer simulator...")
        print("Note: This may take a few minutes due to ZNE overhead.\n")
        try:
            run_comparison_single_backend('aer', shots=1024, use_noise=True)
        except ImportError as e:
            print(f"\n❌ Missing dependency: {e}")
            print("Install Mitiq with: pip install mitiq")
            return
        except Exception as e:
            print(f"\n❌ Error during mitigation: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        print("\n⚠️  qiskit-aer not available. Install with: pip install qiskit-aer")
        print("Skipping noisy simulation demo.")
        return


if __name__ == "__main__":
    main()
