"""
Minimal demo for running an exact cover problem on Qiskit Aer.

Shows how to use QExactCover to build and simulate a circuit
without requiring access to a real quantum backend.
"""

from sudoku_nisq.q_exact_cover import QExactCover


def main():
    # Create a tiny example problem
    qec = QExactCover.create_small_example()

    # Check resource requirements
    resources = qec.report_resources()
    print(f"Estimated qubits: {resources['estimated']['n_qubits']}")
    print(f"Estimated gates: {resources['estimated']['n_gates']}")

    # Build circuit (Qiskit for Aer simulation)
    circuit = qec.build_circuit(sdk="qiskit")
    print(f"Circuit qubits: {circuit.num_qubits}, depth: {circuit.depth()}")

    # Run on Aer with reasonable shots
    result = qec.run_aer(shots=512, opt_level=1)
    counts = result["counts"]
    print(f"Measured {len(counts)} unique outcomes")
    # Show top few outcomes
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
    for bitstring, c in top:
        print(f"{bitstring}: {c}")


if __name__ == "__main__":
    main()