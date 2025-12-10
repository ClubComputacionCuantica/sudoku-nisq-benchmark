# exact.py

from sudoku_nisq import ExactCoverProblem, QExactCover

def main():
    # 1) Define a small exact cover problem
    # Universe elements must be hashable; labels are arbitrary
    universe = [0, 1, 2, 3]
    subsets = {
        'S_0': [0, 3],    # Covers elements 0 and 3
        'S_1': [1, 2],    # Covers elements 1 and 2
        'S_2': [0, 1, 2], # Alternative covering
    }

    # 2) Create the problem dataclass (auto-validates)
    problem = ExactCoverProblem(universe, subsets, num_solutions=1)

    # 3) Create the quantum interface for exact cover
    qec = QExactCover(problem)

    # 4) Build the circuit (SDK auto-selected; Aer uses Qiskit natively)
    circuit = qec.build_circuit()  # or qec.build_circuit(sdk="qiskit")

    # 5) Run on Aer simulator
    result = qec.run_aer(shots=512)

    # 6) Show measurement counts
    counts = result.get("counts", result)
    print("Counts:", counts)

    # 7) Show a concise resource summary
    resources = qec.report_resources()
    est = resources.get("estimated", {})
    print(f"Estimated qubits: {est.get('n_qubits')}")
    print(f"Estimated gates: {est.get('n_gates')}")

if __name__ == "__main__":
    main()