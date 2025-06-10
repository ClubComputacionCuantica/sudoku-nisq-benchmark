from abc import ABC, abstractmethod
from pytket import Circuit, Qubit, OpType
from pytket.passes import FlattenRegisters
from pytket.circuit.display import render_circuit_jupyter as draw
from qiskit_ibm_runtime import QiskitRuntimeService
from pytket.extensions.qiskit import IBMQBackend

class QuantumSolver(ABC):
    """
    Abstract base class for quantum solvers.
    Supports multiple backends (Aer, IBMQ, Quantinuum, etc.) and allows switching.
    """
    def __init__(self, backends=None):
        self.main_circuit = None
        self.sim_counts = None
        # backends is a dict: name -> backend instance
        self.backends = backends.copy() if backends else {}
        self.current_backend = None

    @abstractmethod
    def get_circuit(self):
        """Construct and return a pytket Circuit for the problem."""
        pass

    def _prepare_circuit(self):
        """Flatten registers and prepare the main circuit."""
        if self.main_circuit is None:
            self.main_circuit = self.get_circuit()
        FlattenRegisters().apply(self.main_circuit)
        return self.main_circuit

    def add_backend(self, name: str, backend_instance):
        """Register a backend by name."""
        self.backends[name] = backend_instance

    def set_backend(self, name: str):
        """Select the active backend by name."""
        if name not in self.backends:
            raise KeyError(f"Backend '{name}' not registered")
        self.current_backend = name

    def init_ibm(self, token: str, channel: str = "ibm_quantum", name: str = "ibm"):
        """Authenticate IBM and register its backend."""
        QiskitRuntimeService.save_account(channel=channel, token=token, overwrite=True)
        devices = IBMQBackend.available_devices()
        ibm = IBMQBackend(devices[0])
        self.add_backend(name, ibm)
        return devices

    def init_quantinuum(self, token: str, name: str = "quantinuum"):
        """Authenticate Quantinuum and register its backend (stub)."""
        if QuantinuumBackend is None:
            raise RuntimeError("Quantinuum extension not installed")
        quant = QuantinuumBackend(token)
        self.add_backend(name, quant)

    def aer_simulation(self, shots=1024, aer_name: str = "aer"):
        """Register and run simulation on Aer backend."""
        from pytket.extensions.qiskit import AerBackend
        aer = AerBackend()
        self.add_backend(aer_name, aer)
        self.set_backend(aer_name)
        return self.run(shots=shots)

    def run(self, shots=1024):
        """Execute the circuit on the current backend (Aer/IBMQ/Quantinuum)."""
        if self.current_backend is None:
            raise RuntimeError("No backend selected. Call set_backend() first.")
        backend = self.backends[self.current_backend]
        circ = self._prepare_circuit()
        # Distinguish backends by type
        if backend.__class__.__name__ == 'AerBackend':
            compiled = backend.get_compiled_circuit(circ)
            handle = backend.process_circuit(compiled, n_shots=shots)
            self.sim_counts = backend.get_result(handle).get_counts()
        elif backend.__class__.__name__ == 'IBMQBackend':
            transpiled = backend.get_compiled_circuit(circ, optimisation_level=0)
            handle = backend.process_circuit(transpiled, n_shots=shots)
            self.sim_counts = backend.get_result(handle).get_counts()
        elif QuantinuumBackend and isinstance(backend, QuantinuumBackend):
            # Stub for Quantinuum run
            handle = backend.process_circuit(circ, n_shots=shots)
            self.sim_counts = backend.get_result(handle).get_counts()
        else:
            raise RuntimeError(f"Backend type '{type(backend)}' not supported for run()")
        return self.sim_counts

    def counts_plot(self, counts=None):
        """Generic bar plot of measurement counts."""
        counts = counts or self.sim_counts
        if counts is None:
            raise ValueError("No counts to plot. Run run() first.")
        import matplotlib.pyplot as plt
        labels, values = zip(*counts.items())
        plt.figure(figsize=(6,4))
        plt.bar(labels, values)
        plt.xticks(rotation=90)
        plt.title(f"{self.__class__.__name__} Outcomes ({self.current_backend})")
        plt.tight_layout()
        plt.show()

# Integration: in Sudoku.__init__:
#   shared_backends = {}
#   solver = ExactCoverQuantumSolver(self, backends=shared_backends)
#   solver.init_ibm(ibm_token)
#   solver.init_quantinuum(quant_token)
#   solver.aer_simulation(shots=1024)  # registers Aer
#   solver.set_backend('ibm')
#   solver.run()
#   solver.set_backend('quantinuum')
#   solver.run()
#   solver.counts_plot()  # plots results from the current backend
#   solver.draw_circuit()  # visualizes the circuit
#   # etc.
# Note: The above code assumes that the necessary imports and dependencies are available.
# The QuantumSolver class is now an abstract base class that can be extended by specific solvers.
# The ExactCoverQuantumSolver, GraphColoringQuantumSolver, and BacktrackingQuantumSolver
# classes inherit from QuantumSolver and implement their specific logic.
# The QuantumSolver class provides a unified interface for different quantum backends,
# allowing for easy switching and execution of quantum circuits.
# The code is structured to allow for easy extension and integration with various quantum backends,
# while maintaining a clean and modular design.