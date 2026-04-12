"""Executor factories for Mitiq error mitigation integration.

Provides functions to create executor callables that wrap backend execution
for use with Zero Noise Extrapolation (ZNE) and Probabilistic Error 
Cancellation (PEC) techniques.

Architecture: How Mitiq Integrates with Backends
================================================

**Key Principle: Mitiq ONLY speaks Qiskit/Cirq. Your backends may use Qiskit OR pytket.**

The executor is the translation layer that bridges this gap. When Mitiq provides a Qiskit
circuit, the executor detects your backend's SDK type and converts the circuit format only
if needed (Qiskit → pytket for pytket backends). Here's the COMPLETE flow:

.. code-block:: text

                    Qiskit circuit
    ┌─────────────────────────────────────────────────────────┐
    │              Mitiq (ZNE / PEC)                          │
    │  • Scales noise (gate folding for ZNE)                  │
    │  • Samples circuits (quasiprobabilities for PEC)        │
    │  • Works ONLY with Qiskit/Cirq circuits                 │
    │  • Calls executor(qiskit_circuit) multiple times        │
    └───────────────────┬────────────────────────────────────┘
                        │
                        ▼ Qiskit circuit (modified by Mitiq)
    ┌─────────────────────────────────────────────────────────┐
    │     execute(qiskit_circuit) -> float  [EXECUTOR]        │
    │  ┌───────────────────────────────────────────────────┐  │
    │  │ 1. Get backend from BackendManager                │  │
    │  │ 2. Run: handle = backend.process_circuit(pytket)  │  │
    │  │ 3. Get: result = backend.get_result(handle)       │  │
    │  │ 4. Extract: counts = result.get_counts()          │  │
    │  │ 5. Compute: success_prob from counts + validator  │  │
    │  │ 6. Return: success_prob (scalar in [0, 1])        │  │
    │  └───────────────────────────────────────────────────┘  │
    └───────────────────┬────────────────────────────────────┘
                        │
                        ▼
    ┌─────────────────────────────────────────────────────────┐
    │     Backend (via BackendManager)                        │
    │  • IBM Quantum (Qiskit backend)  → Uses Qiskit directly │
    │  • Aer Simulator (Qiskit backend) → Uses Qiskit directly│
    │  • Quantinuum (pytket backend)    → Converts to pytket  │
    │  Note: AWS Braket NOT supported (incompatible SDK)      │
    └─────────────────────────────────────────────────────────┘

Mitiq thinks it's always working with Qiskit circuits.
1. Receives Qiskit circuit from Mitiq (after noise scaling)
2. Detects backend SDK type (native Qiskit vs pytket)
3. Converts circuit format only if needed (Qiskit → pytket for pytket backends)
4. Executes via backend.run() (Qiskit) or backend.process_circuit() (pytket)
5. Returns success probability to Mitiq

This is why the SAME executor pattern works with all backends despite Mitiq
only supporting Qiskit/Cirq natively.

Multi-Backend Support (Qiskit and PyTKET)
==========================================

The executor pattern works with Qiskit and pytket backends because it detects
the backend SDK type and converts circuit format only when necessary:

.. code-block:: python

    # Backend detection
    is_qiskit_backend = 'qiskit_ibm_runtime' in backend.__module__ or \
                       (hasattr(backend, 'target') or hasattr(backend, 'configuration'))
    is_pytket_backend = hasattr(backend, 'process_circuit') and hasattr(backend, 'get_result')
    
    # Execute based on backend SDK type:
    if is_qiskit_backend:  # Qiskit backends (IBM, Aer)
        # Use Qiskit circuit directly - no conversion needed
        job = backend.run(qiskit_circuit, shots=shots)
        result = job.result()
        counts = result.get_counts()
    else:  # PyTKET backends (Quantinuum)
        # Convert Qiskit → pytket for backend compatibility
        handle = backend.process_circuit(pytket_circuit, n_shots=shots)
        result = backend.get_result(handle)
        counts = result.get_counts()

**Key Point:** Mitiq ONLY works with Qiskit/Cirq circuits. We convert to pytket
ONLY when the backend requires it (e.g., Quantinuum):

.. code-block:: python

    # Step 1: Convert pytket → Qiskit for Mitiq
    from pytket.extensions.qiskit import tk_to_qiskit
    qiskit_circuit = tk_to_qiskit(your_pytket_circuit)
    
    # Step 2: Mitiq works with Qiskit
    # (internally scales noise and calls executor multiple times)
    
    # Step 3: Inside executor, convert Qiskit → pytket
    def executor(qiskit_circuit_from_mitiq):
        from pytket.extensions.qiskit import qiskit_to_tk
        pytket_circuit = qiskit_to_tk(qiskit_circuit_from_mitiq)
        
        # Now use ANY backend that supports pytket!
        handle = backend.process_circuit(pytket_circuit, n_shots=shots)
        result = backend.get_result(handle)
        # ... compute and return success probability

This means you can use the SAME mitigation code with different backend types:

.. code-block:: python

    from sudoku_nisq.backends import BackendManager
    from sudoku_nisq.mitigation.executors import create_zne_executor
    from mitiq import zne
    from pytket.extensions.qiskit import tk_to_qiskit

    manager = BackendManager()
    
    # Your circuit (can be pytket or qiskit format)
    pytket_circuit = solver.build_circuit(puzzle)
    
    # MUST convert to Qiskit for Mitiq (Mitiq only accepts Qiskit/Cirq)
    qiskit_circuit = tk_to_qiskit(pytket_circuit)
    
    # IBM Quantum - Returns Qiskit backend
    ibm_backend = manager.get("ibm_brisbane")
    ibm_executor = create_zne_executor(ibm_backend, solver, shots=4096)
    ibm_result = zne.execute_with_zne(qiskit_circuit, ibm_executor)
    # → Executor uses Qiskit circuit directly (backend is Qiskit)
    
    # Quantinuum - Returns pytket backend
    quant_backend = manager.get("H1-1")
    quant_executor = create_zne_executor(quant_backend, solver, shots=4096)
    quant_result = zne.execute_with_zne(qiskit_circuit, quant_executor)
    # → Executor converts Qiskit → pytket (backend requires pytket)
    
    # Aer simulator - Returns Qiskit backend
    aer_backend = manager.get("aer")
    aer_executor = create_zne_executor(aer_backend, solver, shots=4096)
    aer_result = zne.execute_with_zne(qiskit_circuit, aer_executor)
    # → Executor uses Qiskit circuit directly (backend is Qiskit)

**Critical Flow:** Mitiq provides a Qiskit circuit to the executor. The executor
detects the backend's SDK type and converts ONLY if the backend requires pytket.

Circuit Format Journey (Backend-Dependent):

**For Qiskit Backends (IBM Quantum, Aer):**
- Your code: Can use pytket or qiskit
- For Mitiq: Must be Qiskit (tk_to_qiskit if needed)
- Mitiq: Scales noise, calls executor with Qiskit circuit
- Executor: Uses Qiskit circuit directly (NO CONVERSION)
- Backend: backend.run(qiskit_circuit, shots=shots)
- Result: Qiskit-native execution with mitigation ✓

**For PyTKET Backends (Quantinuum):**
- Your code: Can use pytket or qiskit
- For Mitiq: Must be Qiskit (tk_to_qiskit if needed)
- Mitiq: Scales noise, calls executor with Qiskit circuit
- Executor: Converts Qiskit → pytket (qiskit_to_tk)
- Backend: backend.process_circuit(pytket_circuit, n_shots=shots)
- Result: PyTKET execution with mitigation ✓

**AWS Braket:** NOT SUPPORTED. Braket SDK is incompatible with current
mitigation architecture (would require Qiskit/Cirq → Braket conversion
which is not implemented).

Integration with Your Sudoku Framework
=======================================

The executor factories wrap your EXISTING backend infrastructure. In
``quantum_solver.py``, you already have this pattern:

.. code-block:: python

    # Existing code in QuantumSolver.run():
    backend = BackendManager().get(backend_alias)
    handle = backend.process_circuit(compiled_circuit, n_shots=shots)
    result = backend.get_result(handle)
    counts = result.get_counts()

The executors add ONE additional step:

.. code-block:: python

    def executor(circuit):
        # Your existing backend calls:
        handle = backend.process_circuit(circuit, n_shots=shots)
        result = backend.get_result(handle)
        counts = result.get_counts()
        
        # NEW: Compute success probability for Mitiq
        success_prob = compute_success_expectation(
            counts, 
            solver._is_valid_solution
        )
        return success_prob  # Scalar expectation value

When you call ``solver.run(backend, use_zne=True)``, the flow is:

1. **QuantumSolver** gets backend from ``BackendManager``
2. **Mitigation module** wraps backend in executor function
3. **Mitiq** calls ``executor(circuit)`` multiple times with scaled circuits
4. **Executor** uses your existing ``backend.process_circuit()`` pattern
5. **Executor** computes success probability and returns to Mitiq
6. **Mitiq** extrapolates to zero-noise estimate
7. **QuantumSolver** attaches result as ``.mitigated_success_prob``

Example Usage
=============

Basic usage with ZNE:

.. code-block:: python

    from sudoku_nisq import QSudoku
    from sudoku_nisq.solvers.exact_cover_solver import ExactCoverQuantumSolver
    from sudoku_nisq.backends import BackendManager
    
    # Setup
    qs = QSudoku.generate(size=2, num_missing=2)
    qs.set_solver(ExactCoverQuantumSolver, encoding='simple')
    
    # Get backend
    manager = BackendManager()
    backend = manager.get('ibm_brisbane')  # or 'H1-1', 'aer', etc.
    
    # Run with ZNE
    result = qs.solver.run(
        backend=backend,
        backend_alias='ibm_brisbane',
        shots=4096,
        use_zne=True  # Enable error mitigation
    )
    
    # Access mitigated success probability
    raw_counts = result.get_counts()
    mitigated_prob = result.mitigated_success_prob
    
    print(f"Mitigated success probability: {mitigated_prob:.2%}")

See Also
========

- ``expectation_wrapper.py``: Converts measurement counts to expectation values
- ``quantum_solver.py``: Integration point for mitigation in solver.run()
- ``backends.py``: Unified backend management across providers
- Mitiq documentation: https://mitiq.readthedocs.io/
"""

from typing import Any, Callable, Optional

try:
    from mitiq import zne, pec
    MITIQ_AVAILABLE = True
except ImportError:
    MITIQ_AVAILABLE = False

# NOTE: Mitiq requires additional dependencies for circuit conversions:
# - For Qiskit support: qiskit, cirq (with ply module)
# - For full functionality: pip install mitiq[qiskit] cirq[contrib]
# If you encounter "ModuleNotFoundError: No module named 'ply'",
# install with: pip install python-ply


def create_zne_executor(
    backend: Any,
    solver: Any,
    shots: int = 1024,
    scale_noise: Optional[Callable] = None,
    factory: Optional[Any] = None,
    **kwargs
) -> Callable:
    """Create a ZNE-enabled executor for the quantum solver.
    
    Wraps backend execution to return success probability as an expectation
    value, enabling Zero Noise Extrapolation (ZNE) error mitigation.
    
    The executor bridges Mitiq and your backend infrastructure by wrapping
    the standard pytket backend pattern:
    
    .. code-block:: python
    
        def executor(circuit):
            handle = backend.process_circuit(circuit, n_shots=shots)
            result = backend.get_result(handle)
            counts = result.get_counts()
            return compute_success_expectation(counts, validator)
    
    This works with ANY backend from BackendManager (IBM, Quantinuum, etc.)
    because they all expose the same pytket interface.
    
    Args:
        backend: Quantum backend instance from BackendManager with pytket interface.
            Must support: backend.process_circuit() and backend.get_result().
            Works with: IBM Quantum (via pytket-qiskit), Quantinuum (native pytket), Aer simulator.
        solver: Quantum solver instance with _is_valid_solution method for
            validating measurement outcomes as correct exact cover solutions.
        shots (int): Number of measurement shots per execution. Higher values
            improve statistical accuracy but increase runtime. Default: 1024.
            Recommended: 4096+ for real hardware.
        scale_noise (Callable, optional): Noise scaling function for ZNE.
            If None, uses Mitiq's default unitary folding strategy.
            TODO: Make configurable per backend noise characteristics.
        factory (Any, optional): Extrapolation factory for ZNE.
            If None, uses Richardson extrapolation with default polynomial degree.
            TODO: Support LinearFactory, RichardsonFactory, etc. via config.
        **kwargs: Additional backend execution parameters passed to
            backend.process_circuit().
    
    Returns:
        Callable: Executor function with signature ``execute(circuit) -> float``
            that Mitiq calls to get success probability expectation values.
    
    Raises:
        ImportError: If Mitiq is not installed.
    
    Example:
        Basic usage with IBM backend:
        
        >>> from sudoku_nisq.backends import BackendManager
        >>> from mitiq import zne
        >>> 
        >>> manager = BackendManager()
        >>> backend = manager.get("ibm_brisbane")
        >>> 
        >>> executor = create_zne_executor(backend, solver, shots=4096)
        >>> mitigated_prob = zne.execute_with_zne(circuit, executor)
        >>> print(f"Success probability: {mitigated_prob:.2%}")
        
        Multi-backend comparison:
        
        >>> backends = {
        ...     'ibm': manager.get('ibm_brisbane'),
        ...     'quantinuum': manager.get('H1-1'),
        ...     'aer': manager.get('aer')
        ... }
        >>> 
        >>> results = {}
        >>> for name, backend in backends.items():
        ...     executor = create_zne_executor(backend, solver, shots=4096)
        ...     results[name] = zne.execute_with_zne(circuit, executor)
    
    See Also:
        - apply_zne(): Convenience function that combines executor creation and ZNE
        - create_pec_executor(): PEC version of this function
        - BackendManager: Unified backend access across providers
    """
    if not MITIQ_AVAILABLE:
        raise ImportError(
            "Mitiq is required for ZNE. Install with: pip install mitiq"
        )
    
    from sudoku_nisq.mitigation.expectation_wrapper import compute_success_expectation
    
    def executor(circuit: Any) -> float:
        """Execute circuit and return success probability expectation.
        
        Supports both PyTKET and native Qiskit backends:
        - PyTKET backends: Use backend.process_circuit() interface
        - Qiskit backends: Use backend.run() interface
        
        Conversion rules:
        - Accepts either a `pytket.Circuit` or `qiskit.QuantumCircuit` provided by Mitiq.
        - If a Qiskit circuit is received it is converted via `qiskit_to_tk` for PyTKET backends.
        - For native Qiskit backends, Qiskit circuits are used directly.
        - Any other type raises `TypeError` to fail fast with a clear message.
        
        Returns:
            float: Success probability as expectation value.
        """
        from pytket import Circuit
        from qiskit import QuantumCircuit
        
        # Detect backend type
        backend_module = getattr(backend, '__module__', '')
        is_qiskit_backend = 'qiskit_ibm_runtime' in backend_module or \
                           (hasattr(backend, 'target') or hasattr(backend, 'configuration'))
        is_pytket_backend = hasattr(backend, 'process_circuit') and hasattr(backend, 'get_result')
        
        # Handle circuit format based on backend type
        if is_qiskit_backend and not is_pytket_backend:
            # Native Qiskit backend - use Qiskit circuit directly
            if isinstance(circuit, Circuit):
                # Convert pytket to Qiskit
                try:
                    from pytket.extensions.qiskit import tk_to_qiskit
                    qiskit_circuit = tk_to_qiskit(circuit)
                except ImportError as exc:
                    raise ImportError(
                        "pytket-qiskit extension is required for pytket→Qiskit conversion. "
                        "Install with: pip install 'pytket[qiskit]'"
                    ) from exc
            elif isinstance(circuit, QuantumCircuit):
                qiskit_circuit = circuit
            else:
                raise TypeError(
                    f"Unsupported circuit type from Mitiq: {type(circuit)}. "
                    "Only pytket.Circuit and qiskit.QuantumCircuit are supported."
                )
            
            # Execute on native Qiskit backend
            job = backend.run(qiskit_circuit, shots=shots)
            result = job.result()
            counts = result.get_counts()
            
        else:
            # PyTKET backend - convert to pytket circuit
            if isinstance(circuit, QuantumCircuit):
                try:
                    from pytket.extensions.qiskit import qiskit_to_tk
                    pytket_circuit = qiskit_to_tk(circuit)
                except ImportError as exc:
                    raise ImportError(
                        "pytket-qiskit extension and qiskit are required for Qiskit→pytket conversion. "
                        "Install with: pip install 'pytket[qiskit]' qiskit"
                    ) from exc
            elif isinstance(circuit, Circuit):
                pytket_circuit = circuit
            else:
                raise TypeError(
                    f"Unsupported circuit type from Mitiq: {type(circuit)}. "
                    "Only pytket.Circuit and qiskit.QuantumCircuit are supported."
                )
            
            # Execute on PyTKET backend
            handle = backend.process_circuit(pytket_circuit, n_shots=shots, **kwargs)
            result = backend.get_result(handle)
            counts = result.get_counts()

        # Safety: ensure solver has a validator
        validator = getattr(solver, "_is_valid_solution", None)
        if not callable(validator):
            raise AttributeError(
                "Solver is missing a callable '_is_valid_solution' validator required for mitigation."
            )

        return compute_success_expectation(counts, validator)
    
    return executor


def create_pec_executor(
    backend: Any,
    solver: Any,
    shots: int = 1024,
    representations: Optional[Any] = None,
    **kwargs
) -> Callable:
    """Create a PEC-enabled executor for the quantum solver.
    
    Wraps backend execution for Probabilistic Error Cancellation (PEC),
    which requires decomposing gates into noisy basis operations.
    
    Like create_zne_executor(), this wraps your existing backend infrastructure
    but adds support for PEC's quasiprobability sampling strategy. The executor
    uses the same pytket backend pattern that works across all providers.
    
    Args:
        backend: Quantum backend instance from BackendManager with pytket interface.
            Must support: backend.process_circuit() and backend.get_result().
            Works with: IBM Quantum (via pytket-qiskit), Quantinuum (native pytket), Aer simulator.
        solver: Quantum solver instance with _is_valid_solution method for
            validating measurement outcomes.
        shots (int): Number of measurement shots per execution. PEC typically
            requires more shots than ZNE due to sampling overhead. Default: 1024.
            Recommended: 8192+ for real hardware.
        representations (Any, optional): OperationRepresentation list for PEC.
            Maps ideal gates to noisy implementations with quasiprobabilities.
            If None, raises ValueError (must be provided externally).
            TODO: Auto-generate from backend noise model or calibration data.
        **kwargs: Additional backend execution parameters passed to
            backend.process_circuit().
    
    Returns:
        Callable: Executor function with signature ``execute(circuit) -> float``
            for PEC-mitigated execution.
    
    Raises:
        ImportError: If Mitiq is not installed.
        ValueError: If representations is None (required for PEC).
    
    Note:
        PEC requires sampling many circuits with quasiprobability weights,
        making it significantly more expensive than ZNE (~100-1000x more shots).
        The representations must accurately model the backend's noise for PEC
        to provide unbiased estimates. Use ZNE for initial experiments.
    
    Example:
        Basic PEC usage (requires representations):
        
        >>> from sudoku_nisq.backends import BackendManager
        >>> from mitiq import pec
        >>> 
        >>> # Generate representations from backend noise model
        >>> # (implementation depends on backend calibration data)
        >>> representations = get_backend_representations(backend)
        >>> 
        >>> manager = BackendManager()
        >>> backend = manager.get("ibm_brisbane")
        >>> 
        >>> executor = create_pec_executor(
        ...     backend, solver, shots=8192, representations=representations
        ... )
        >>> mitigated_prob = pec.execute_with_pec(
        ...     circuit, executor, representations=representations
        ... )
        
        Comparing ZNE and PEC:
        
        >>> zne_executor = create_zne_executor(backend, solver, shots=4096)
        >>> pec_executor = create_pec_executor(
        ...     backend, solver, shots=8192, representations=reps
        ... )
        >>> 
        >>> zne_result = zne.execute_with_zne(circuit, zne_executor)
        >>> pec_result = pec.execute_with_pec(circuit, pec_executor, reps)
        >>> 
        >>> print(f"ZNE success prob: {zne_result:.2%}")
        >>> print(f"PEC success prob: {pec_result:.2%}")
    
    See Also:
        - apply_pec(): Convenience function combining executor creation and PEC
        - create_zne_executor(): Simpler ZNE alternative
        - Mitiq PEC docs: https://mitiq.readthedocs.io/en/stable/guide/pec.html
    """
    if not MITIQ_AVAILABLE:
        raise ImportError(
            "Mitiq is required for PEC. Install with: pip install mitiq"
        )
    
    if representations is None:
        raise ValueError(
            "PEC requires 'representations' parameter defining gate decompositions. "
            "See Mitiq documentation for OperationRepresentation generation."
        )
    
    from sudoku_nisq.mitigation.expectation_wrapper import compute_success_expectation
    
    def executor(circuit: Any) -> float:
        """Execute circuit and return success probability expectation.
        
        Supports both PyTKET and native Qiskit backends via backend detection.
        Accepts pytket or Qiskit circuits (after PEC sampling). Fails fast for
        unsupported types to surface misconfiguration early.
        """
        from pytket import Circuit
        from qiskit import QuantumCircuit
        
        # Detect backend type
        backend_module = getattr(backend, '__module__', '')
        is_qiskit_backend = 'qiskit_ibm_runtime' in backend_module or \
                           (hasattr(backend, 'target') or hasattr(backend, 'configuration'))
        is_pytket_backend = hasattr(backend, 'process_circuit') and hasattr(backend, 'get_result')
        
        # Handle circuit format based on backend type
        if is_qiskit_backend and not is_pytket_backend:
            # Native Qiskit backend
            if isinstance(circuit, Circuit):
                try:
                    from pytket.extensions.qiskit import tk_to_qiskit
                    qiskit_circuit = tk_to_qiskit(circuit)
                except ImportError as exc:
                    raise ImportError(
                        "pytket-qiskit extension is required for pytket→Qiskit conversion. "
                        "Install with: pip install 'pytket[qiskit]'"
                    ) from exc
            elif isinstance(circuit, QuantumCircuit):
                qiskit_circuit = circuit
            else:
                raise TypeError(
                    f"Unsupported circuit type from Mitiq/PEC: {type(circuit)}. "
                    "Only pytket.Circuit and qiskit.QuantumCircuit are supported."
                )
            
            job = backend.run(qiskit_circuit, shots=shots)
            result = job.result()
            counts = result.get_counts()
            
        else:
            # PyTKET backend
            if isinstance(circuit, QuantumCircuit):
                try:
                    from pytket.extensions.qiskit import qiskit_to_tk
                    pytket_circuit = qiskit_to_tk(circuit)
                except ImportError as exc:
                    raise ImportError(
                        "pytket-qiskit extension and qiskit are required for Qiskit→pytket conversion. "
                        "Install with: pip install 'pytket[qiskit]' qiskit"
                    ) from exc
            elif isinstance(circuit, Circuit):
                pytket_circuit = circuit
            else:
                raise TypeError(
                    f"Unsupported circuit type from Mitiq/PEC: {type(circuit)}. "
                    "Only pytket.Circuit and qiskit.QuantumCircuit are supported."
                )
            
            handle = backend.process_circuit(pytket_circuit, n_shots=shots, **kwargs)
            result = backend.get_result(handle)
            counts = result.get_counts()

        validator = getattr(solver, "_is_valid_solution", None)
        if not callable(validator):
            raise AttributeError(
                "Solver is missing a callable '_is_valid_solution' validator required for mitigation."
            )

        return compute_success_expectation(counts, validator)
    
    return executor


def apply_zne(
    circuit: Any,
    backend: Any,
    solver: Any,
    shots: int = 1024,
    scale_noise: Optional[Callable] = None,
    factory: Optional[Any] = None,
    **kwargs
) -> float:
    """Apply Zero Noise Extrapolation to a quantum circuit.
    
    Convenience function that creates an executor and applies ZNE in one call.
    This is used internally by QuantumSolver.run() when use_zne=True.
    
    The function wraps the three-step process:
    1. Create executor from backend (provider-agnostic)
    2. Let Mitiq scale noise and call executor multiple times
    3. Extrapolate to zero-noise success probability
    
    Args:
        circuit: Quantum circuit to execute with mitigation. Should be the
            transpiled/compiled circuit ready for backend execution (pytket.Circuit).
        backend: Quantum backend instance from BackendManager.
            Works with native Qiskit backends (IBM Quantum, Aer) and pytket backends (Quantinuum).
        solver: Quantum solver with _is_valid_solution method for validating
            measurement outcomes against exact cover constraints.
        shots (int): Number of shots per execution. ZNE will execute the circuit
            multiple times at different noise levels. Total shots ≈ shots × N_scales.
            Default: 1024. Recommended: 4096+ for real hardware.
        scale_noise (Callable, optional): Noise scaling function for ZNE.
            If None, uses Mitiq's default unitary folding with scale factors [1, 3, 5].
            TODO: Default to optimal folding for Grover-like algorithms.
        factory (Any, optional): Extrapolation factory for combining noisy results.
            If None, uses Richardson extrapolation with polynomial fitting.
            TODO: Auto-select based on circuit characteristics.
        **kwargs: Additional executor parameters passed to backend.process_circuit().
    
    Returns:
        float: Mitigated success probability in [0, 1], representing the
            extrapolated probability at zero noise. This is the "cleaned up"
            estimate of Pr[valid exact cover solution].
    
    Raises:
        ImportError: If Mitiq is not installed.
    
    Example:
        Direct usage (typically called by QuantumSolver.run()):
        
        >>> from sudoku_nisq.backends import BackendManager
        >>> from sudoku_nisq.mitigation.executors import apply_zne
        >>> 
        >>> manager = BackendManager()
        >>> backend = manager.get("ibm_brisbane")
        >>> 
        >>> # Circuit should be transpiled for the backend
        >>> transpiled_circuit = backend.get_compiled_circuit(
        ...     grover_circuit, optimisation_level=2
        ... )
        >>> 
        >>> mitigated_prob = apply_zne(
        ...     transpiled_circuit, backend, solver, shots=4096
        ... )
        >>> print(f"Mitigated success: {mitigated_prob:.2%}")
        
        Comparing raw vs mitigated:
        
        >>> # Raw execution
        >>> handle = backend.process_circuit(transpiled_circuit, n_shots=4096)
        >>> result = backend.get_result(handle)
        >>> counts = result.get_counts()
        >>> raw_prob = compute_success_expectation(counts, solver._is_valid_solution)
        >>> 
        >>> # Mitigated execution
        >>> mitigated_prob = apply_zne(transpiled_circuit, backend, solver, shots=4096)
        >>> 
        >>> improvement = mitigated_prob - raw_prob
        >>> print(f"Raw: {raw_prob:.2%}, Mitigated: {mitigated_prob:.2%}")
        >>> print(f"Improvement: {improvement:+.2%}")
    
    See Also:
        - create_zne_executor(): Lower-level executor creation
        - QuantumSolver.run(): High-level interface with use_zne parameter
        - apply_pec(): PEC alternative to ZNE
    """
    if not MITIQ_AVAILABLE:
        raise ImportError("Mitiq is required for ZNE")
    
    executor = create_zne_executor(
        backend, solver, shots, scale_noise, factory, **kwargs
    )
    
    # Convert pytket circuit to Qiskit for Mitiq compatibility
    # Mitiq supports Qiskit, Cirq, and Braket natively
    from pytket import Circuit
    mitiq_circuit = circuit
    if isinstance(circuit, Circuit):
        try:
            from pytket.extensions.qiskit import tk_to_qiskit
            mitiq_circuit = tk_to_qiskit(circuit)
        except ImportError:
            # If conversion not available, try as-is
            pass
    
    # TODO: Make scale_factors and factory configurable
    # Default: fold circuit at scales [1, 3, 5] and use Richardson extrapolation
    # Only pass scale_noise and factory if explicitly provided (Mitiq requires callables, not None)
    zne_kwargs = {}
    if scale_noise is not None:
        zne_kwargs['scale_noise'] = scale_noise
    if factory is not None:
        zne_kwargs['factory'] = factory
    
    return zne.execute_with_zne(
        mitiq_circuit,
        executor,
        **zne_kwargs
    )


def apply_pec(
    circuit: Any,
    backend: Any,
    solver: Any,
    representations: Any,
    shots: int = 1024,
    **kwargs
) -> float:
    """Apply Probabilistic Error Cancellation to a quantum circuit.
    
    Convenience function for PEC execution. This is used internally by
    QuantumSolver.run() when use_pec=True.
    
    PEC samples many circuits according to quasiprobability distributions
    derived from noisy gate decompositions, making it more expensive but
    potentially more accurate than ZNE.
    
    Args:
        circuit: Quantum circuit to execute with mitigation. Should be the
            transpiled/compiled circuit ready for backend execution (pytket.Circuit).
        backend: Quantum backend instance from BackendManager.
            Supported: Qiskit backends (IBM, Aer) or pytket backends (Quantinuum).
            NOT supported: AWS Braket.
        solver: Quantum solver with _is_valid_solution method for validating
            measurement outcomes.
        representations: OperationRepresentation list mapping ideal gates to
            noisy implementations. REQUIRED for PEC. Must be generated from
            backend calibration data or noise models.
            TODO: Provide helper functions to auto-generate from backend.
        shots (int): Number of shots per sampled circuit. PEC samples many circuits,
            so total cost ≈ shots × N_samples. Default: 1024.
            Recommended: 8192+ for real hardware.
        **kwargs: Additional executor parameters passed to backend.process_circuit().
    
    Returns:
        float: Mitigated success probability in [0, 1], representing the
            unbiased estimate of Pr[valid exact cover solution] after error
            cancellation.
    
    Raises:
        ImportError: If Mitiq is not installed.
        ValueError: If representations is None.
    
    Warning:
        PEC is significantly more expensive than ZNE (often 100-1000x more shots).
        Use ZNE first for initial experiments. PEC is most useful when:
        - You have accurate noise characterization (representations)
        - You need unbiased estimates (ZNE can have systematic errors)
        - You have sufficient quantum runtime budget
    
    Example:
        Basic PEC usage:
        
        >>> from sudoku_nisq.backends import BackendManager
        >>> from sudoku_nisq.mitigation.executors import apply_pec
        >>> 
        >>> # Generate representations (user must provide this)
        >>> # This requires backend noise characterization
        >>> representations = generate_representations_from_calibration(backend)
        >>> 
        >>> manager = BackendManager()
        >>> backend = manager.get("ibm_brisbane")
        >>> 
        >>> transpiled_circuit = backend.get_compiled_circuit(
        ...     grover_circuit, optimisation_level=2
        ... )
        >>> 
        >>> mitigated_prob = apply_pec(
        ...     transpiled_circuit, backend, solver, 
        ...     representations, shots=8192
        ... )
        >>> print(f"PEC mitigated: {mitigated_prob:.2%}")
        
        Cost comparison ZNE vs PEC:
        
        >>> import time
        >>> 
        >>> # ZNE: ~3-5 circuit executions
        >>> t0 = time.time()
        >>> zne_result = apply_zne(circuit, backend, solver, shots=4096)
        >>> zne_time = time.time() - t0
        >>> 
        >>> # PEC: ~100-1000 circuit executions  
        >>> t0 = time.time()
        >>> pec_result = apply_pec(circuit, backend, solver, reps, shots=4096)
        >>> pec_time = time.time() - t0
        >>> 
        >>> print(f"ZNE: {zne_result:.2%} in {zne_time:.1f}s")
        >>> print(f"PEC: {pec_result:.2%} in {pec_time:.1f}s")
        >>> print(f"PEC is {pec_time/zne_time:.0f}x slower")
    
    See Also:
        - create_pec_executor(): Lower-level executor creation
        - apply_zne(): Simpler and cheaper alternative
        - QuantumSolver.run(): High-level interface with use_pec parameter
        - Mitiq PEC tutorial: https://mitiq.readthedocs.io/en/stable/guide/pec.html
    """
    if not MITIQ_AVAILABLE:
        raise ImportError("Mitiq is required for PEC")
    
    executor = create_pec_executor(
        backend, solver, shots, representations, **kwargs
    )
    
    # Convert pytket circuit to Qiskit for Mitiq compatibility
    from pytket import Circuit
    mitiq_circuit = circuit
    if isinstance(circuit, Circuit):
        try:
            from pytket.extensions.qiskit import tk_to_qiskit
            mitiq_circuit = tk_to_qiskit(circuit)
        except ImportError:
            pass
    
    return pec.execute_with_pec(mitiq_circuit, executor, representations=representations)
