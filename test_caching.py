#!/usr/bin/env python3
"""
Test script to verify that the get_main_circuit() method properly caches circuits to disk.
"""

import os
import sys
import tempfile
import shutil
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from sudoku_nisq.quantum_solver import QuantumSolver
from pytket import Circuit

# Set up logging to see cache operations
logging.basicConfig(level=logging.DEBUG)

class TestSolver(QuantumSolver):
    """Simple test solver that creates a basic circuit."""
    
    def get_circuit(self):
        """Return a simple 2-qubit circuit for testing."""
        circuit = Circuit(2)
        circuit.H(0)
        circuit.CX(0, 1)
        circuit.measure_all()
        return circuit
    
    def resource_estimation(self):
        """Return basic resource estimation."""
        return {
            'qubits': 2,
            'gates': 3,
            'depth': 2
        }

def test_circuit_caching():
    """Test that get_main_circuit() properly caches to disk."""
    
    # Create a temporary directory for cache testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set the cache directory to our temp directory
        os.environ['QUANTUM_SOLVER_CACHE_DIR'] = temp_dir
        
        print(f"Testing with cache directory: {temp_dir}")
        
        # Create a test solver
        solver = TestSolver()
        
        # Check that cache directory doesn't exist yet
        expected_cache_path = os.path.join(temp_dir, 'TestSolver', 'no_puzzle', 'main_circuit.json')
        print(f"Expected cache file: {expected_cache_path}")
        
        # First call - should build circuit and save to cache
        print("\n1. First call to get_main_circuit():")
        circuit1 = solver.get_main_circuit()
        print(f"Circuit has {circuit1.n_qubits} qubits and {circuit1.n_gates} gates")
        
        # Check if cache file was created
        if os.path.exists(expected_cache_path):
            print("✓ Cache file was created successfully!")
        else:
            print("✗ Cache file was NOT created")
            print(f"Cache directory contents: {os.listdir(temp_dir) if os.path.exists(temp_dir) else 'Directory does not exist'}")
            # List all files recursively
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    print(f"  Found file: {os.path.join(root, file)}")
        
        # Second call - should load from cache
        print("\n2. Second call to get_main_circuit():")
        circuit2 = solver.get_main_circuit()
        print(f"Circuit has {circuit2.n_qubits} qubits and {circuit2.n_gates} gates")
        
        # Verify circuits are equivalent
        if circuit1.n_qubits == circuit2.n_qubits and circuit1.n_gates == circuit2.n_gates:
            print("✓ Cached circuit matches original")
        else:
            print("✗ Cached circuit does not match original")
        
        # Test force rebuild
        print("\n3. Third call with force_rebuild=True:")
        circuit3 = solver.get_main_circuit(force_rebuild=True)
        print(f"Circuit has {circuit3.n_qubits} qubits and {circuit3.n_gates} gates")
        
        print(f"\nCache file exists: {os.path.exists(expected_cache_path)}")
        if os.path.exists(expected_cache_path):
            file_size = os.path.getsize(expected_cache_path)
            print(f"Cache file size: {file_size} bytes")

if __name__ == "__main__":
    test_circuit_caching()
