"""Integration tests for Phase 3 usability features.

Tests the complete workflow of solution enumeration, validation context creation,
and metrics computation for the user-friendly API.
"""

import pytest
from sudoku_nisq.sudoku_puzzle import SudokuPuzzle
from sudoku_nisq.exact_cover_problem import ExactCoverProblem
from sudoku_nisq.metrics.data_models import ValidationContext


class TestSolutionEnumeration:
    """Test solution enumeration for Sudoku and exact cover problems."""
    
    def test_sudoku_enumerate_solutions_2x2(self):
        """Test that 2×2 puzzle can enumerate solutions."""
        # Create a simple 2×2 puzzle with 2 givens
        board = [[1, 0], [0, 1]]
        puzzle = SudokuPuzzle.from_board(board)
        
        solutions = puzzle.enumerate_solutions(max_solutions=10)
        
        assert len(solutions) > 0, "Should find at least one solution"
        assert len(solutions) <= 10, "Should respect max_solutions limit"
        
        # Check that solutions are complete (no zeros)
        for solution in solutions:
            for row in solution:
                for cell in row:
                    assert cell != 0, "Solution should have no empty cells"
    
    def test_exact_cover_enumerate_solutions(self):
        """Test that exact cover problem can enumerate solutions."""
        problem = ExactCoverProblem.create_small_example()
        
        solutions = problem.enumerate_solutions(max_solutions=10)
        
        assert len(solutions) == 1, "Small example should have 1 solution"
        assert isinstance(solutions[0], list), "Solution should be a list"
        assert all(isinstance(key, str) for key in solutions[0]), "Keys should be strings"


class TestValidationContextCreation:
    """Test validation context creation from solution enumeration."""
    
    def test_sudoku_create_validation_context(self):
        """Test creating validation context from Sudoku puzzle."""
        # Create a simple 2×2 puzzle
        board = [[1, 0], [0, 1]]
        puzzle = SudokuPuzzle.from_board(board)
        
        context = puzzle.create_validation_context(encoding_type='simple', max_solutions=10)
        
        assert isinstance(context, ValidationContext)
        assert len(context.valid_solutions) > 0
        assert context.total_valid_count == len(context.valid_solutions)
        assert callable(context.solution_validator)
        
        # Test validator works
        first_solution = context.valid_solutions[0]
        assert context.solution_validator(first_solution), "Should validate own solutions"
        assert not context.solution_validator("0" * 100), "Should reject invalid bitstrings"
    
    def test_exact_cover_create_validation_context(self):
        """Test creating validation context from exact cover problem."""
        problem = ExactCoverProblem.create_small_example()
        
        context = problem.create_validation_context(max_solutions=10)
        
        assert isinstance(context, ValidationContext)
        assert len(context.valid_solutions) == 1
        assert context.total_valid_count == 1
        assert callable(context.solution_validator)
    
    def test_validation_context_cached(self):
        """Test that validation context is cached."""
        board = [[1, 0], [0, 1]]
        puzzle = SudokuPuzzle.from_board(board)
        
        puzzle.create_validation_context(encoding_type='simple')
        puzzle.create_validation_context(encoding_type='simple')
        
        # Should be cached (same object or at least same data)
        assert hasattr(puzzle, '_cached_validation_contexts')
        assert 'simple' in puzzle._cached_validation_contexts


class TestBitstringConversion:
    """Test bitstring conversion from solution boards."""
    
    def test_simple_encoding_conversion(self):
        """Test that solutions convert to valid bitstrings."""
        # Create a simple 2×2 puzzle
        board = [[1, 0], [0, 1]]
        puzzle = SudokuPuzzle.from_board(board)
        
        # Enumerate solutions
        solutions = puzzle.enumerate_solutions(max_solutions=5)
        assert len(solutions) > 0
        
        # Try to convert to bitstrings via validation context
        context = puzzle.create_validation_context(encoding_type='simple')
        
        # Check that bitstrings are valid format
        for bitstring in context.valid_solutions:
            assert isinstance(bitstring, str)
            assert all(c in '01' for c in bitstring), "Bitstring should be binary"


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Test complete workflow from puzzle generation to metrics (without actual quantum execution)."""
    
    def test_workflow_without_execution(self):
        """Test that all components work together."""
        # Step 1: Create puzzle
        board = [[1, 0], [0, 1]]
        puzzle = SudokuPuzzle.from_board(board)
        
        # Step 2: Enumerate solutions
        solutions = puzzle.enumerate_solutions(max_solutions=10)
        assert len(solutions) > 0
        
        # Step 3: Create validation context
        context = puzzle.create_validation_context(encoding_type='simple')
        assert isinstance(context, ValidationContext)
        assert len(context.valid_solutions) > 0
        
        # Step 4: Verify validation works
        for bitstring in context.valid_solutions:
            assert context.solution_validator(bitstring)
        
        print(f"✓ Found {len(solutions)} solutions")
        print(f"✓ Converted to {len(context.valid_solutions)} valid bitstrings")
        print("✓ Validator function works correctly")


if __name__ == "__main__":
    # Run basic smoke tests
    test = TestSolutionEnumeration()
    test.test_sudoku_enumerate_solutions_2x2()
    test.test_exact_cover_enumerate_solutions()
    
    test2 = TestValidationContextCreation()
    test2.test_sudoku_create_validation_context()
    test2.test_exact_cover_create_validation_context()
    
    test3 = TestEndToEndWorkflow()
    test3.test_workflow_without_execution()
    
    print("\n✅ All basic tests passed!")
