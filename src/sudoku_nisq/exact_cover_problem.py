"""
Generic Exact Cover Problem representation and utilities.

This module provides:
- ExactCoverProblem: A dataclass representing any exact cover instance
- Canonicalization: Mapping labeled constraints to indexed universe {0, ..., n-1}
- Instance generation: Enumerating small exact cover instances for benchmarking
"""

import hashlib
from dataclasses import dataclass
from itertools import product
from typing import List, Dict, Any, Tuple, Set, Optional


@dataclass
class ExactCoverProblem:
    """
    Represents a generic exact cover problem instance.
    
    An exact cover problem consists of:
    - Universe U: A set of elements that must be covered
    - Subsets S: A collection of subsets of U
    
    The goal is to find a subcollection of S that covers each element of U exactly once.
    
    Attributes:
        universe: List of universe elements (can be any hashable objects)
        subsets: Dictionary mapping subset keys to lists of universe elements
        num_solutions: Expected number of solutions (optional, for resource estimation)
        metadata: Optional metadata (e.g., problem source, generation parameters)
    
    Example:
        >>> # Classic example: covering {1, 2, 3, 4} with subsets
        >>> universe = [1, 2, 3, 4]
        >>> subsets = {
        ...     'S_0': [1, 2],
        ...     'S_1': [2, 3],
        ...     'S_2': [3, 4],
        ...     'S_3': [1, 4]
        ... }
        >>> problem = ExactCoverProblem(universe, subsets, num_solutions=2)
        >>> problem.validate()  # Checks consistency
    """
    
    universe: List[Any]
    subsets: Dict[str, List[Any]]
    num_solutions: Optional[int] = 1
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate the exact cover problem after initialization."""
        self.validate()
    
    def validate(self):
        """
        Validate the exact cover problem structure.
        
        Checks:
        1. Universe is non-empty
        2. Subsets is non-empty
        3. All subset elements are in the universe
        4. No duplicate elements within individual subsets
        
        Raises:
            ValueError: If validation fails
        """
        if not self.universe:
            raise ValueError("Universe cannot be empty")
        
        if not self.subsets:
            raise ValueError("Subsets cannot be empty")
        
        universe_set = set(self.universe)
        
        # Check that universe has no duplicates
        if len(universe_set) != len(self.universe):
            raise ValueError("Universe contains duplicate elements")
        
        # Validate each subset
        for key, subset in self.subsets.items():
            if not subset:
                raise ValueError(f"Subset {key} is empty")
            
            # Check for duplicates within subset
            if len(subset) != len(set(subset)):
                raise ValueError(f"Subset {key} contains duplicate elements")
            
            # Check that all subset elements are in universe
            for elem in subset:
                if elem not in universe_set:
                    raise ValueError(
                        f"Subset {key} contains element {elem} not in universe"
                    )
    
    def canonicalize(self) -> Tuple[int, List[Set[int]]]:
        """
        Convert this exact cover problem to canonical form.
        
        Maps labeled universe elements to indices {0, ..., n-1} and represents
        subsets as sets of integers. This allows comparison of problems that
        differ only in element labels (they are isomorphic as exact cover problems).
        
        Returns:
            Tuple of:
                - n (int): Size of universe
                - canonical_subsets (List[Set[int]]): Subsets as index sets
        
        Example:
            >>> universe = ['a', 'b', 'c']
            >>> subsets = {'S_0': ['a', 'b'], 'S_1': ['b', 'c']}
            >>> problem = ExactCoverProblem(universe, subsets)
            >>> n, canon = problem.canonicalize()
            >>> n
            3
            >>> canon
            [{0, 1}, {1, 2}]  # 'a'->0, 'b'->1, 'c'->2
        """
        # Assign indices to universe elements
        elem_to_idx = {elem: i for i, elem in enumerate(self.universe)}
        n = len(self.universe)
        
        # Map subsets to index sets (preserve order of subsets dict)
        canonical_subsets = []
        for subset_key in sorted(self.subsets.keys()):  # Sort for deterministic ordering
            idx_set = {elem_to_idx[elem] for elem in self.subsets[subset_key]}
            canonical_subsets.append(idx_set)
        
        return n, canonical_subsets
    
    def get_hash(self) -> str:
        """
        Compute a hash of the canonical representation for caching/metadata.
        
        Two isomorphic exact cover problems (differing only in labels) will
        produce the same hash. Useful for MetadataManager integration.
        
        Returns:
            str: Hexadecimal hash string
        
        Example:
            >>> problem1 = ExactCoverProblem([1, 2, 3], {'S_0': [1, 2]})
            >>> problem2 = ExactCoverProblem(['a', 'b', 'c'], {'S_0': ['a', 'b']})
            >>> problem1.get_hash() == problem2.get_hash()
            True  # Isomorphic problems have same hash
        """
        n, canonical_subsets = self.canonicalize()
        
        # Create a canonical string representation
        # Format: "n;<subset1>;<subset2>;..."
        # Each subset is sorted indices joined by commas
        subset_strs = []
        for idx_set in canonical_subsets:
            sorted_indices = sorted(idx_set)
            subset_strs.append(','.join(map(str, sorted_indices)))
        
        canonical_str = f"{n};{';'.join(subset_strs)}"
        
        # Hash it
        return hashlib.sha256(canonical_str.encode()).hexdigest()
    
    def to_incidence_matrix(self) -> List[List[int]]:
        """
        Convert to incidence matrix representation.
        
        Returns an n×m binary matrix A where:
        - n = len(universe)
        - m = len(subsets)
        - A[i][j] = 1 if universe[i] is in subsets[j], else 0
        
        Returns:
            List[List[int]]: Binary matrix (n rows, m columns)
        
        Example:
            >>> universe = [1, 2, 3]
            >>> subsets = {'S_0': [1, 2], 'S_1': [2, 3]}
            >>> problem = ExactCoverProblem(universe, subsets)
            >>> problem.to_incidence_matrix()
            [[1, 0],
             [1, 1],
             [0, 1]]
        """
        n = len(self.universe)
        m = len(self.subsets)
        
        # Create index mapping
        elem_to_idx = {elem: i for i, elem in enumerate(self.universe)}
        
        # Initialize matrix
        matrix = [[0] * m for _ in range(n)]
        
        # Fill matrix
        for j, (subset_key, subset) in enumerate(sorted(self.subsets.items())):
            for elem in subset:
                i = elem_to_idx[elem]
                matrix[i][j] = 1
        
        return matrix
    
    def to_canonical_matrix(self) -> Tuple[List[List[int]], List[Any]]:
        """Return a canonical incidence matrix for this instance.

        The canonical matrix is built by:
        - sorting the universe labels;
        - converting each subset into a column bitvector over the sorted universe;
        - dropping duplicate columns; and
        - sorting the remaining columns lexicographically.

        Returns:
            A tuple ``(canonical_matrix, ordered_universe)`` where ``canonical_matrix``
            is an ``n x m*`` binary matrix in row-major order and ``ordered_universe``
            is the sorted list of universe elements.
        """
        # Step 1: Order universe elements canonically
        try:
            ordered_universe = sorted(self.universe)
        except TypeError:
            # Fallback for non-comparable types (use string representation)
            ordered_universe = sorted(self.universe, key=str)
        
        n = len(ordered_universe)
        {elem: i for i, elem in enumerate(ordered_universe)}
        
        # Step 2: Convert each subset to a column bitvector
        columns: List[Tuple[str, Tuple[int, ...]]] = []  # (bitstring, bitvector)
        
        for subset_key in sorted(self.subsets.keys()):
            subset = self.subsets[subset_key]
            # Build column bitvector: (b_j)_i = 1 iff u_i ∈ S_j
            col_bits = tuple(
                1 if ordered_universe[i] in subset else 0 
                for i in range(n)
            )
            col_bitstring = ''.join(map(str, col_bits))
            columns.append((col_bitstring, col_bits))
        
        # Step 3: Remove duplicate columns
        seen_bitstrings = set()
        unique_columns = []
        for bitstring, col_bits in columns:
            if bitstring not in seen_bitstrings:
                seen_bitstrings.add(bitstring)
                unique_columns.append(col_bits)
        
        # Step 4: Sort columns lexicographically
        unique_columns.sort()
        
        # Convert to matrix format (row-major: list of rows)
        m = len(unique_columns)
        canonical_matrix: list[list[int]]
        if m == 0:
            canonical_matrix = [[] for _ in range(n)]
        else:
            canonical_matrix = [
                [unique_columns[j][i] for j in range(m)] 
                for i in range(n)
            ]
        
        return canonical_matrix, ordered_universe
    
    def to_canonical_encoding(self) -> str:
        """Encode the canonical matrix as a shortlex-stable bitstring.

        The encoding is ``un(n) + un(m) + bits(A*)`` where ``un(k)`` is unary
        (``"1" * k + "0"``) and ``bits(A*)`` are the entries of the canonical
        incidence matrix in row-major order. Isomorphic instances share the same
        encoding.
        """
        canonical_matrix, _ = self.to_canonical_matrix()
        
        if not canonical_matrix:
            n, m = 0, 0
        else:
            n = len(canonical_matrix)
            m = len(canonical_matrix[0]) if canonical_matrix else 0
        
        # Helper: unary encoding un(k) = "1"*k + "0"
        def unary(k: int) -> str:
            if k < 0:
                raise ValueError("k must be non-negative")
            if k == 0:
                return "0"
            return "1" * k + "0"
        
        # Build encoding: un(n) ; un(m) ; bits(A*)
        parts = [unary(n), unary(m)]
        
        # Append matrix bits in row-major order
        for row in canonical_matrix:
            for bit in row:
                parts.append(str(bit))
        
        return ''.join(parts)
    
    def canonical_order_index(self) -> int:
        """
        Compute the position of this instance in the global shortlex order.
        
        The shortlex order on binary strings is:
        - First by length (shorter < longer)
        - Then lexicographically (for equal length)
        
        Since canonical encodings are always non-empty (they include at least un(n)
        and un(m)), we enumerate non-empty binary strings starting from index 1.
        
        For a non-empty string s of length ℓ ≥ 1:
            index(s) = (2^ℓ - 1) + int(s, 2)
        
        This gives indices in the range [2^ℓ - 1, 2^(ℓ+1) - 2] for strings of length ℓ.
        
        Returns:
            int: The index in the global ordering (starting from 1 for "0")
        
        Example:
            >>> # Small encodings have manageable indices
            >>> # Larger problems have exponentially larger indices
            >>> problem = ExactCoverProblem.from_incidence_matrix([[1, 0], [0, 1]])
            >>> idx = problem.canonical_order_index()
            >>> # idx = 1893 for encoding "1101100110" (length 10)
        
        Note:
            For non-trivial instances, this index can be astronomically large
            (e.g., >10^1000 for a moderate-sized problem). This is primarily
            useful for:
            - Theoretical analysis
            - Small instance comparison
            - Formal proofs about problem space structure
        
        Warning:
            Do not attempt to enumerate all instances up to a large index!
            The number of instances grows doubly exponentially.
        """
        encoding = self.to_canonical_encoding()
        length = len(encoding)
        
        # For non-empty strings (which all canonical encodings are):
        # Count all shorter strings: sum_{i=1}^{length-1} 2^i = 2^length - 2
        # Then add position within strings of this length: int(encoding, 2)
        # Combined: (2^length - 1) + int(encoding, 2)
        
        count_shorter = (1 << length) - 1
        lex_position = int(encoding, 2)
        
        return count_shorter + lex_position
    
    @classmethod
    def from_incidence_matrix(cls, matrix: List[List[int]], 
                             num_solutions: Optional[int] = None) -> 'ExactCoverProblem':
        """
        Create an ExactCoverProblem from an incidence matrix.
        
        Args:
            matrix: Binary n×m matrix where A[i][j] = 1 if element i is in subset j
            num_solutions: Optional expected number of solutions
        
        Returns:
            ExactCoverProblem: Problem with integer-labeled universe {0, ..., n-1}
        
        Example:
            >>> matrix = [[1, 0], [1, 1], [0, 1]]
            >>> problem = ExactCoverProblem.from_incidence_matrix(matrix)
            >>> problem.universe
            [0, 1, 2]
            >>> problem.subsets
            {'S_0': [0, 1], 'S_1': [1, 2]}
        """
        if not matrix:
            raise ValueError("Matrix cannot be empty")
        
        n = len(matrix)  # number of universe elements
        m = len(matrix[0]) if matrix else 0  # number of subsets
        
        # Validate matrix dimensions
        if not all(len(row) == m for row in matrix):
            raise ValueError("Matrix rows must have consistent length")
        
        # Universe is {0, 1, ..., n-1}
        universe = list(range(n))
        
        # Build subsets from columns
        subsets = {}
        for j in range(m):
            subset = [i for i in range(n) if matrix[i][j] == 1]
            subsets[f'S_{j}'] = subset
        
        return cls(universe=universe, subsets=subsets, num_solutions=num_solutions)
    
    @staticmethod
    def enumerate_instances(max_n: int = 5, max_m: int = 5, 
                          max_total: int = 20):
        """
        Generator that enumerates small exact cover instances.
        
        Iterates over all possible n×m binary matrices (incidence matrices)
        up to specified limits. Useful for benchmarking and testing.
        
        Args:
            max_n: Maximum universe size
            max_m: Maximum number of subsets
            max_total: Maximum n*m (total matrix size, to avoid explosion)
        
        Yields:
            ExactCoverProblem: Successive exact cover instances
        
        Example:
            >>> # Generate all 2×2 exact cover instances
            >>> gen = ExactCoverProblem.enumerate_instances(max_n=2, max_m=2)
            >>> problems = list(gen)
            >>> len(problems)
            16  # 2^(2*2) = 16 possible 2×2 matrices
        
        Note:
            This is a conceptual enumeration. For n=5, m=5, this generates
            2^25 ≈ 33 million instances. Use with caution!
        """
        for n in range(1, max_n + 1):
            for m in range(1, max_m + 1):
                if n * m > max_total:
                    continue  # Skip large instances
                
                # Iterate over all n*m binary matrices
                for bits in product([0, 1], repeat=n * m):
                    # Reshape into n×m matrix
                    matrix = [list(bits[i*m:(i+1)*m]) for i in range(n)]
                    
                    # Skip if any column (subset) is empty
                    if any(all(matrix[i][j] == 0 for i in range(n)) for j in range(m)):
                        continue
                    
                    # Create problem from matrix
                    try:
                        problem = ExactCoverProblem.from_incidence_matrix(matrix)
                        yield problem
                    except ValueError:
                        # Skip invalid problems (e.g., empty subsets)
                        continue
    
    def count_solutions(self, max_solutions: Optional[int] = None) -> int:
        """Count exact covers with a pruning backtracking search.

        The search branches on the element with the fewest covering subsets and
        stops early when ``max_solutions`` is reached.

        Args:
            max_solutions: Stop after this many solutions (helps SAT-style checks).

        Returns:
            Number of solutions found (capped at ``max_solutions`` when set).
        """
        # Map each element to the subset keys that contain it
        elem_to_subsets: Dict[Any, List[str]] = {}
        for key, elems in self.subsets.items():
            for e in elems:
                elem_to_subsets.setdefault(e, []).append(key)

        # Precompute subset contents as sets for fast intersection/subset checks
        subset_elems: Dict[str, Set[Any]] = {
            key: set(elems) for key, elems in self.subsets.items()
        }

        remaining = set(self.universe)

        def backtrack(rem: Set[Any]) -> int:
            # All elements covered → one exact cover found
            if not rem:
                return 1

            # Choose an element with the smallest branching factor
            best_candidates: Optional[List[str]] = None

            for e in rem:
                # Subsets that:
                # - contain e
                # - do not use elements outside 'rem' (no overlaps with already-covered)
                candidates = [
                    key for key in elem_to_subsets.get(e, [])
                    if subset_elems[key] <= rem
                ]

                # If no subset can cover this element, dead end
                if not candidates:
                    return 0

                if best_candidates is None or len(candidates) < len(best_candidates):
                    best_candidates = candidates
                    if len(best_candidates) == 1:
                        break  # can't do better than 1

            total = 0
            assert best_candidates is not None  # for type checkers

            # Try each subset that covers best_elem
            for key in best_candidates:
                new_rem = rem - subset_elems[key]
                total += backtrack(new_rem)

                # Early stop if we hit the cap
                if max_solutions is not None and total >= max_solutions:
                    return total

            return total

        return backtrack(remaining)
    
    def enumerate_solutions(self, max_solutions: int = 100) -> list[list[str]]:
        """Enumerate exact cover solutions using backtracking.
        
        Collects actual solution subset keys instead of just counting. Each solution
        is a list of subset keys whose elements partition the universe exactly once.
        
        Args:
            max_solutions: Maximum number of solutions to collect. Default 100 is
                reasonable for small exact cover problems. Larger problems may have
                exponentially many solutions.
        
        Returns:
            List of solutions, where each solution is a list of subset keys (strings).
            Returns empty list if no exact covers exist. May return fewer than
            max_solutions if problem has fewer valid solutions.
        
        Example:
            >>> problem = ExactCoverProblem.create_small_example()
            >>> solutions = problem.enumerate_solutions(max_solutions=10)
            >>> len(solutions)
            1  # Small example has 1 exact cover
            >>> solutions[0]
            ['S_0', 'S_2']  # These subsets partition the universe exactly
        
        Note:
            For large exact cover problems, enumeration can be computationally
            expensive. Consider using count_solutions() first to check feasibility.
        """
        solutions: List[List[str]] = []
        
        # Map each element to the subset keys that contain it
        elem_to_subsets: Dict[Any, List[str]] = {}
        for key, elems in self.subsets.items():
            for e in elems:
                elem_to_subsets.setdefault(e, []).append(key)
        
        # Precompute subset contents as sets for fast intersection/subset checks
        subset_elems: Dict[str, Set[Any]] = {
            key: set(elems) for key, elems in self.subsets.items()
        }
        
        remaining = set(self.universe)
        
        def backtrack(rem: Set[Any], chosen: List[str]) -> None:
            """Recursive backtracking that collects solutions."""
            if len(solutions) >= max_solutions:
                return  # Early stopping
            
            # All elements covered → found an exact cover
            if not rem:
                solutions.append(chosen[:])  # Copy the current solution
                return
            
            # Choose an element with the smallest branching factor
            best_candidates: Optional[List[str]] = None
            
            for e in rem:
                # Subsets that contain e and don't use elements outside rem
                candidates = [
                    key for key in elem_to_subsets.get(e, [])
                    if subset_elems[key] <= rem
                ]
                
                # If no subset can cover this element, dead end
                if not candidates:
                    return
                
                if best_candidates is None or len(candidates) < len(best_candidates):
                    best_candidates = candidates
                    if len(best_candidates) == 1:
                        break  # Can't do better than 1
            
            assert best_candidates is not None  # For type checkers
            
            # Try each subset that covers best_elem
            for key in best_candidates:
                if len(solutions) >= max_solutions:
                    return  # Early stopping
                
                new_rem = rem - subset_elems[key]
                chosen.append(key)
                backtrack(new_rem, chosen)
                chosen.pop()  # Backtrack
        
        backtrack(remaining, [])
        return solutions
    
    @staticmethod
    def create_small_example() -> 'ExactCoverProblem':
        """Return a 4-element toy instance with one exact cover solution."""
        universe = [0, 1, 2, 3]
        subsets = {
            'S_0': [0, 3],
            'S_1': [0, 1, 2],
            'S_2': [1, 2],
            'S_3': [2, 3],
            'S_4': [0],
            'S_5': [1, 3]
        }
        return ExactCoverProblem(
            universe=universe,
            subsets=subsets,
            num_solutions=1,
            metadata={'description': 'Small example with 1 solution'}
        )
    
    def create_validation_context(self, max_solutions: int = 100):
        """Create a ValidationContext for metrics computation from solution enumeration.
        
        Enumerates exact cover solutions and packages them into a ValidationContext
        suitable for automatic metrics computation during quantum execution.
        
        Args:
            max_solutions: Maximum number of solutions to enumerate. Default 100.
        
        Returns:
            ValidationContext dataclass with:
            - valid_solutions: List of bitstrings (one per exact cover solution)
            - total_valid_count: Number of valid solutions found (may be capped)
            - solution_validator: Function to check if a bitstring is valid
        
        Raises:
            ValueError: If problem has no exact covers.
        
        Example:
            >>> problem = ExactCoverProblem.create_small_example()
            >>> context = problem.create_validation_context()
            >>> len(context.valid_solutions)
            1  # Small example has 1 exact cover
        
        Note:
            For exact cover problems, bitstrings are constructed by checking
            which subsets appear in each solution.
        """
        from sudoku_nisq.metrics.data_models import ValidationContext
        
        # Enumerate solutions
        solutions = self.enumerate_solutions(max_solutions=max_solutions)
        
        if not solutions:
            raise ValueError("Problem has no exact covers. Cannot create ValidationContext.")
        
        # Convert solutions to bitstrings
        # Each solution is a list of subset keys
        # Bitstring has '1' at position i if subset S_i is selected
        subset_keys = sorted(self.subsets.keys())  # Consistent ordering
        key_to_index = {key: idx for idx, key in enumerate(subset_keys)}
        n_subsets = len(subset_keys)
        
        valid_bitstrings = []
        for solution in solutions:
            bitstring = ['0'] * n_subsets
            for key in solution:
                if key in key_to_index:
                    bitstring[key_to_index[key]] = '1'
            valid_bitstrings.append(''.join(bitstring))
        
        # Create validator function
        valid_set = set(valid_bitstrings)
        def validator(bitstring: str) -> bool:
            return bitstring in valid_set
        
        # Create and return ValidationContext
        context = ValidationContext(
            valid_solutions=valid_bitstrings,
            total_valid_count=len(valid_bitstrings),
            solution_validator=validator
        )
        
        return context
