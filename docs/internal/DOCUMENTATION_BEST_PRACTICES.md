# How to Document Code Changes: Best Practices Guide

This guide explains the standard practices for documenting code changes in software projects, specifically tailored for the sudoku-nisq-benchmark project but applicable to any professional software development.

## Documentation Hierarchy

Professional projects maintain documentation at multiple levels:

### 1. **Commit Messages** (Micro-level)
- **Purpose**: Explain what changed and why
- **Audience**: Developers reviewing git history
- **Lifespan**: Permanent part of git history

### 2. **Pull Request Descriptions** (Feature-level)
- **Purpose**: Provide context for code review
- **Audience**: Code reviewers and future developers
- **Lifespan**: Permanent, searchable reference

### 3. **Code Comments & Docstrings** (Implementation-level)
- **Purpose**: Explain how code works
- **Audience**: Developers reading/maintaining code
- **Lifespan**: Lives with the code

### 4. **User Documentation** (Product-level)
- **Purpose**: Explain how to use features
- **Audience**: End users and API consumers
- **Lifespan**: Updated with each release

### 5. **Implementation Documents** (Architecture-level)
- **Purpose**: Document design decisions and rationale
- **Audience**: Maintainers and contributors
- **Lifespan**: Reference material for complex features

## 1. Writing Good Commit Messages

### Standard Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Commit Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code restructuring without behavior change
- `perf`: Performance improvements
- `test`: Adding or fixing tests
- `chore`: Build process, dependencies, tooling
- `ci`: Continuous integration changes

### Examples

**Good commit message:**
```
feat(providers): add provider pattern for quantum backends

Refactored BackendManager to use a provider pattern, separating
quantum platform-specific code into individual provider classes.

- Created QuantumProvider abstract base class
- Implemented IBMProvider and QuantinuumProvider
- Updated BackendManager to use provider registry
- Maintained backward compatibility with existing API

This change makes it easier to add new quantum platforms without
modifying existing code and improves testability.

Closes #123
```

**Bad commit message:**
```
updated stuff
```

### Commit Message Best Practices

✅ **DO**:
- Use imperative mood ("add feature" not "added feature")
- Capitalize first letter of subject
- Keep subject line under 50 characters
- Separate subject from body with blank line
- Wrap body at 72 characters
- Explain **what** and **why**, not **how**
- Reference issue numbers when applicable

❌ **DON'T**:
- Use vague descriptions ("fix stuff", "update code")
- Include implementation details in subject
- End subject with period
- Mix multiple unrelated changes in one commit

## 2. Creating Pull Requests

### PR Title Format

Same as commit message subjects:
```
feat(scope): clear description of change
```

### PR Description Template

```markdown
## Summary
Brief overview of what this PR does (1-2 sentences)

## Motivation
Why is this change needed? What problem does it solve?

## Changes
- Bullet list of specific changes made
- Each item should be concrete and testable
- Group related changes together

## Testing
- How was this tested?
- What test cases were added?
- Any manual testing performed?

## Documentation
- [ ] User documentation updated
- [ ] Docstrings added/updated
- [ ] Examples provided
- [ ] CHANGELOG updated

## Breaking Changes
List any breaking changes and migration steps (if applicable)

## Related Issues
Closes #123
Related to #456

## Checklist
- [ ] Tests pass locally
- [ ] Code follows project style guide
- [ ] Documentation is updated
- [ ] Changes are backward compatible (or breaking changes documented)
```

### Example PR Description

```markdown
## Summary
Adds gate counting feature to track quantum gate usage during circuit construction.

## Motivation
Users need visibility into gate-level resource usage for algorithm analysis
and optimization. Current resource estimation only provides high-level metrics
like total gate count without gate type breakdown.

## Changes
- Added `GateCounter` class to track gates by type
- Updated circuit builders (PyTKET, Qiskit, Braket) to count gates
- Modified `ExactCoverQuantumSolver` to store and expose gate counts
- Added `get_gate_counts()` method to `QuantumSolver` base class
- Integrated gate counts into metadata and circuit caching
- Created comprehensive test suite (test_gate_counting.py)

## Testing
- Added 5 new unit tests covering all SDK implementations
- Verified gate counting accuracy against manual circuit analysis
- Tested metadata persistence and retrieval
- Confirmed negligible performance overhead (<1%)

## Documentation
- [x] User documentation updated (docs/guide/features.md)
- [x] Docstrings added to all new methods
- [x] Usage examples provided
- [x] Implementation guide created (GATE_COUNTING_IMPLEMENTATION.md)

## Breaking Changes
None - feature is additive and enabled by default

## Related Issues
Closes #89 - Request for gate-level resource tracking
Related to #102 - Circuit optimization analysis tools

## Checklist
- [x] Tests pass locally
- [x] Code follows project style guide
- [x] Documentation is updated
- [x] Changes are backward compatible
```

## 3. Writing Code Documentation

### Python Docstrings (reStructuredText)

Use Google-style or NumPy-style docstrings with reStructuredText syntax:

```python
def compute_circuit_depth(circuit, decompose_cnz=True):
    """
    Compute the depth of a quantum circuit.
    
    Depth is defined as the longest path from input to output,
    counting gate layers.
    
    :param circuit: The quantum circuit to analyze
    :type circuit: pytket.Circuit or qiskit.QuantumCircuit
    :param decompose_cnz: Whether to decompose multi-controlled Z gates
    :type decompose_cnz: bool
    :return: Circuit depth
    :rtype: int
    :raises ValueError: If circuit is empty or invalid
    
    .. code-block:: python
    
        from sudoku_nisq import QSudoku
        
        puzzle = QSudoku.generate(size=4)
        circuit = puzzle.build_circuit()
        depth = compute_circuit_depth(circuit)
        print(f"Circuit depth: {depth}")
    
    .. note::
        Decomposition affects depth calculation. With decompose_cnz=True,
        each CnZ gate adds 2 to depth (H + MCX + H).
    
    .. seealso::
        :func:`compute_gate_count` for gate-level analysis
    """
    if circuit is None or len(circuit) == 0:
        raise ValueError("Circuit must be non-empty")
    
    # Implementation...
    return depth
```

### Class Docstrings

```python
class ExactCoverQuantumSolver(QuantumSolver):
    """
    Quantum solver for exact cover problems using Grover's algorithm.
    
    This solver transforms exact cover instances into quantum circuits
    using Grover's search algorithm. It supports multiple encoding
    strategies (simple, pattern) and automatic SDK selection based on
    the target quantum backend.
    
    :ivar _encoding: Exact cover encoding instance
    :ivar _num_solutions: Expected number of solutions
    :ivar _gate_counts: Gate usage statistics
    :ivar _memory_usage: Memory profiling data (if tracking enabled)
    
    .. versionadded:: 0.2.0
        Added gate counting and memory tracking features
    
    .. code-block:: python
    
        from sudoku_nisq import QSudoku, ExactCoverQuantumSolver
        
        puzzle = QSudoku.generate(size=4, num_missing_cells=2)
        puzzle.set_solver(ExactCoverQuantumSolver, encoding="simple")
        circuit = puzzle.build_circuit()
        result = puzzle.run_aer(shots=1024)
    """
    
    def __init__(self, puzzle=None, encoding="simple", **kwargs):
        """
        Initialize the exact cover quantum solver.
        
        :param puzzle: Sudoku puzzle instance (optional for generic problems)
        :type puzzle: SudokuPuzzle or None
        :param encoding: Encoding strategy ('simple' or 'pattern')
        :type encoding: str
        :param kwargs: Additional solver configuration
        :raises ValueError: If encoding is not supported
        """
        # Implementation...
```

### Module Docstrings

```python
"""
Quantum provider implementations for backend management.

This module implements the provider pattern for abstracting quantum
computing platforms. Each provider encapsulates authentication, device
management, and execution logic for a specific platform.

.. module:: sudoku_nisq.providers
   :synopsis: Quantum backend provider implementations

.. moduleauthor:: Your Name <your.email@example.com>

Available Providers
-------------------

- :class:`IBMProvider` - IBM Quantum platform
- :class:`QuantinuumProvider` - Quantinuum H-series systems
- :class:`BraketProvider` - AWS Braket (planned)

Example Usage
-------------

.. code-block:: python

    from sudoku_nisq.backends import BackendManager
    
    manager = BackendManager()
    manager.init_ibm("ibm_brisbane", alias="ibm", api_token="...", instance="...")
    backend = manager.get("ibm")

See Also
--------
:class:`sudoku_nisq.backends.BackendManager` : Backend registry
:class:`QuantumProvider` : Base provider interface
"""
```

## 4. User Documentation (Markdown)

### Structure for Feature Documentation

```markdown
# Feature Name

Brief description (1-2 sentences)

## Overview

Longer explanation of what the feature does and why it's useful.

## Quick Start

Minimal example showing the most common use case:

\```python
from sudoku_nisq import Feature

# Create instance
feature = Feature()

# Use it
result = feature.do_something()
print(result)
\```

## Detailed Usage

### Basic Usage

Explain the simplest usage pattern with example.

### Advanced Usage

More complex scenarios with configuration options.

### Configuration Options

Table or list of available parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `option1` | str | "default" | What it does |
| `option2` | bool | True | What it controls |

## Use Cases

Concrete examples of when to use this feature:

**Scenario 1: [Use case name]**
\```python
# Code example for this scenario
\```

**Scenario 2: [Use case name]**
\```python
# Code example for this scenario
\```

## Best Practices

- ✅ **DO**: Recommendation with explanation
- ❌ **DON'T**: Anti-pattern with explanation

## Common Issues

**Issue**: Description of problem
**Solution**: How to fix it

## See Also

- [Related Feature](link.md)
- [API Reference](../api/module.html)
```

## 5. Implementation Documents

For complex features, create implementation documents in the root directory:

### File Naming Convention

- Use SCREAMING_SNAKE_CASE for visibility
- Be descriptive but concise
- Examples:
  - `FEATURE_NAME_IMPLEMENTATION.md`
  - `ARCHITECTURAL_DECISION.md`
  - `MIGRATION_GUIDE.md`

### Template for Implementation Docs

```markdown
# Feature Name Implementation

**Date**: YYYY-MM-DD
**Status**: ✅ Implemented | 🔄 In Progress | 📋 Planned
**Related Issues**: #123, #456

## Summary

One-paragraph overview of what was implemented and why.

## Motivation

### Problem
What problem does this solve?

### Goals
- Goal 1
- Goal 2

### Non-Goals
What is explicitly out of scope?

## Design

### Architecture

High-level design with diagrams if applicable.

### Components

List and describe major components:

#### Component 1
What it does and how it fits in.

#### Component 2
What it does and how it fits in.

## Implementation Details

### Files Created
- `path/to/file.py` - Description
- `path/to/test.py` - Description

### Files Modified
- `path/to/existing.py` - What changed and why

### Key Design Decisions

**Decision 1: [Topic]**
- **Choice**: What we chose
- **Alternatives considered**: What we didn't choose
- **Rationale**: Why we chose this way

## Usage Examples

### Basic Example
\```python
# Code
\```

### Advanced Example
\```python
# Code
\```

## Testing

How the feature is tested:
- Unit tests
- Integration tests
- Manual testing performed

## Performance Considerations

Any performance implications:
- Memory usage
- Computational overhead
- Scalability limits

## Future Work

What could be improved:
- [ ] Enhancement 1
- [ ] Enhancement 2

## References

- Links to related docs
- External resources
- Research papers (if applicable)
```

## 6. Changelog (CHANGELOG.md)

Maintain a changelog following [Keep a Changelog](https://keepachangelog.com/) format:

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Gate counting feature for circuit resource analysis
- Memory tracking for development profiling

### Changed
- Improved SDK abstraction architecture
- Enhanced provider pattern implementation

### Deprecated
- Old BackendManager class methods (use instance methods)

### Removed
- Legacy circuit caching format

### Fixed
- Memory leak in large circuit construction
- Race condition in parallel execution

### Security
- Updated dependencies to patch vulnerabilities

## [0.2.0] - 2024-12-09

### Added
- Provider pattern for quantum backend management
- Automatic SDK selection based on provider
- Canonical encoding framework for exact cover problems
- ZNE and PEC error mitigation integration

### Changed
- Refactored backend architecture for better extensibility
- Improved documentation structure

## [0.1.0] - 2024-11-01

Initial release
```

## 7. When to Update Each Documentation Type

| Change Type | Commit | PR | Code Docs | User Docs | Impl Docs | Changelog |
|-------------|--------|----|-----------|-----------|-----------|-----------| 
| Bug fix | ✅ | ✅ | Sometimes | Rarely | No | ✅ |
| New feature | ✅ | ✅ | ✅ | ✅ | Complex only | ✅ |
| Refactoring | ✅ | ✅ | ✅ | If API changes | Complex only | If breaking |
| Documentation | ✅ | ✅ | N/A | ✅ | Sometimes | If major |
| Tests | ✅ | ✅ | No | No | No | Rarely |
| Dependencies | ✅ | ✅ | No | If affects users | No | ✅ |

## 8. Documentation Review Checklist

Before submitting changes:

### Code Review
- [ ] All new functions/classes have docstrings
- [ ] Docstrings follow project style (reST for Python)
- [ ] Complex logic has inline comments explaining "why"
- [ ] Type hints are present and accurate

### User Documentation
- [ ] New features documented in user guides
- [ ] Examples provided for common use cases
- [ ] Breaking changes clearly explained
- [ ] Migration guides provided if needed

### Implementation Documentation
- [ ] Complex features have implementation docs
- [ ] Design decisions are documented
- [ ] Alternatives considered are noted

### Process Documentation
- [ ] Commit messages follow convention
- [ ] PR description is complete and clear
- [ ] CHANGELOG.md is updated
- [ ] Related issues are linked

### Verification
- [ ] Documentation builds without warnings
- [ ] All links work (internal and external)
- [ ] Code examples run successfully
- [ ] API references render correctly

## 9. Tools for Documentation

### Recommended Tools

**Sphinx** (Python projects):
- Auto-generates API docs from docstrings
- Supports multiple output formats (HTML, PDF)
- Extensive plugin ecosystem

**MkDocs** (Markdown-focused):
- Simple Markdown-based documentation
- Great for user guides and tutorials
- Built-in themes and search

**Docusaurus** (JavaScript/React):
- Modern documentation framework
- Versioned documentation support
- Interactive components

### Quality Tools

**Documentation Linters**:
- `pydocstyle` - Check docstring conventions
- `darglint` - Validate docstring arguments match function signatures
- `doc8` - Style checker for reStructuredText

**Link Checkers**:
- `sphinx-linkcheck` - Find broken links in Sphinx docs
- `markdown-link-check` - Validate Markdown links

**Coverage**:
- `interrogate` - Check docstring coverage
- `coverage.py` with `--include` for doc tests

## 10. Common Documentation Mistakes to Avoid

### ❌ Bad Practices

1. **No documentation** - "The code is self-documenting"
2. **Outdated documentation** - Doesn't match current implementation
3. **Vague descriptions** - "This function does stuff"
4. **Missing examples** - Theory without practical usage
5. **No "why"** - Only explains "what" and "how"
6. **Copy-paste errors** - Old function names in new docs
7. **Broken links** - References to moved/deleted files
8. **Mixed styles** - Inconsistent formatting across docs

### ✅ Good Practices

1. **Document as you code** - Don't leave it for later
2. **Start with examples** - Show usage before explaining theory
3. **Explain trade-offs** - Document why decisions were made
4. **Keep it current** - Update docs with code changes
5. **Test documentation** - Run code examples as tests
6. **Link related content** - Cross-reference relevant docs
7. **Use clear structure** - Consistent organization
8. **Review for clarity** - Have someone else read it

## Summary

Good documentation is:
- **Comprehensive**: Covers all levels from commit messages to user guides
- **Current**: Updated with every code change
- **Clear**: Written for the audience (users vs. developers)
- **Tested**: Examples run successfully
- **Discoverable**: Properly linked and indexed

Follow these practices to maintain professional-quality documentation that helps users, contributors, and your future self understand and maintain the codebase effectively.
