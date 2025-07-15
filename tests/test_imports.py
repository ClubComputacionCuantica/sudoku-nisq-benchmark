"""Test that basic Python modules can be imported."""
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import pytest


def test_basic_imports():
    """Test that basic Python modules can be imported."""
    assert os is not None
    assert json is not None
    assert csv is not None
    assert tempfile is not None
    assert Path is not None


def test_typing_imports():
    """Test that typing modules work correctly."""
    assert Any is not None
    assert Mapping is not None


def test_project_imports():
    """Test that project modules can be imported."""
    try:
        from sudoku_nisq.q_sudoku import QSudoku
        from sudoku_nisq.metadata_manager import MetadataManager

        assert QSudoku is not None
        assert MetadataManager is not None
    except ImportError as e:
        pytest.fail(f"Failed to import project modules: {e}")


def test_additional_project_imports():
    """Test that additional project modules can be imported."""
    try:
        from sudoku_nisq import backtracking_solver
        from sudoku_nisq import graph_coloring_solver

        assert backtracking_solver is not None
        assert graph_coloring_solver is not None
    except ImportError as e:
        pytest.fail(f"Failed to import additional project modules: {e}")
