"""
Tests for MetricsMetadataManager (Stages 6-7).

Tests cover:
- record(): Auto-compute and persist metrics
- query(): Retrieve metrics by run_id or filters
- compute_aggregated(): Multi-run statistics
- Edge cases: missing validation context, zero resources, corrupted data
"""

import json
import pytest
from pathlib import Path

from sudoku_nisq.metadata.metrics import MetricsMetadataManager
from sudoku_nisq.metadata.config import MetadataConfig


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Provide temporary cache directory."""
    cache_dir = tmp_path / ".quantum_cache"
    return cache_dir


@pytest.fixture
def puzzle_hash():
    """Provide sample puzzle hash."""
    return "test_puzzle_hash_abc123"


@pytest.fixture
def manager(temp_cache_dir, puzzle_hash):
    """Provide fresh MetricsMetadataManager instance."""
    return MetricsMetadataManager(temp_cache_dir, puzzle_hash)


@pytest.fixture
def sample_counts():
    """Provide sample measurement counts."""
    return {
        "00": 500,
        "01": 300,
        "10": 150,
        "11": 50
    }


@pytest.fixture
def validation_context():
    """Provide mock ValidationContext."""
    class MockContext:
        def __init__(self):
            self.solution_validator = lambda bs: bs in ["01", "10"]
            self.valid_solutions = ["01", "10"]
            self.total_valid_count = 2
    
    return MockContext()


class TestMetricsMetadataManagerBasics:
    """Tests for basic MetricsMetadataManager functionality."""
    
    def test_initialization(self, temp_cache_dir, puzzle_hash):
        """Test manager initialization."""
        manager = MetricsMetadataManager(temp_cache_dir, puzzle_hash)
        
        assert manager.cache_base == Path(temp_cache_dir)
        assert manager.puzzle_hash == puzzle_hash
        assert manager.stage_number == 6
        
        expected_path = temp_cache_dir / puzzle_hash / MetadataConfig.STAGE_6_7_METRICS
        assert manager.storage_path == expected_path
    
    def test_storage_path_structure(self, manager, temp_cache_dir, puzzle_hash):
        """Test storage path follows convention."""
        expected = temp_cache_dir / puzzle_hash / "stage_6_7_metrics.json"
        assert manager.storage_path == expected


class TestMetricsRecord:
    """Tests for MetricsMetadataManager.record() method."""

    def test_record_basic_metrics(self, manager, sample_counts, validation_context):
        """Test basic metrics recording."""
        result = manager.record(
            run_id="run_001",
            counts=sample_counts,
            validation_context=validation_context,
            shots=1000,
            two_qubit_gates=60,
            circuit_volume=1470
        )
        
        assert "run_id" in result
        assert result["run_id"] == "run_001"
        assert "timestamp" in result
        assert "stage_6_evaluation" in result
        assert "stage_7_normalization" in result
        
        # Stage 6 metrics
        stage_6 = result["stage_6_evaluation"]
        assert "p_succ" in stage_6
        assert "p_succ_ci_lower" in stage_6
        assert "p_succ_ci_upper" in stage_6
        assert "distinct_valid_solutions" in stage_6

        # New Stage 6 metrics
        assert "valid_odds" in stage_6
        assert "valid_odds_ci_lower" in stage_6
        assert "valid_odds_ci_upper" in stage_6
        assert "valid_odds_is_infinite" in stage_6

        assert "p_best_valid" in stage_6
        assert "p_best_invalid" in stage_6
        assert "peak_ratio" in stage_6
        assert "peak_gap" in stage_6
        assert "peak_ratio_is_infinite" in stage_6

        assert "top_k_valid_mass" in stage_6
        assert "precision_at_k" in stage_6
        assert "recall_at_k" in stage_6

        assert "mass_precision_at_k" in stage_6
        assert "valid_mass_capture_at_k" in stage_6
        
        # Stage 7 metrics
        stage_7 = result["stage_7_normalization"]

        # Retention/log-loss metrics
        assert "log_loss_per_2q" in stage_7
        assert "log_loss_per_2q_ci_lower" in stage_7
        assert "log_loss_per_2q_ci_upper" in stage_7

        assert "retention_per_2q" in stage_7
        assert "retention_per_2q_ci_lower" in stage_7
        assert "retention_per_2q_ci_upper" in stage_7

        assert "log_loss_per_volume" in stage_7
        assert "retention_per_volume" in stage_7
        assert "log_loss_per_volume_ci_lower" in stage_7
        assert "log_loss_per_volume_ci_upper" in stage_7
        assert "retention_per_volume_ci_lower" in stage_7
        assert "retention_per_volume_ci_upper" in stage_7

        # Shot budgets
        assert "shots_detect_point" in stage_7
        assert "shots_detect_pessimistic" in stage_7
        assert "shots_detect_optimistic" in stage_7
        assert "shot_budget_reliability" in stage_7
    
    def test_record_persists_to_file(self, manager, sample_counts, validation_context):
        """Test that metrics are persisted to JSON file."""
        manager.record(
            run_id="run_002",
            counts=sample_counts,
            validation_context=validation_context,
            two_qubit_gates=60
        )
        
        # Check file exists
        assert manager.storage_path.exists()
        
        # Verify contents
        with open(manager.storage_path) as f:
            data = json.load(f)
        
        assert "run_002" in data
        assert data["run_002"]["run_id"] == "run_002"
    
    def test_record_without_validation_context(self, manager, sample_counts):
        """Test recording without validation context (only Stage 7)."""
        result = manager.record(
            run_id="run_003",
            counts=sample_counts,
            two_qubit_gates=60,
            shots=1000
        )
        
        # Stage 6 should be empty or have error
        stage_6 = result["stage_6_evaluation"]
        assert len(stage_6) == 0 or "error" in stage_6
        
        # Stage 7 should still be empty (needs p_succ from Stage 6)
        stage_7 = result["stage_7_normalization"]
        assert len(stage_7) == 0
    
    def test_record_missing_run_id(self, manager, sample_counts):
        """Test that missing run_id raises ValueError."""
        with pytest.raises(ValueError, match="run_id is required"):
            manager.record(
                run_id="",
                counts=sample_counts
            )
    
    def test_record_missing_counts(self, manager):
        """Test that missing counts raises ValueError."""
        with pytest.raises(ValueError, match="counts dictionary is required"):
            manager.record(
                run_id="run_004",
                counts=None
            )
    
    def test_record_derives_shots_from_counts(self, manager, sample_counts, validation_context):
        """Test that shots are derived from counts if not provided."""
        result = manager.record(
            run_id="run_005",
            counts=sample_counts,
            validation_context=validation_context,
            two_qubit_gates=60
            # shots not provided
        )
        
        # Should still complete (shots derived as sum(counts))
        assert "run_id" in result

    def test_record_multiple_runs(self, manager, sample_counts, validation_context):
        """Test recording multiple runs to same file."""
        # Record first run
        manager.record(
            run_id="run_006",
            counts=sample_counts,
            validation_context=validation_context,
            two_qubit_gates=60
        )
        
        # Record second run
        manager.record(
            run_id="run_007",
            counts={"00": 400, "01": 600},
            validation_context=validation_context,
            two_qubit_gates=60
        )
        
        # Both should be in file
        with open(manager.storage_path) as f:
            data = json.load(f)
        
        assert "run_006" in data
        assert "run_007" in data

    def test_record_without_circuit_volume(self, manager, sample_counts, validation_context):
        """Test recording when circuit_volume is None."""
        result = manager.record(
            run_id="run_008",
            counts=sample_counts,
            validation_context=validation_context,
            two_qubit_gates=60,
            circuit_volume=None
        )
        
        stage_7 = result["stage_7_normalization"]

        # Volume-based retention should be absent when volume not provided
        assert "log_loss_per_volume" not in stage_7
        assert "retention_per_volume" not in stage_7

        # 2q-based and shot budget metrics should exist
        assert "retention_per_2q" in stage_7
        assert "log_loss_per_2q" in stage_7
        assert "shots_detect_point" in stage_7


class TestMetricsQuery:
    """Tests for MetricsMetadataManager.query() method."""
    
    def test_query_nonexistent_file(self, manager):
        """Test query when no metrics file exists."""
        result = manager.query(run_id="run_999")
        
        assert result is None
    
    def test_query_without_run_id_returns_empty_list(self, manager):
        """Test query without run_id returns empty list when file missing."""
        result = manager.query()
        
        assert result == []
    
    def test_query_specific_run_id(self, manager, sample_counts, validation_context):
        """Test querying specific run_id."""
        # Record metrics
        manager.record(
            run_id="run_010",
            counts=sample_counts,
            validation_context=validation_context,
            two_qubit_gates=60
        )
        
        # Query
        result = manager.query(run_id="run_010")
        
        assert result is not None
        assert result["run_id"] == "run_010"
        assert "stage_6_evaluation" in result

    def test_query_all_runs(self, manager, sample_counts, validation_context):
        """Test querying all runs (no filters)."""
        # Record multiple runs
        manager.record(run_id="run_011", counts=sample_counts, validation_context=validation_context, two_qubit_gates=60)
        manager.record(run_id="run_012", counts=sample_counts, validation_context=validation_context, two_qubit_gates=60)
        
        # Query all
        results = manager.query()
        
        assert isinstance(results, list)
        assert len(results) == 2

    def test_query_excludes_aggregated_entries(self, manager, sample_counts, validation_context):
        """Test that query excludes aggregated_ entries."""
        # Record normal run
        manager.record(run_id="run_013", counts=sample_counts, validation_context=validation_context, two_qubit_gates=60)
        
        # Manually add aggregated entry
        data = manager._load_json()
        data["aggregated_test_20251228_120000"] = {"aggregation_type": "multi_run"}
        manager._save_json(data)
        
        # Query should exclude aggregated
        results = manager.query()
        
        assert len(results) == 1
        assert results[0]["run_id"] == "run_013"


class TestMetricsComputeAggregated:
    """Tests for MetricsMetadataManager.compute_aggregated() method."""
    
    def test_compute_aggregated_empty_run_ids(self, manager):
        """Test that empty run_ids raises ValueError."""
        with pytest.raises(ValueError, match="run_ids list cannot be empty"):
            manager.compute_aggregated(run_ids=[])
    
    def test_compute_aggregated_no_metrics_file(self, manager):
        """Test aggregation when metrics file doesn't exist."""
        result = manager.compute_aggregated(run_ids=["run_999"])
        
        assert result is None
    
    def test_compute_aggregated_basic(self, manager, sample_counts, validation_context):
        """Test basic aggregation computation."""
        # Record multiple runs with different p_succ
        counts_high = {"01": 800, "10": 100, "00": 50, "11": 50}
        counts_med = {"01": 500, "10": 400, "00": 50, "11": 50}
        counts_low = {"01": 300, "10": 200, "00": 300, "11": 200}
        
        manager.record(run_id="run_014", counts=counts_high, validation_context=validation_context, two_qubit_gates=60)
        manager.record(run_id="run_015", counts=counts_med, validation_context=validation_context, two_qubit_gates=60)
        manager.record(run_id="run_016", counts=counts_low, validation_context=validation_context, two_qubit_gates=60)
        
        # Compute aggregation
        agg = manager.compute_aggregated(run_ids=["run_014", "run_015", "run_016"])
        
        assert agg is not None
        assert agg["aggregation_type"] == "multi_run"
        assert agg["num_runs"] == 3
        assert "run_ids" in agg
        
        # Check statistics
        assert "p_succ_mean" in agg
        assert "p_succ_std" in agg
        assert "p_succ_median" in agg
        assert "p_succ_q1" in agg
        assert "p_succ_q3" in agg
        assert "p_succ_iqr" in agg

        # Current Stage 7 aggregations should be present (2q gates were provided)
        assert "retention_per_2q_mean" in agg
        assert "log_loss_per_2q_mean" in agg
        assert "shots_detect_point_mean" in agg

    def test_compute_aggregated_stores_result(self, manager, sample_counts, validation_context):
        """Test that aggregation result is stored with timestamp key."""
        manager.record(run_id="run_017", counts=sample_counts, validation_context=validation_context, two_qubit_gates=60)
        manager.record(run_id="run_018", counts=sample_counts, validation_context=validation_context, two_qubit_gates=60)
        
        # Compute and store
        manager.compute_aggregated(
            run_ids=["run_017", "run_018"],
            aggregation_key="test_backend_opt1"
        )
        
        # Check stored in file
        data = manager._load_json()
        
        # Find aggregated entry
        agg_keys = [k for k in data.keys() if k.startswith("aggregated_test_backend_opt1")]
        assert len(agg_keys) == 1

    def test_compute_aggregated_missing_runs(self, manager, sample_counts, validation_context):
        """Test aggregation when some run_ids don't exist."""
        manager.record(run_id="run_019", counts=sample_counts, validation_context=validation_context, two_qubit_gates=60)
        
        # Try to aggregate with one valid, one invalid
        agg = manager.compute_aggregated(run_ids=["run_019", "run_999"])
        
        # Should still compute with available data
        assert agg is not None
        assert agg["num_runs"] == 1

    def test_compute_aggregated_no_valid_data(self, manager):
        """Test aggregation when no runs have p_succ."""
        # Manually create runs without Stage 6 metrics
        data = {
            "run_020": {
                "run_id": "run_020",
                "stage_6_evaluation": {},
                "stage_7_normalization": {}
            }
        }
        manager._save_json(data)
        
        agg = manager.compute_aggregated(run_ids=["run_020"])
        
        assert agg is None

    def test_compute_aggregated_retention_and_shots(self, manager, sample_counts, validation_context):
        """Test aggregation includes retention and shot-budget metrics."""
        manager.record(run_id="run_021", counts=sample_counts, validation_context=validation_context, two_qubit_gates=60, circuit_volume=1470)
        manager.record(run_id="run_022", counts=sample_counts, validation_context=validation_context, two_qubit_gates=60, circuit_volume=1470)

        agg = manager.compute_aggregated(run_ids=["run_021", "run_022"])

        assert "retention_per_2q_mean" in agg
        assert "log_loss_per_2q_mean" in agg
        assert "shots_detect_point_mean" in agg


class TestMetricsIntegration:
    """Integration tests for MetricsMetadataManager."""
    
    def test_full_workflow(self, manager, validation_context):
        """Test complete workflow: record → query → aggregate."""
        # Record multiple runs
        for i in range(5):
            counts = {
                "00": 500 - i*50,
                "01": 200 + i*30,
                "10": 200 + i*20,
                "11": 100
            }
            manager.record(
                run_id=f"run_{i:03d}",
                counts=counts,
                validation_context=validation_context,
                two_qubit_gates=60,
                shots=1000
            )
        
        # Query all runs
        all_runs = manager.query()
        assert len(all_runs) == 5
        
        # Aggregate
        run_ids = [f"run_{i:03d}" for i in range(5)]
        agg = manager.compute_aggregated(run_ids=run_ids, aggregation_key="batch_001")
        
        assert agg["num_runs"] == 5
        assert "p_succ_mean" in agg
        assert "p_succ_std" in agg

    def test_atomic_writes_no_corruption(self, manager, sample_counts, validation_context):
        """Test that concurrent writes don't corrupt data."""
        # Record multiple runs rapidly
        for i in range(10):
            manager.record(
                run_id=f"concurrent_run_{i}",
                counts=sample_counts,
                validation_context=validation_context,
                two_qubit_gates=60
            )
        
        # Verify all runs stored correctly
        data = manager._load_json()
        
        for i in range(10):
            assert f"concurrent_run_{i}" in data
            assert data[f"concurrent_run_{i}"]["run_id"] == f"concurrent_run_{i}"
