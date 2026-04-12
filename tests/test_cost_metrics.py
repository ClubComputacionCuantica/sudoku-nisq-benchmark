"""
Tests for cost-normalized efficiency metrics.

Tests cover:
- calculate_eta_product(): Product-based cost normalization
- calculate_eta_weighted_sum(): Weighted sum cost normalization
- calculate_decay_rate(): Exponential decay model
- fit_cost_weights(): Weight estimation from data
"""

import pytest
import math

class TestCalculateEtaProduct:
    """Tests for calculate_eta_product function."""
    
    def test_basic_eta_product(self):
        """Test basic product efficiency calculation."""
        from sudoku_nisq.metrics.calculators import calculate_eta_product
        
        p_succ = 0.8
        depth = 10
        two_qubit_gates = 5
        
        result = calculate_eta_product(p_succ, depth, two_qubit_gates)
        
        # 0.8 / (10 * 5) = 0.8 / 50 = 0.016
        assert result == pytest.approx(0.016)
    
    def test_zero_cost_returns_none(self):
        """Test that zero cost returns None."""
        from sudoku_nisq.metrics.calculators import calculate_eta_product
        
        assert calculate_eta_product(0.8, depth=0, two_qubit_gates=5) is None
        assert calculate_eta_product(0.8, depth=10, two_qubit_gates=0) is None
    
    def test_invalid_probability(self):
        """Test invalid probability handling."""
        from sudoku_nisq.metrics.calculators import calculate_eta_product
        
        assert calculate_eta_product(1.1, 10, 5) is None
        assert calculate_eta_product(-0.1, 10, 5) is None


class TestCalculateEtaWeightedSum:
    """Tests for calculate_eta_weighted_sum function."""
    
    def test_default_weights(self):
        """Test with default weights (alpha=1, beta=1)."""
        from sudoku_nisq.metrics.calculators import calculate_eta_weighted_sum
        
        # 0.8 / (10 + 5) = 0.8 / 15
        result = calculate_eta_weighted_sum(0.8, depth=10, two_qubit_gates=5)
        assert result == pytest.approx(0.0533333)
    
    def test_custom_weights(self):
        """Test with custom weights."""
        from sudoku_nisq.metrics.calculators import calculate_eta_weighted_sum
        
        # alpha=10, beta=1
        # 0.8 / (10*10 + 5*1) = 0.8 / 105
        result = calculate_eta_weighted_sum(0.8, 10, 5, alpha=10, beta=1)
        assert result == pytest.approx(0.8 / 105)
    
    def test_zero_weights_behavior(self):
        """Test behavior when weights zero out a component."""
        from sudoku_nisq.metrics.calculators import calculate_eta_weighted_sum
        
        # Ignore depth: 0.8 / (0*10 + 1*5) = 0.16
        result = calculate_eta_weighted_sum(0.8, 10, 5, alpha=0, beta=1)
        assert result == pytest.approx(0.16)


class TestCalculateDecayRate:
    """Tests for calculate_decay_rate function."""
    
    def test_basic_decay_rate(self):
        """Test basic decay rate calculation."""
        from sudoku_nisq.metrics.calculators import calculate_decay_rate
        
        # -log(0.8) / (10 + 5) ≈ 0.22314 / 15
        result = calculate_decay_rate(0.8, depth=10, two_qubit_gates=5)
        expected = -math.log(0.8) / 15
        assert result == pytest.approx(expected)
    
    def test_perfect_success(self):
        """Test with p_succ=1.0 (should be 0 decay)."""
        from sudoku_nisq.metrics.calculators import calculate_decay_rate
        
        result = calculate_decay_rate(1.0, 10, 5)
        assert result == 0.0
    
    def test_zero_success_clamping(self):
        """Test that zero success is clamped to epsilon."""
        from sudoku_nisq.metrics.calculators import calculate_decay_rate
        
        epsilon = 1e-10
        result = calculate_decay_rate(0.0, 10, 5, epsilon=epsilon)
        expected = -math.log(epsilon) / 15
        assert result == pytest.approx(expected)


class TestFitCostWeights:
    """Tests for fit_cost_weights function."""
    
    def test_fit_synthetic_data(self):
        """Test fitting on perfect synthetic data."""
        from sudoku_nisq.metrics.calculators import fit_cost_weights
        
        # Model: -log(p) = 0.02*depth + 0.02*gates
        alpha_true = 0.02
        beta_true = 0.02
        
        data = []
        # Use uncorrelated data points to avoid singular matrix
        # (depth, gates) pairs: (10, 5), (10, 10), (20, 5)
        for d, g in [(10, 5), (10, 10), (20, 5)]:
            cost = alpha_true * d + beta_true * g
            p = math.exp(-cost)
            data.append((p, d, g))
            
        alpha, beta, r2 = fit_cost_weights(data)
        
        assert alpha == pytest.approx(alpha_true, abs=1e-5)
        assert beta == pytest.approx(beta_true, abs=1e-5)
        assert r2 == pytest.approx(1.0, abs=1e-5)
    
    def test_insufficient_data(self):
        """Test error when not enough data points."""
        from sudoku_nisq.metrics.calculators import fit_cost_weights
        
        with pytest.raises(ValueError, match="Need at least 2 data points"):
            fit_cost_weights([(0.8, 10, 5)])
    
    def test_degenerate_depths(self):
        """Test error when all depths are identical."""
        from sudoku_nisq.metrics.calculators import fit_cost_weights
        
        data = [
            (0.8, 10, 5),
            (0.7, 10, 10),
            (0.6, 10, 15)
        ]
        with pytest.raises(ValueError, match="All depths are identical"):
            fit_cost_weights(data)
