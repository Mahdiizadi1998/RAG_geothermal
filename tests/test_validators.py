"""Tests for physics validators."""

import pytest
from utils.validators import PhysicsValidator


class TestPhysicsValidator:
    """Test suite for PhysicsValidator class."""
    
    def test_validate_trajectory_point_valid(self):
        """Test validation of valid trajectory point."""
        result = PhysicsValidator.validate_trajectory_point(
            md=1000.0,
            tvd=995.0,
            inclination=5.0
        )
        
        assert result.is_valid
        assert result.confidence > 0.9
        assert len(result.errors) == 0
    
    def test_validate_trajectory_point_invalid_md_tvd(self):
        """Test validation when MD < TVD (invalid)."""
        result = PhysicsValidator.validate_trajectory_point(
            md=995.0,
            tvd=1000.0,
            inclination=5.0
        )
        
        assert not result.is_valid
        assert result.confidence == 0.0
        assert any("MD" in error and "TVD" in error for error in result.errors)
    
    def test_validate_trajectory_point_out_of_range(self):
        """Test validation with out-of-range values."""
        result = PhysicsValidator.validate_trajectory_point(
            md=15000.0,  # Too deep
            tvd=14000.0,
            inclination=5.0
        )
        
        assert not result.is_valid
        assert len(result.errors) > 0
    
    def test_validate_trajectory_point_invalid_inclination(self):
        """Test validation with invalid inclination."""
        result = PhysicsValidator.validate_trajectory_point(
            md=1000.0,
            tvd=995.0,
            inclination=95.0  # > 90 degrees
        )
        
        assert not result.is_valid
        assert any("Inclination" in error for error in result.errors)
    
    def test_validate_trajectory_sequence_valid(self):
        """Test validation of valid trajectory sequence."""
        trajectory = [
            {'md': 0, 'tvd': 0, 'inclination': 0},
            {'md': 500, 'tvd': 499, 'inclination': 2},
            {'md': 1000, 'tvd': 995, 'inclination': 5},
        ]
        
        result = PhysicsValidator.validate_trajectory_sequence(trajectory)
        
        assert result.is_valid
        assert result.confidence > 0.9
    
    def test_validate_trajectory_sequence_decreasing_tvd(self):
        """Test validation with decreasing TVD (invalid)."""
        trajectory = [
            {'md': 0, 'tvd': 0, 'inclination': 0},
            {'md': 500, 'tvd': 499, 'inclination': 2},
            {'md': 1000, 'tvd': 490, 'inclination': 5},  # TVD decreased
        ]
        
        result = PhysicsValidator.validate_trajectory_sequence(trajectory)
        
        assert not result.is_valid
        assert any("Decreasing TVD" in error for error in result.errors)
    
    def test_validate_casing_diameter_valid(self):
        """Test validation of valid casing diameter."""
        # 13 3/8" = 0.33973 m
        result = PhysicsValidator.validate_casing_diameter(0.33973)
        
        assert result.is_valid
        assert result.confidence > 0.9
    
    def test_validate_casing_diameter_invalid(self):
        """Test validation of invalid casing diameter."""
        result = PhysicsValidator.validate_casing_diameter(2.0)  # Too large
        
        assert not result.is_valid
        assert len(result.errors) > 0
    
    def test_validate_pressure_valid(self):
        """Test validation of valid pressure."""
        result = PhysicsValidator.validate_pressure(250.0, context="reservoir")
        
        assert result.is_valid
        assert result.confidence > 0.9
    
    def test_validate_pressure_negative(self):
        """Test validation of negative pressure (invalid)."""
        result = PhysicsValidator.validate_pressure(-10.0)
        
        assert not result.is_valid
        assert result.confidence == 0.0
    
    def test_validate_temperature_valid(self):
        """Test validation of valid temperature."""
        result = PhysicsValidator.validate_temperature(150.0)
        
        assert result.is_valid
        assert result.confidence > 0.9
    
    def test_validate_temperature_out_of_range(self):
        """Test validation of out-of-range temperature."""
        result = PhysicsValidator.validate_temperature(350.0)  # Too high
        
        assert not result.is_valid
        assert len(result.errors) > 0
