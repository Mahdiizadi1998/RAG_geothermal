"""Tests for unit conversion utilities."""

import pytest
from utils.unit_converter import UnitConverter


class TestUnitConverter:
    """Test suite for UnitConverter class."""
    
    def test_parse_fractional_inch(self):
        """Test parsing fractional inch notation."""
        assert UnitConverter.parse_fractional_inch("13 3/8") == 13.375
        assert UnitConverter.parse_fractional_inch("9 5/8") == 9.625
        assert UnitConverter.parse_fractional_inch("7") == 7.0
        assert UnitConverter.parse_fractional_inch("13.375") == 13.375
    
    def test_inch_to_meter(self):
        """Test inch to meter conversion."""
        result = UnitConverter.inch_to_meter(13.375)
        assert abs(result - 0.33973) < 0.001
        
        # Test with fractional string
        result = UnitConverter.inch_to_meter("13 3/8")
        assert abs(result - 0.33973) < 0.001
    
    def test_meter_to_inch(self):
        """Test meter to inch conversion."""
        result = UnitConverter.meter_to_inch(0.33973)
        assert abs(result - 13.375) < 0.01
    
    def test_pressure_conversions(self):
        """Test pressure conversions."""
        # bar to psi
        result = UnitConverter.bar_to_psi(10.0)
        assert abs(result - 145.038) < 0.1
        
        # psi to bar
        result = UnitConverter.psi_to_bar(145.038)
        assert abs(result - 10.0) < 0.1
    
    def test_temperature_conversions(self):
        """Test temperature conversions."""
        # Celsius to Fahrenheit
        assert UnitConverter.celsius_to_fahrenheit(0) == 32
        assert UnitConverter.celsius_to_fahrenheit(100) == 212
        
        # Fahrenheit to Celsius
        assert UnitConverter.fahrenheit_to_celsius(32) == 0
        assert UnitConverter.fahrenheit_to_celsius(212) == 100
    
    def test_normalize_pressure(self):
        """Test pressure normalization to bar."""
        assert UnitConverter.normalize_pressure(10.0, "bar") == 10.0
        assert abs(UnitConverter.normalize_pressure(145.038, "psi") - 10.0) < 0.1
        assert UnitConverter.normalize_pressure(1000, "kPa") == 10.0
        assert UnitConverter.normalize_pressure(1, "MPa") == 10.0
    
    def test_normalize_temperature(self):
        """Test temperature normalization to Celsius."""
        assert UnitConverter.normalize_temperature(100, "C") == 100
        assert UnitConverter.normalize_temperature(100, "Celsius") == 100
        assert UnitConverter.normalize_temperature(212, "F") == 100
        assert UnitConverter.normalize_temperature(212, "Fahrenheit") == 100
