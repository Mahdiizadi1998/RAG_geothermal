"""Unit conversion utilities for geothermal well parameters."""

import re
from typing import Union, Optional


class UnitConverter:
    """Handle unit conversions for various geothermal well parameters."""
    
    # Conversion factors
    METERS_TO_FEET = 3.28084
    FEET_TO_METERS = 0.3048
    INCHES_TO_METERS = 0.0254
    METERS_TO_INCHES = 39.3701
    BAR_TO_PSI = 14.5038
    PSI_TO_BAR = 0.068948
    KPA_TO_BAR = 0.01
    MPA_TO_BAR = 10.0
    
    @staticmethod
    def parse_fractional_inch(inch_str: str) -> float:
        """
        Convert fractional inch notation to decimal.
        
        Examples:
            "13 3/8" -> 13.375
            "9 5/8" -> 9.625
            "13.375" -> 13.375
        
        Args:
            inch_str: String representation of inches (fractional or decimal)
            
        Returns:
            Decimal inch value
        """
        inch_str = inch_str.strip().replace('"', '').replace('inch', '').strip()
        
        # Check for fractional format: "13 3/8"
        match = re.match(r'(\d+)\s+(\d+)/(\d+)', inch_str)
        if match:
            whole = int(match.group(1))
            numerator = int(match.group(2))
            denominator = int(match.group(3))
            return whole + (numerator / denominator)
        
        # Check for decimal format
        try:
            return float(inch_str)
        except ValueError:
            raise ValueError(f"Unable to parse inch value: {inch_str}")
    
    @classmethod
    def inch_to_meter(cls, inches: Union[float, str]) -> float:
        """
        Convert inches to meters.
        
        Args:
            inches: Inch value (can be fractional string or float)
            
        Returns:
            Value in meters
        """
        if isinstance(inches, str):
            inches = cls.parse_fractional_inch(inches)
        return inches * cls.INCHES_TO_METERS
    
    @classmethod
    def meter_to_inch(cls, meters: float) -> float:
        """Convert meters to inches."""
        return meters * cls.METERS_TO_INCHES
    
    @classmethod
    def feet_to_meter(cls, feet: float) -> float:
        """Convert feet to meters."""
        return feet * cls.FEET_TO_METERS
    
    @classmethod
    def meter_to_feet(cls, meters: float) -> float:
        """Convert meters to feet."""
        return meters * cls.METERS_TO_FEET
    
    @classmethod
    def psi_to_bar(cls, psi: float) -> float:
        """Convert PSI to bar."""
        return psi * cls.PSI_TO_BAR
    
    @classmethod
    def bar_to_psi(cls, bar: float) -> float:
        """Convert bar to PSI."""
        return bar * cls.BAR_TO_PSI
    
    @classmethod
    def kpa_to_bar(cls, kpa: float) -> float:
        """Convert kPa to bar."""
        return kpa * cls.KPA_TO_BAR
    
    @classmethod
    def mpa_to_bar(cls, mpa: float) -> float:
        """Convert MPa to bar."""
        return mpa * cls.MPA_TO_BAR
    
    @classmethod
    def celsius_to_fahrenheit(cls, celsius: float) -> float:
        """Convert Celsius to Fahrenheit."""
        return celsius * 1.8 + 32
    
    @classmethod
    def fahrenheit_to_celsius(cls, fahrenheit: float) -> float:
        """Convert Fahrenheit to Celsius."""
        return (fahrenheit - 32) / 1.8
    
    @staticmethod
    def normalize_pressure(value: float, unit: str) -> float:
        """
        Normalize pressure to bar.
        
        Args:
            value: Pressure value
            unit: Unit string ('bar', 'psi', 'kPa', 'MPa')
            
        Returns:
            Pressure in bar
        """
        unit_lower = unit.lower().strip()
        
        if unit_lower in ['bar', 'bars']:
            return value
        elif unit_lower in ['psi', 'pound per square inch']:
            return UnitConverter.psi_to_bar(value)
        elif unit_lower in ['kpa', 'kilopascal']:
            return UnitConverter.kpa_to_bar(value)
        elif unit_lower in ['mpa', 'megapascal']:
            return UnitConverter.mpa_to_bar(value)
        else:
            raise ValueError(f"Unknown pressure unit: {unit}")
    
    @staticmethod
    def normalize_temperature(value: float, unit: str) -> float:
        """
        Normalize temperature to Celsius.
        
        Args:
            value: Temperature value
            unit: Unit string ('C', 'F', 'Celsius', 'Fahrenheit')
            
        Returns:
            Temperature in Celsius
        """
        unit_lower = unit.lower().strip()
        
        if unit_lower in ['c', 'celsius', '°c']:
            return value
        elif unit_lower in ['f', 'fahrenheit', '°f']:
            return UnitConverter.fahrenheit_to_celsius(value)
        else:
            raise ValueError(f"Unknown temperature unit: {unit}")
