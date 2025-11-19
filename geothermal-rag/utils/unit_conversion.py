"""
Unit Conversion Utilities for Geothermal Well Data
Handles Imperial ↔ Metric conversions with validation
"""

from typing import Union

class UnitConverter:
    """Unit conversion utilities for geothermal well engineering"""
    
    # ============================================================================
    # LENGTH CONVERSIONS
    # ============================================================================
    
    @staticmethod
    def inches_to_meters(inches: Union[float, int]) -> float:
        """Convert inches to meters"""
        return inches * 0.0254
    
    @staticmethod
    def meters_to_inches(meters: Union[float, int]) -> float:
        """Convert meters to inches"""
        return meters / 0.0254
    
    @staticmethod
    def feet_to_meters(feet: Union[float, int]) -> float:
        """Convert feet to meters"""
        return feet * 0.3048
    
    @staticmethod
    def meters_to_feet(meters: Union[float, int]) -> float:
        """Convert meters to feet"""
        return meters / 0.3048
    
    @staticmethod
    def mm_to_meters(mm: Union[float, int]) -> float:
        """Convert millimeters to meters"""
        return mm / 1000.0
    
    @staticmethod
    def meters_to_mm(meters: Union[float, int]) -> float:
        """Convert meters to millimeters"""
        return meters * 1000.0
    
    # ============================================================================
    # PRESSURE CONVERSIONS
    # ============================================================================
    
    @staticmethod
    def bar_to_pa(bar: Union[float, int]) -> float:
        """Convert bar to Pascal"""
        return bar * 100000.0
    
    @staticmethod
    def pa_to_bar(pa: Union[float, int]) -> float:
        """Convert Pascal to bar"""
        return pa / 100000.0
    
    @staticmethod
    def psi_to_pa(psi: Union[float, int]) -> float:
        """Convert psi to Pascal"""
        return psi * 6894.76
    
    @staticmethod
    def pa_to_psi(pa: Union[float, int]) -> float:
        """Convert Pascal to psi"""
        return pa / 6894.76
    
    @staticmethod
    def bar_to_psi(bar: Union[float, int]) -> float:
        """Convert bar to psi"""
        return bar * 14.5038
    
    @staticmethod
    def psi_to_bar(psi: Union[float, int]) -> float:
        """Convert psi to bar"""
        return psi / 14.5038
    
    @staticmethod
    def kpa_to_pa(kpa: Union[float, int]) -> float:
        """Convert kiloPascal to Pascal"""
        return kpa * 1000.0
    
    # ============================================================================
    # TEMPERATURE CONVERSIONS
    # ============================================================================
    
    @staticmethod
    def fahrenheit_to_celsius(f: Union[float, int]) -> float:
        """Convert Fahrenheit to Celsius"""
        return (f - 32.0) * 5.0 / 9.0
    
    @staticmethod
    def celsius_to_fahrenheit(c: Union[float, int]) -> float:
        """Convert Celsius to Fahrenheit"""
        return c * 9.0 / 5.0 + 32.0
    
    @staticmethod
    def celsius_to_kelvin(c: Union[float, int]) -> float:
        """Convert Celsius to Kelvin"""
        return c + 273.15
    
    @staticmethod
    def kelvin_to_celsius(k: Union[float, int]) -> float:
        """Convert Kelvin to Celsius"""
        return k - 273.15
    
    # ============================================================================
    # VISCOSITY CONVERSIONS
    # ============================================================================
    
    @staticmethod
    def centipoise_to_pas(cp: Union[float, int]) -> float:
        """Convert centipoise to Pascal-second"""
        return cp / 1000.0
    
    @staticmethod
    def pas_to_centipoise(pas: Union[float, int]) -> float:
        """Convert Pascal-second to centipoise"""
        return pas * 1000.0
    
    # ============================================================================
    # FLOW RATE CONVERSIONS
    # ============================================================================
    
    @staticmethod
    def m3h_to_m3s(m3h: Union[float, int]) -> float:
        """Convert m³/h to m³/s"""
        return m3h / 3600.0
    
    @staticmethod
    def m3s_to_m3h(m3s: Union[float, int]) -> float:
        """Convert m³/s to m³/h"""
        return m3s * 3600.0
    
    @staticmethod
    def bpd_to_m3s(bpd: Union[float, int]) -> float:
        """Convert barrels per day to m³/s"""
        return bpd * 0.158987 / 86400.0
    
    @staticmethod
    def m3s_to_bpd(m3s: Union[float, int]) -> float:
        """Convert m³/s to barrels per day"""
        return m3s * 86400.0 / 0.158987
    
    # ============================================================================
    # FRACTIONAL INCH PARSING
    # ============================================================================
    
    @staticmethod
    def parse_fractional_inches(size_str: str) -> float:
        """
        Parse fractional inch notation to decimal inches
        
        Examples:
            "13 3/8"" -> 13.375
            "9 5/8"" -> 9.625
            "7"" -> 7.0
            "13.375" -> 13.375
        
        Args:
            size_str: String representation of size
            
        Returns:
            Decimal inches as float
        """
        # Remove quotes and extra whitespace
        size_str = size_str.strip().replace('"', '').strip()
        
        # Check if already decimal
        if '/' not in size_str:
            try:
                return float(size_str)
            except ValueError:
                raise ValueError(f"Cannot parse size: {size_str}")
        
        # Parse fractional notation
        parts = size_str.split()
        
        if len(parts) == 1:
            # Just a fraction like "3/8"
            frac_parts = parts[0].split('/')
            if len(frac_parts) == 2:
                numerator = float(frac_parts[0])
                denominator = float(frac_parts[1])
                return numerator / denominator
            else:
                raise ValueError(f"Invalid fraction format: {size_str}")
        
        elif len(parts) == 2:
            # Whole number and fraction like "13 3/8"
            whole = float(parts[0])
            frac_parts = parts[1].split('/')
            
            if len(frac_parts) == 2:
                numerator = float(frac_parts[0])
                denominator = float(frac_parts[1])
                return whole + (numerator / denominator)
            else:
                raise ValueError(f"Invalid fraction format: {size_str}")
        
        else:
            raise ValueError(f"Cannot parse size: {size_str}")
    
    @staticmethod
    def fractional_inches_to_meters(size_str: str) -> float:
        """
        Convert fractional inches string to meters
        
        Example:
            "13 3/8"" -> 0.33975 meters
        """
        inches = UnitConverter.parse_fractional_inches(size_str)
        return UnitConverter.inches_to_meters(inches)
    
    # ============================================================================
    # VALIDATION HELPERS
    # ============================================================================
    
    @staticmethod
    def validate_pipe_id_mm(id_mm: float, min_mm: float = 50, max_mm: float = 1000) -> bool:
        """
        Validate pipe internal diameter is within realistic range
        
        Args:
            id_mm: Internal diameter in millimeters
            min_mm: Minimum realistic ID
            max_mm: Maximum realistic ID
            
        Returns:
            True if valid, False otherwise
        """
        return min_mm <= id_mm <= max_mm
    
    @staticmethod
    def validate_md_tvd(md: float, tvd: float, tolerance: float = 1.0) -> bool:
        """
        Validate that MD >= TVD (measured depth >= true vertical depth)
        
        Args:
            md: Measured depth in meters
            tvd: True vertical depth in meters
            tolerance: Allowed tolerance for rounding errors
            
        Returns:
            True if valid, False otherwise
        """
        return md >= tvd - tolerance
    
    @staticmethod
    def validate_inclination(inclination: float) -> bool:
        """
        Validate inclination angle
        
        Args:
            inclination: Angle in degrees
            
        Returns:
            True if valid (0-90 degrees), False otherwise
        """
        return 0 <= inclination <= 90
    
    @staticmethod
    def detect_unit(value: float, unit_type: str) -> str:
        """
        Auto-detect likely unit based on value magnitude
        
        Args:
            value: Numeric value
            unit_type: Type of unit ('length', 'pressure', 'temperature')
            
        Returns:
            Likely unit as string
        """
        if unit_type == 'length':
            if value < 10:
                return 'meters'
            elif value < 100:
                return 'meters or inches'
            elif value < 1000:
                return 'meters'
            else:
                return 'millimeters'
        
        elif unit_type == 'pressure':
            if value < 100:
                return 'bar'
            elif value < 10000:
                return 'psi or kPa'
            else:
                return 'Pa'
        
        elif unit_type == 'temperature':
            if value < 100:
                return '°C'
            elif value < 250:
                return '°C or °F'
            else:
                return 'K'
        
        return 'unknown'


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def convert_casing_id_to_meters(id_value: Union[str, float], unit: str = 'inches') -> float:
    """
    Convert casing internal diameter to meters
    
    Args:
        id_value: ID value (can be fractional string or float)
        unit: 'inches', 'mm', or 'meters'
        
    Returns:
        ID in meters
    """
    if isinstance(id_value, str):
        # Assume fractional inches
        inches = UnitConverter.parse_fractional_inches(id_value)
        return UnitConverter.inches_to_meters(inches)
    
    if unit == 'inches':
        return UnitConverter.inches_to_meters(id_value)
    elif unit == 'mm':
        return UnitConverter.mm_to_meters(id_value)
    elif unit == 'meters':
        return float(id_value)
    else:
        raise ValueError(f"Unknown unit: {unit}")
