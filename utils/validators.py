"""Physics-based validation for geothermal well parameters."""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    confidence: float
    errors: List[str]
    warnings: List[str]


class PhysicsValidator:
    """Validate extracted parameters against physics constraints."""
    
    # Default ranges (can be overridden via config)
    MD_RANGE = (0, 10000)  # meters
    TVD_RANGE = (0, 10000)  # meters
    INCLINATION_RANGE = (0, 90)  # degrees
    DIAMETER_RANGE = (0.05, 1.0)  # meters
    PRESSURE_RANGE = (0, 1000)  # bar
    TEMPERATURE_RANGE = (0, 300)  # Celsius
    
    @staticmethod
    def validate_trajectory_point(
        md: float,
        tvd: float,
        inclination: Optional[float] = None
    ) -> ValidationResult:
        """
        Validate a single trajectory point.
        
        Args:
            md: Measured depth (meters)
            tvd: True vertical depth (meters)
            inclination: Inclination angle (degrees from vertical)
            
        Returns:
            ValidationResult with validation status
        """
        errors = []
        warnings = []
        confidence = 1.0
        
        # Check MD >= TVD (fundamental constraint)
        if md < tvd:
            errors.append(f"MD ({md}m) must be >= TVD ({tvd}m)")
            confidence = 0.0
        
        # Check if MD and TVD are within realistic ranges
        if not (PhysicsValidator.MD_RANGE[0] <= md <= PhysicsValidator.MD_RANGE[1]):
            errors.append(f"MD ({md}m) outside valid range {PhysicsValidator.MD_RANGE}")
            confidence *= 0.5
        
        if not (PhysicsValidator.TVD_RANGE[0] <= tvd <= PhysicsValidator.TVD_RANGE[1]):
            errors.append(f"TVD ({tvd}m) outside valid range {PhysicsValidator.TVD_RANGE}")
            confidence *= 0.5
        
        # Validate inclination if provided
        if inclination is not None:
            if not (PhysicsValidator.INCLINATION_RANGE[0] <= inclination <= PhysicsValidator.INCLINATION_RANGE[1]):
                errors.append(
                    f"Inclination ({inclination}°) outside valid range {PhysicsValidator.INCLINATION_RANGE}"
                )
                confidence *= 0.5
            
            # Check consistency: if vertical (0°), MD should equal TVD
            if abs(inclination) < 0.1 and abs(md - tvd) > 0.1:
                warnings.append(
                    f"Vertical well (inc={inclination}°) but MD ({md}m) != TVD ({tvd}m)"
                )
                confidence *= 0.9
        
        # Check if difference is suspiciously large
        if md - tvd > md * 0.5:  # MD is more than 50% longer than TVD
            warnings.append(
                f"Large difference between MD ({md}m) and TVD ({tvd}m). "
                "Check for highly deviated well."
            )
            confidence *= 0.95
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, confidence, errors, warnings)
    
    @staticmethod
    def validate_trajectory_sequence(
        trajectory_data: List[Dict[str, float]]
    ) -> ValidationResult:
        """
        Validate a sequence of trajectory points.
        
        Args:
            trajectory_data: List of dicts with 'md', 'tvd', 'inclination' keys
            
        Returns:
            ValidationResult with validation status
        """
        errors = []
        warnings = []
        confidence = 1.0
        
        if not trajectory_data:
            errors.append("Empty trajectory data")
            return ValidationResult(False, 0.0, errors, warnings)
        
        # Sort by MD
        sorted_data = sorted(trajectory_data, key=lambda x: x.get('md', 0))
        
        # Validate each point
        for i, point in enumerate(sorted_data):
            md = point.get('md')
            tvd = point.get('tvd')
            inc = point.get('inclination')
            
            if md is None or tvd is None:
                errors.append(f"Point {i}: Missing MD or TVD")
                confidence *= 0.8
                continue
            
            result = PhysicsValidator.validate_trajectory_point(md, tvd, inc)
            errors.extend(result.errors)
            warnings.extend(result.warnings)
            confidence = min(confidence, result.confidence)
        
        # Check monotonicity: MD and TVD should be increasing
        for i in range(1, len(sorted_data)):
            prev_md = sorted_data[i-1].get('md', 0)
            curr_md = sorted_data[i].get('md', 0)
            prev_tvd = sorted_data[i-1].get('tvd', 0)
            curr_tvd = sorted_data[i].get('tvd', 0)
            
            if curr_md <= prev_md:
                warnings.append(f"Non-increasing MD at index {i}: {prev_md} -> {curr_md}")
                confidence *= 0.95
            
            if curr_tvd < prev_tvd:
                errors.append(f"Decreasing TVD at index {i}: {prev_tvd} -> {curr_tvd}")
                confidence *= 0.7
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, confidence, errors, warnings)
    
    @staticmethod
    def validate_casing_diameter(diameter_m: float) -> ValidationResult:
        """
        Validate casing diameter.
        
        Args:
            diameter_m: Casing inner diameter in meters
            
        Returns:
            ValidationResult with validation status
        """
        errors = []
        warnings = []
        confidence = 1.0
        
        if not (PhysicsValidator.DIAMETER_RANGE[0] <= diameter_m <= PhysicsValidator.DIAMETER_RANGE[1]):
            errors.append(
                f"Diameter ({diameter_m}m) outside valid range {PhysicsValidator.DIAMETER_RANGE}"
            )
            confidence = 0.5
        
        # Common casing sizes (in inches converted to meters)
        common_sizes = [0.05, 0.114, 0.140, 0.178, 0.244, 0.340]  # 2", 4.5", 5.5", 7", 9.625", 13.375"
        
        # Check if close to a common size
        if not any(abs(diameter_m - size) < 0.01 for size in common_sizes):
            warnings.append(
                f"Diameter ({diameter_m:.3f}m) doesn't match common casing sizes"
            )
            confidence *= 0.95
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, confidence, errors, warnings)
    
    @staticmethod
    def validate_pressure(pressure_bar: float, context: str = "generic") -> ValidationResult:
        """
        Validate pressure value.
        
        Args:
            pressure_bar: Pressure in bar
            context: Context ('reservoir', 'wellhead', 'generic')
            
        Returns:
            ValidationResult with validation status
        """
        errors = []
        warnings = []
        confidence = 1.0
        
        if pressure_bar < 0:
            errors.append(f"Negative pressure: {pressure_bar} bar")
            confidence = 0.0
        
        if context == "reservoir" and pressure_bar > 1000:
            errors.append(f"Reservoir pressure ({pressure_bar} bar) unrealistically high")
            confidence *= 0.5
        elif context == "wellhead" and pressure_bar > 300:
            warnings.append(f"Wellhead pressure ({pressure_bar} bar) unusually high")
            confidence *= 0.8
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, confidence, errors, warnings)
    
    @staticmethod
    def validate_temperature(temp_celsius: float) -> ValidationResult:
        """
        Validate temperature value.
        
        Args:
            temp_celsius: Temperature in Celsius
            
        Returns:
            ValidationResult with validation status
        """
        errors = []
        warnings = []
        confidence = 1.0
        
        if not (PhysicsValidator.TEMPERATURE_RANGE[0] <= temp_celsius <= PhysicsValidator.TEMPERATURE_RANGE[1]):
            errors.append(
                f"Temperature ({temp_celsius}°C) outside valid range {PhysicsValidator.TEMPERATURE_RANGE}"
            )
            confidence *= 0.5
        
        # Typical geothermal range
        if temp_celsius < 50:
            warnings.append(f"Low temperature ({temp_celsius}°C) for geothermal application")
            confidence *= 0.9
        
        if temp_celsius > 250:
            warnings.append(f"Very high temperature ({temp_celsius}°C)")
            confidence *= 0.9
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, confidence, errors, warnings)
