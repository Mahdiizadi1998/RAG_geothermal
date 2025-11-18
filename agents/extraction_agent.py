"""Parameter extraction agent for geothermal well data."""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from utils.unit_converter import UnitConverter
from utils.validators import PhysicsValidator, ValidationResult


@dataclass
class TrajectoryPoint:
    """Represents a single trajectory survey point."""
    md: float  # Measured depth (meters)
    tvd: float  # True vertical depth (meters)
    inclination: Optional[float] = None  # Inclination (degrees)
    azimuth: Optional[float] = None  # Azimuth (degrees)
    confidence: float = 1.0


@dataclass
class CasingSection:
    """Represents a casing section."""
    top_depth: float  # meters
    bottom_depth: float  # meters
    diameter: float  # inner diameter in meters
    grade: Optional[str] = None
    weight: Optional[float] = None
    confidence: float = 1.0


@dataclass
class WellParameters:
    """Complete well parameters extracted from document."""
    well_name: Optional[str] = None
    trajectory: List[TrajectoryPoint] = field(default_factory=list)
    casing: List[CasingSection] = field(default_factory=list)
    reservoir_pressure: Optional[float] = None  # bar
    reservoir_temperature: Optional[float] = None  # Celsius
    fluid_properties: Dict = field(default_factory=dict)
    validation_results: List[ValidationResult] = field(default_factory=list)


class ParameterExtractionAgent:
    """
    Agent responsible for extracting structured parameters from text chunks.
    
    Extracts:
    1. Trajectory data (MD, TVD, Inclination)
    2. Casing design (depths and diameters)
    3. Pressure and temperature data
    4. Fluid properties
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize extraction agent.
        
        Args:
            config: Configuration dict with extraction patterns
        """
        self.config = config or {}
        self.patterns = self.config.get('patterns', {})
        self.unit_converter = UnitConverter()
        self.validator = PhysicsValidator()
    
    def extract_trajectory_from_text(self, text: str) -> List[TrajectoryPoint]:
        """
        Extract trajectory points from text.
        
        Args:
            text: Text containing trajectory data
            
        Returns:
            List of trajectory points
        """
        trajectory_points = []
        
        # Pattern for trajectory table rows
        # Example: "1000.0    995.2    2.5    180.0"
        # or: "MD: 1000.0 m, TVD: 995.2 m, Inc: 2.5°"
        
        # Try table format first
        table_pattern = r'(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s*(?:(\d+\.?\d*))?'
        matches = re.findall(table_pattern, text)
        
        for match in matches:
            try:
                md = float(match[0])
                tvd = float(match[1])
                inc = float(match[2]) if match[2] else None
                azimuth = float(match[3]) if match[3] else None
                
                # Validate that this looks like trajectory data
                if md >= tvd and 0 <= md <= 10000:
                    trajectory_points.append(TrajectoryPoint(
                        md=md,
                        tvd=tvd,
                        inclination=inc,
                        azimuth=azimuth,
                        confidence=0.8
                    ))
            except (ValueError, IndexError):
                continue
        
        # If no points found, try named format
        if not trajectory_points:
            md_pattern = r'MD[:\s]*(\d+\.?\d*)\s*m'
            tvd_pattern = r'TVD[:\s]*(\d+\.?\d*)\s*m'
            inc_pattern = r'Inc[:\s]*(\d+\.?\d*)\s*[°d]'
            
            md_matches = re.findall(md_pattern, text, re.IGNORECASE)
            tvd_matches = re.findall(tvd_pattern, text, re.IGNORECASE)
            inc_matches = re.findall(inc_pattern, text, re.IGNORECASE)
            
            # Zip together if we have matching counts
            if len(md_matches) == len(tvd_matches):
                for i in range(len(md_matches)):
                    md = float(md_matches[i])
                    tvd = float(tvd_matches[i])
                    inc = float(inc_matches[i]) if i < len(inc_matches) else None
                    
                    trajectory_points.append(TrajectoryPoint(
                        md=md,
                        tvd=tvd,
                        inclination=inc,
                        confidence=0.7
                    ))
        
        return trajectory_points
    
    def extract_casing_from_text(self, text: str) -> List[CasingSection]:
        """
        Extract casing sections from text.
        
        Args:
            text: Text containing casing data
            
        Returns:
            List of casing sections
        """
        casing_sections = []
        
        # Pattern for casing descriptions
        # Example: "13 3/8\" casing to 1000m"
        # or: "9.625 inch @ 1500-2500m"
        
        # Fractional inch format
        pattern1 = r'(\d+)\s+(\d+)/(\d+)\s*(?:"|inch|in)?\s*(?:casing|pipe)?\s*(?:to|@|at)\s*(\d+\.?\d*)\s*(?:-\s*(\d+\.?\d*))?\s*m'
        matches1 = re.findall(pattern1, text, re.IGNORECASE)
        
        for match in matches1:
            try:
                # Parse fractional inch
                whole = int(match[0])
                numerator = int(match[1])
                denominator = int(match[2])
                diameter_inch = whole + (numerator / denominator)
                diameter_m = self.unit_converter.inch_to_meter(diameter_inch)
                
                bottom_depth = float(match[3])
                top_depth = float(match[4]) if match[4] else 0.0
                
                casing_sections.append(CasingSection(
                    top_depth=top_depth,
                    bottom_depth=bottom_depth,
                    diameter=diameter_m,
                    confidence=0.8
                ))
            except (ValueError, IndexError):
                continue
        
        # Decimal inch format
        pattern2 = r'(\d+\.?\d*)\s*(?:"|inch|in)\s*(?:casing|pipe)?\s*(?:to|@|at)\s*(\d+\.?\d*)\s*(?:-\s*(\d+\.?\d*))?\s*m'
        matches2 = re.findall(pattern2, text, re.IGNORECASE)
        
        for match in matches2:
            try:
                diameter_inch = float(match[0])
                diameter_m = self.unit_converter.inch_to_meter(diameter_inch)
                
                bottom_depth = float(match[1])
                top_depth = float(match[2]) if match[2] else 0.0
                
                # Avoid duplicates
                is_duplicate = any(
                    abs(c.diameter - diameter_m) < 0.01 and abs(c.bottom_depth - bottom_depth) < 1.0
                    for c in casing_sections
                )
                
                if not is_duplicate:
                    casing_sections.append(CasingSection(
                        top_depth=top_depth,
                        bottom_depth=bottom_depth,
                        diameter=diameter_m,
                        confidence=0.8
                    ))
            except (ValueError, IndexError):
                continue
        
        return casing_sections
    
    def extract_pressure(self, text: str) -> Optional[Tuple[float, str]]:
        """
        Extract pressure values from text.
        
        Args:
            text: Text containing pressure data
            
        Returns:
            Tuple of (pressure in bar, context) or None
        """
        # Pattern for pressure
        # Example: "Reservoir pressure: 250 bar"
        pattern = r'(?:reservoir|wellhead|static|bottom.?hole)?\s*pressure[:\s]*(\d+\.?\d*)\s*(bar|psi|kPa|MPa)'
        
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            
            # Normalize to bar
            pressure_bar = self.unit_converter.normalize_pressure(value, unit)
            
            # Determine context
            context = "generic"
            if "reservoir" in text.lower():
                context = "reservoir"
            elif "wellhead" in text.lower():
                context = "wellhead"
            
            return (pressure_bar, context)
        
        return None
    
    def extract_temperature(self, text: str) -> Optional[float]:
        """
        Extract temperature values from text.
        
        Args:
            text: Text containing temperature data
            
        Returns:
            Temperature in Celsius or None
        """
        # Pattern for temperature
        # Example: "Reservoir temperature: 150°C"
        pattern = r'temperature[:\s]*(\d+\.?\d*)\s*°?\s*([CF])'
        
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            
            # Normalize to Celsius
            temp_celsius = self.unit_converter.normalize_temperature(value, unit)
            
            return temp_celsius
        
        return None
    
    def merge_trajectory_and_casing(
        self, 
        trajectory: List[TrajectoryPoint],
        casing: List[CasingSection]
    ) -> List[Dict]:
        """
        Merge trajectory and casing data.
        
        Args:
            trajectory: List of trajectory points
            casing: List of casing sections
            
        Returns:
            Combined data with MD, TVD, Inc, and Pipe_ID
        """
        merged_data = []
        
        # Sort both lists by depth
        trajectory_sorted = sorted(trajectory, key=lambda x: x.md)
        casing_sorted = sorted(casing, key=lambda x: x.top_depth)
        
        for traj_point in trajectory_sorted:
            # Find applicable casing for this depth
            pipe_id = None
            for casing_section in casing_sorted:
                if casing_section.top_depth <= traj_point.md <= casing_section.bottom_depth:
                    pipe_id = casing_section.diameter
                    break
            
            merged_data.append({
                'md': traj_point.md,
                'tvd': traj_point.tvd,
                'inclination': traj_point.inclination,
                'pipe_id': pipe_id,
                'azimuth': traj_point.azimuth
            })
        
        return merged_data
    
    def extract_from_chunks(self, chunks: List) -> WellParameters:
        """
        Extract parameters from all chunks.
        
        Args:
            chunks: List of Chunk objects
            
        Returns:
            WellParameters with extracted data
        """
        params = WellParameters()
        
        for chunk in chunks:
            # Extract trajectory
            traj_points = self.extract_trajectory_from_text(chunk.text)
            params.trajectory.extend(traj_points)
            
            # Extract casing
            casing_sections = self.extract_casing_from_text(chunk.text)
            params.casing.extend(casing_sections)
            
            # Extract pressure
            pressure_data = self.extract_pressure(chunk.text)
            if pressure_data and params.reservoir_pressure is None:
                params.reservoir_pressure, context = pressure_data
            
            # Extract temperature
            temperature = self.extract_temperature(chunk.text)
            if temperature and params.reservoir_temperature is None:
                params.reservoir_temperature = temperature
        
        # Validate extracted data
        if params.trajectory:
            validation = self.validator.validate_trajectory_sequence([
                {'md': p.md, 'tvd': p.tvd, 'inclination': p.inclination}
                for p in params.trajectory
            ])
            params.validation_results.append(validation)
        
        return params
