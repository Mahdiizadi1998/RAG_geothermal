"""
Validation Agent - Data Quality Checks and User Interaction
Validates extracted parameters against physical constraints
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.unit_conversion import UnitConverter
from typing import Dict, List, Tuple, Optional
import logging
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationAgent:
    """
    Validate extracted data and handle user interaction for missing/ambiguous values
    
    Validation hierarchy:
    1. Critical errors (block execution): MD < TVD, pipe ID out of range
    2. Warnings (proceed with caution): unusual values
    3. Missing data (prompt user): PVT properties, partial trajectory
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize validation agent
        
        Args:
            config_path: Path to config.yaml file
        """
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.validation_config = self.config['validation']
        self.converter = UnitConverter()
    
    def validate(self, extracted_data: Dict) -> Dict:
        """
        Validate extracted data
        
        Args:
            extracted_data: Dict from ParameterExtractionAgent
            
        Returns:
            Dict with:
            {
                'valid': bool,
                'critical_errors': List[str],
                'warnings': List[str],
                'missing_data': List[str],
                'suggestions': Dict  # Suggested default values
            }
        """
        critical_errors = []
        warnings = []
        missing_data = []
        suggestions = {}
        
        trajectory = extracted_data.get('trajectory', [])
        casing = extracted_data.get('casing_design', [])
        pvt = extracted_data.get('pvt_data', {})
        
        # ===================================================================
        # CRITICAL VALIDATIONS (Must pass)
        # ===================================================================
        
        # Check MD >= TVD for all trajectory points
        md_tvd_issues = self._validate_md_tvd(trajectory)
        if md_tvd_issues:
            critical_errors.extend(md_tvd_issues)
        
        # Check pipe ID ranges
        pipe_id_issues = self._validate_pipe_ids(trajectory)
        if pipe_id_issues:
            critical_errors.extend(pipe_id_issues)
        
        # Check inclination ranges
        inc_issues = self._validate_inclinations(trajectory)
        if inc_issues:
            critical_errors.extend(inc_issues)
        
        # Check well depth range
        depth_issues = self._validate_well_depth(trajectory)
        if depth_issues:
            warnings.extend(depth_issues)  # Warning, not critical
        
        # ===================================================================
        # WARNING VALIDATIONS (Unusual but possible)
        # ===================================================================
        
        # Check for unusual inclinations (>80°)
        if trajectory:
            max_inc = max(p['inclination'] for p in trajectory)
            if max_inc > 80:
                warnings.append(f"⚠️ High inclination detected: {max_inc:.1f}° (unusual for geothermal)")
        
        # Check PVT data ranges
        pvt_warnings = self._validate_pvt_data(pvt)
        warnings.extend(pvt_warnings)
        
        # ===================================================================
        # MISSING DATA CHECKS
        # ===================================================================
        
        if not trajectory:
            missing_data.append("No trajectory data extracted")
        elif len(trajectory) < 3:
            missing_data.append(f"Only {len(trajectory)} trajectory points (need ≥3 for analysis)")
        
        if not casing and not any('pipe_id' in p for p in trajectory):
            missing_data.append("No casing design or pipe ID data")
            suggestions['pipe_id'] = self.converter.inches_to_meters(6.276)  # 7" casing typical
        
        if 'density' not in pvt:
            missing_data.append("Fluid density not found")
            suggestions['density'] = 1000.0  # Water at standard conditions
        
        if 'viscosity' not in pvt:
            missing_data.append("Fluid viscosity not found")
            suggestions['viscosity'] = 0.001  # Water at 20°C
        
        if 'temp_gradient' not in pvt:
            missing_data.append("Temperature gradient not found")
            suggestions['temp_gradient'] = 30.0  # Typical geothermal gradient
        
        # ===================================================================
        # OVERALL VALIDITY
        # ===================================================================
        
        valid = len(critical_errors) == 0
        
        return {
            'valid': valid,
            'critical_errors': critical_errors,
            'warnings': warnings,
            'missing_data': missing_data,
            'suggestions': suggestions
        }
    
    def _validate_md_tvd(self, trajectory: List[Dict]) -> List[str]:
        """Validate MD >= TVD for all points"""
        errors = []
        tolerance = self.validation_config['md_tvd_tolerance']
        
        for i, point in enumerate(trajectory):
            md = point.get('md', 0)
            tvd = point.get('tvd', 0)
            
            if md < tvd - tolerance:
                errors.append(
                    f"❌ Point {i+1}: MD ({md:.1f}m) < TVD ({tvd:.1f}m) - physically impossible"
                )
        
        return errors
    
    def _validate_pipe_ids(self, trajectory: List[Dict]) -> List[str]:
        """Validate pipe IDs are within realistic range"""
        errors = []
        min_mm = self.validation_config['pipe_id_min_mm']
        max_mm = self.validation_config['pipe_id_max_mm']
        
        for i, point in enumerate(trajectory):
            pipe_id = point.get('pipe_id')
            if pipe_id is None:
                continue
            
            # Convert to mm for comparison
            pipe_id_mm = pipe_id * 1000
            
            if pipe_id_mm < min_mm or pipe_id_mm > max_mm:
                errors.append(
                    f"❌ Point {i+1}: Pipe ID ({pipe_id_mm:.1f}mm) out of range [{min_mm}-{max_mm}mm] - "
                    f"likely unit conversion error"
                )
        
        return errors
    
    def _validate_inclinations(self, trajectory: List[Dict]) -> List[str]:
        """Validate inclination angles"""
        errors = []
        max_inc = self.validation_config['inclination_max']
        
        for i, point in enumerate(trajectory):
            inc = point.get('inclination', 0)
            
            if inc < 0 or inc > max_inc:
                errors.append(
                    f"❌ Point {i+1}: Inclination ({inc:.1f}°) out of range [0-{max_inc}°]"
                )
        
        return errors
    
    def _validate_well_depth(self, trajectory: List[Dict]) -> List[str]:
        """Validate well depth is reasonable for geothermal"""
        warnings = []
        
        if not trajectory:
            return warnings
        
        max_md = max(p['md'] for p in trajectory)
        min_depth = self.validation_config['well_depth_min']
        max_depth = self.validation_config['well_depth_max']
        
        if max_md < min_depth:
            warnings.append(
                f"⚠️ Well depth ({max_md:.0f}m) is shallow for geothermal (typical: {min_depth}-{max_depth}m)"
            )
        elif max_md > max_depth:
            warnings.append(
                f"⚠️ Well depth ({max_md:.0f}m) is very deep (typical: {min_depth}-{max_depth}m)"
            )
        
        return warnings
    
    def _validate_pvt_data(self, pvt: Dict) -> List[str]:
        """Validate PVT data ranges"""
        warnings = []
        
        # Check fluid density (typical range: 800-1200 kg/m³)
        if 'density' in pvt:
            density = pvt['density']
            if density < 800 or density > 1200:
                warnings.append(
                    f"⚠️ Fluid density ({density:.0f} kg/m³) outside typical range [800-1200]"
                )
        
        # Check viscosity (typical range: 0.0001-0.01 Pa·s)
        if 'viscosity' in pvt:
            viscosity = pvt['viscosity']
            if viscosity < 0.0001 or viscosity > 0.01:
                warnings.append(
                    f"⚠️ Viscosity ({viscosity:.4f} Pa·s) outside typical range [0.0001-0.01]"
                )
        
        # Check temperature gradient
        if 'temp_gradient' in pvt:
            grad = pvt['temp_gradient']
            min_grad = self.validation_config['temperature_gradient_min']
            max_grad = self.validation_config['temperature_gradient_max']
            
            if grad < min_grad or grad > max_grad:
                warnings.append(
                    f"⚠️ Temperature gradient ({grad:.1f}°C/km) outside typical range [{min_grad}-{max_grad}]"
                )
        
        return warnings
    
    def apply_defaults(self, extracted_data: Dict, suggestions: Dict) -> Dict:
        """
        Apply default values for missing data
        
        Args:
            extracted_data: Original extracted data
            suggestions: Dict of suggested defaults from validate()
            
        Returns:
            Updated extracted_data with defaults applied
        """
        pvt = extracted_data.get('pvt_data', {})
        
        # Apply PVT defaults
        if 'density' in suggestions and 'density' not in pvt:
            pvt['density'] = suggestions['density']
            logger.info(f"Applied default density: {pvt['density']} kg/m³")
        
        if 'viscosity' in suggestions and 'viscosity' not in pvt:
            pvt['viscosity'] = suggestions['viscosity']
            logger.info(f"Applied default viscosity: {pvt['viscosity']} Pa·s")
        
        if 'temp_gradient' in suggestions and 'temp_gradient' not in pvt:
            pvt['temp_gradient'] = suggestions['temp_gradient']
            logger.info(f"Applied default temperature gradient: {pvt['temp_gradient']} °C/km")
        
        # Apply pipe ID default if needed
        if 'pipe_id' in suggestions:
            trajectory = extracted_data.get('trajectory', [])
            for point in trajectory:
                if 'pipe_id' not in point or point['pipe_id'] is None:
                    point['pipe_id'] = suggestions['pipe_id']
            logger.info(f"Applied default pipe ID: {suggestions['pipe_id']:.4f} m")
        
        extracted_data['pvt_data'] = pvt
        
        return extracted_data
    
    def format_validation_report(self, validation_result: Dict) -> str:
        """
        Format validation results as readable text
        
        Args:
            validation_result: Dict from validate()
            
        Returns:
            Formatted text report
        """
        report = []
        
        report.append("=" * 60)
        report.append("VALIDATION REPORT")
        report.append("=" * 60)
        
        # Overall status
        if validation_result['valid']:
            report.append("✓ Status: VALID - Data passed all critical checks")
        else:
            report.append("✗ Status: INVALID - Critical errors detected")
        
        report.append("")
        
        # Critical errors
        if validation_result['critical_errors']:
            report.append("CRITICAL ERRORS (Must fix):")
            for error in validation_result['critical_errors']:
                report.append(f"  {error}")
            report.append("")
        
        # Warnings
        if validation_result['warnings']:
            report.append("WARNINGS (Review recommended):")
            for warning in validation_result['warnings']:
                report.append(f"  {warning}")
            report.append("")
        
        # Missing data
        if validation_result['missing_data']:
            report.append("MISSING DATA:")
            for missing in validation_result['missing_data']:
                report.append(f"  • {missing}")
            report.append("")
        
        # Suggestions
        if validation_result['suggestions']:
            report.append("SUGGESTED DEFAULTS:")
            for key, value in validation_result['suggestions'].items():
                if isinstance(value, float):
                    report.append(f"  • {key}: {value:.4f}")
                else:
                    report.append(f"  • {key}: {value}")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)
