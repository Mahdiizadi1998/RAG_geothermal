"""
Regex Pattern Library for Geothermal Well Data Extraction
Contains patterns for trajectory surveys, casing design, and equipment specs
"""

import re
from typing import Dict, List, Optional, Tuple

class PatternLibrary:
    """Collection of regex patterns for extracting structured data from geothermal well reports"""
    
    # ============================================================================
    # TRAJECTORY PATTERNS
    # ============================================================================
    
    # Pattern: Space-separated MD, TVD, Inclination
    # Example: "[md]  [tvd]  [inc]"
    TRAJECTORY_SPACE_SEPARATED = re.compile(
        r'(\d{1,5}\.?\d*)\s+(\d{1,5}\.?\d*)\s+(\d{1,2}\.?\d*)',
        re.MULTILINE
    )
    
    # Pattern: Pipe-separated trajectory table
    # Example: "| [md] | [tvd] | [inc] |"
    TRAJECTORY_PIPE_SEPARATED = re.compile(
        r'\|\s*(\d{1,5}\.?\d*)\s*\|\s*(\d{1,5}\.?\d*)\s*\|\s*(\d{1,2}\.?\d*)\s*\|',
        re.MULTILINE
    )
    
    # Pattern: Tab-separated trajectory
    TRAJECTORY_TAB_SEPARATED = re.compile(
        r'(\d{1,5}\.?\d*)\t+(\d{1,5}\.?\d*)\t+(\d{1,2}\.?\d*)',
        re.MULTILINE
    )
    
    # Pattern: Detect trajectory table headers
    TRAJECTORY_HEADER = re.compile(
        r'(MD|Measured\s+Depth|Along\s+Hole|AH).*?(TVD|True\s+Vertical\s+Depth).*?(Inc|Inclination|Angle)',
        re.IGNORECASE | re.MULTILINE
    )
    
    # ============================================================================
    # CASING PATTERNS
    # ============================================================================
    
    # Pattern: Fractional inches with casing/liner
    # Example: "13 3/8" casing from 0 to 1331 m, ID 12.615""
    CASING_FRACTIONAL = re.compile(
        r'(\d+)\s+(\d+)/(\d+)"?\s+(?:casing|liner|tubing)[^\n]{0,150}?'
        r'(?:from|top)?\s*(\d{1,5}\.?\d*)\s*(?:to|bottom|-)\s*(\d{1,5}\.?\d*)\s*'
        r'(?:m|meters)?[^\n]{0,100}?'
        r'(?:ID|id|I\.D\.|internal\s+diameter)[^\d]{0,10}(\d+\.?\d*)',
        re.IGNORECASE | re.MULTILINE
    )
    
    # Pattern: Decimal inches with casing
    # Example: "13.375" casing 0-1331m ID 12.615""
    CASING_DECIMAL = re.compile(
        r'(\d+\.\d+)"?\s+(?:casing|liner|tubing)[^\n]{0,150}?'
        r'(?:from|top)?\s*(\d{1,5}\.?\d*)\s*(?:to|bottom|-)\s*(\d{1,5}\.?\d*)\s*'
        r'(?:m|meters)?[^\n]{0,100}?'
        r'(?:ID|id|I\.D\.|internal\s+diameter)[^\d]{0,10}(\d+\.?\d*)',
        re.IGNORECASE | re.MULTILINE
    )
    
    # Pattern: Simplified casing with just size and depth
    CASING_SIMPLE = re.compile(
        r'(\d+\s+\d+/\d+|\d+\.\d+)"?\s+(?:casing|liner)[^\n]{0,100}?(\d{3,5})\s*(?:m|meters)',
        re.IGNORECASE
    )
    
    # ============================================================================
    # WELL NAME PATTERNS
    # ============================================================================
    
    # Pattern: Dutch geothermal well names
    # Example: "[CODE]-GT-##", "[CODE]-GT-##-S#", "[CODE2]-GT-##"
    WELL_NAME = re.compile(
        r'\b([A-Z]{2,10}-GT-\d{2}(?:-S\d+)?)\b'
    )
    
    # ============================================================================
    # PVT DATA PATTERNS
    # ============================================================================
    
    # Pattern: Fluid density
    # Example: "density: 1000 kg/m³" or "ρ = 1050 kg/m3"
    FLUID_DENSITY = re.compile(
        r'(?:density|ρ|rho)[^\d]{0,10}(\d{3,4}\.?\d*)\s*(?:kg/m[³3]|kg/m\^3)',
        re.IGNORECASE
    )
    
    # Pattern: Viscosity
    # Example: "viscosity: 0.001 Pa·s" or "μ = 0.001 Pa.s"
    VISCOSITY = re.compile(
        r'(?:viscosity|μ|mu|η)[^\d]{0,10}(\d+\.?\d*(?:e-?\d+)?)\s*(?:Pa[·.]?s|mPa[·.]?s)',
        re.IGNORECASE
    )
    
    # Pattern: Temperature gradient
    # Example: "temperature gradient: 30°C/km" or "temp. grad. 35 °C/km"
    TEMP_GRADIENT = re.compile(
        r'(?:temperature\s+gradient|temp\.?\s+grad\.?)[^\d]{0,10}(\d{1,2}\.?\d*)\s*°?C/km',
        re.IGNORECASE
    )
    
    # ============================================================================
    # EQUIPMENT PATTERNS
    # ============================================================================
    
    # Pattern: Pump specifications
    PUMP_SPEC = re.compile(
        r'(?:pump|ESP)[^\n]{0,100}?(?:type|model)[^\n]{0,100}?([A-Z0-9-]+)',
        re.IGNORECASE
    )
    
    # Pattern: Wellhead pressure rating
    WELLHEAD_PRESSURE = re.compile(
        r'(?:wellhead|WH)[^\d]{0,50}(\d{2,4})\s*(?:bar|psi)',
        re.IGNORECASE
    )
    
    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    
    @staticmethod
    def parse_fractional_inches(whole: int, numerator: int, denominator: int) -> float:
        """
        Convert fractional inches to decimal
        Example: 13, 3, 8 -> 13.375
        """
        return whole + (numerator / denominator)
    
    @staticmethod
    def extract_trajectory_points(text: str) -> List[Dict[str, float]]:
        """
        Extract trajectory survey points from text using multiple pattern strategies
        
        Args:
            text: Text chunk potentially containing trajectory data
            
        Returns:
            List of dicts with keys: md, tvd, inclination
        """
        points = []
        
        # Try all trajectory patterns
        patterns = [
            PatternLibrary.TRAJECTORY_SPACE_SEPARATED,
            PatternLibrary.TRAJECTORY_PIPE_SEPARATED,
            PatternLibrary.TRAJECTORY_TAB_SEPARATED
        ]
        
        for pattern in patterns:
            matches = pattern.findall(text)
            if matches:
                for match in matches:
                    try:
                        md = float(match[0])
                        tvd = float(match[1])
                        inc = float(match[2])
                        
                        # Basic validation: MD >= TVD (allow 1m tolerance for rounding)
                        if md >= tvd - 1.0 and 0 <= inc <= 90:
                            points.append({
                                'md': md,
                                'tvd': tvd,
                                'inclination': inc
                            })
                    except (ValueError, IndexError):
                        continue
        
        # Remove duplicates (same MD)
        unique_points = []
        seen_mds = set()
        for point in points:
            md_rounded = round(point['md'], 1)
            if md_rounded not in seen_mds:
                unique_points.append(point)
                seen_mds.add(md_rounded)
        
        return sorted(unique_points, key=lambda x: x['md'])
    
    @staticmethod
    def extract_casing_design(text: str) -> List[Dict]:
        """
        Extract casing design information
        
        Returns:
            List of dicts with keys: od, top_md, bottom_md, id
        """
        casing_strings = []
        
        # Try fractional pattern
        matches = PatternLibrary.CASING_FRACTIONAL.findall(text)
        for match in matches:
            try:
                whole = int(match[0])
                numerator = int(match[1])
                denominator = int(match[2])
                od = PatternLibrary.parse_fractional_inches(whole, numerator, denominator)
                
                top_md = float(match[3])
                bottom_md = float(match[4])
                pipe_id = float(match[5])
                
                casing_strings.append({
                    'od': od,
                    'top_md': min(top_md, bottom_md),
                    'bottom_md': max(top_md, bottom_md),
                    'id': pipe_id
                })
            except (ValueError, IndexError):
                continue
        
        # Try decimal pattern
        matches = PatternLibrary.CASING_DECIMAL.findall(text)
        for match in matches:
            try:
                od = float(match[0])
                top_md = float(match[1])
                bottom_md = float(match[2])
                pipe_id = float(match[3])
                
                casing_strings.append({
                    'od': od,
                    'top_md': min(top_md, bottom_md),
                    'bottom_md': max(top_md, bottom_md),
                    'id': pipe_id
                })
            except (ValueError, IndexError):
                continue
        
        return sorted(casing_strings, key=lambda x: x['top_md'])
    
    @staticmethod
    def detect_content_type(text: str) -> str:
        """
        Classify chunk content type
        
        Returns:
            'trajectory', 'casing', 'pvt', 'equipment', or 'unknown'
        """
        text_lower = text.lower()
        
        # Check for trajectory keywords and patterns
        trajectory_keywords = ['md', 'tvd', 'inclination', 'survey', 'directional', 'measured depth']
        trajectory_score = sum(1 for kw in trajectory_keywords if kw in text_lower)
        has_trajectory_pattern = bool(PatternLibrary.TRAJECTORY_HEADER.search(text))
        
        if trajectory_score >= 2 or has_trajectory_pattern:
            return 'trajectory'
        
        # Check for casing keywords
        casing_keywords = ['casing', 'liner', 'tubing', 'pipe id', 'drift', 'tubular', 'schematic']
        casing_score = sum(1 for kw in casing_keywords if kw in text_lower)
        has_fractional = bool(re.search(r'\d+\s+\d+/\d+"', text))
        
        if casing_score >= 2 or has_fractional:
            return 'casing'
        
        # Check for PVT data
        pvt_keywords = ['density', 'viscosity', 'temperature gradient', 'pressure gradient', 'fluid properties']
        pvt_score = sum(1 for kw in pvt_keywords if kw in text_lower)
        
        if pvt_score >= 2:
            return 'pvt'
        
        # Check for equipment
        equipment_keywords = ['pump', 'esp', 'wellhead', 'flowline', 'equipment']
        equipment_score = sum(1 for kw in equipment_keywords if kw in text_lower)
        
        if equipment_score >= 2:
            return 'equipment'
        
        return 'unknown'
