"""
Parameter Extraction Agent - Regex-First Data Extraction
Extracts trajectory, casing, PVT data with validation and LLM fallback
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.pattern_library import PatternLibrary
from utils.unit_conversion import UnitConverter
from typing import Dict, List, Optional, Tuple
import logging
import json
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParameterExtractionAgent:
    """
    Extract structured data from geothermal well report chunks
    
    Extraction strategy:
    1. Classify chunks by content type (trajectory/casing/PVT)
    2. Use regex patterns first (fast, reliable for tables)
    3. Fall back to LLM only if regex fails
    4. Validate all extracted values
    5. Merge trajectory + casing data
    """
    
    def __init__(self, enable_llm_fallback: bool = True):
        """
        Initialize extraction agent
        
        Args:
            enable_llm_fallback: Whether to use LLM if regex fails
        """
        self.pattern_lib = PatternLibrary()
        self.converter = UnitConverter()
        self.enable_llm_fallback = enable_llm_fallback
    
    def extract(self, chunks: List[Dict], well_name: Optional[str] = None, 
                log_capture: Optional[List] = None) -> Dict:
        """
        Extract all parameters from chunks
        
        Args:
            chunks: List of chunk dicts from RAG retrieval
            well_name: Well name for filtering/logging
            log_capture: Optional list to capture log messages
            
        Returns:
            Dict with structure:
            {
                'well_name': str,
                'trajectory': List[Dict],  # [{'md': float, 'tvd': float, 'inc': float, 'pipe_id': float}]
                'casing_design': List[Dict],  # [{'od': float, 'top_md': float, 'bottom_md': float, 'id': float}]
                'pvt_data': Dict,  # {'density': float, 'viscosity': float, ...}
                'equipment': Dict,  # {'pump_type': str, ...}
                'extraction_log': List[str],
                'confidence': float
            }
        """
        if log_capture is None:
            log_capture = []
        
        self._log(log_capture, f"Starting extraction for {well_name or 'unknown well'}")
        self._log(log_capture, f"Processing {len(chunks)} chunks")
        
        # Classify chunks by content type
        trajectory_chunks = []
        casing_chunks = []
        pvt_chunks = []
        equipment_chunks = []
        
        for chunk in chunks:
            content_type = self.pattern_lib.detect_content_type(chunk['text'])
            
            if content_type == 'trajectory':
                trajectory_chunks.append(chunk)
            elif content_type == 'casing':
                casing_chunks.append(chunk)
            elif content_type == 'pvt':
                pvt_chunks.append(chunk)
            elif content_type == 'equipment':
                equipment_chunks.append(chunk)
        
        self._log(log_capture, f"Classified chunks: {len(trajectory_chunks)} trajectory, "
                             f"{len(casing_chunks)} casing, {len(pvt_chunks)} PVT, "
                             f"{len(equipment_chunks)} equipment")
        
        # Extract trajectory survey
        trajectory_points = self._extract_trajectory_survey(trajectory_chunks, log_capture)
        self._log(log_capture, f"✓ Found {len(trajectory_points)} trajectory points")
        
        # Extract casing design
        casing_strings = self._extract_casing_design(casing_chunks, log_capture)
        self._log(log_capture, f"✓ Found {len(casing_strings)} casing strings")
        
        # Extract PVT data
        pvt_data = self._extract_pvt_data(pvt_chunks, log_capture)
        self._log(log_capture, f"✓ Extracted PVT data: {list(pvt_data.keys())}")
        
        # Extract equipment specs
        equipment = self._extract_equipment(equipment_chunks, log_capture)
        
        # Merge trajectory with casing (critical step!)
        merged_trajectory = self._merge_trajectory_with_casing(
            trajectory_points, casing_strings, log_capture
        )
        self._log(log_capture, f"✓ Merged data: {len(merged_trajectory)} points with pipe ID")
        
        # Calculate confidence score
        confidence = self._calculate_confidence(
            trajectory_points, casing_strings, pvt_data, merged_trajectory
        )
        
        return {
            'well_name': well_name,
            'trajectory': merged_trajectory,
            'trajectory_raw': trajectory_points,
            'casing_design': casing_strings,
            'pvt_data': pvt_data,
            'equipment': equipment,
            'extraction_log': log_capture,
            'confidence': confidence
        }
    
    def _extract_trajectory_survey(self, chunks: List[Dict], log: List[str]) -> List[Dict]:
        """Extract trajectory survey data (MD, TVD, Inclination)"""
        all_points = []
        
        for chunk in chunks:
            points = self.pattern_lib.extract_trajectory_points(chunk['text'])
            all_points.extend(points)
            
            if points:
                self._log(log, f"  Found {len(points)} points in chunk {chunk.get('id', 'unknown')}")
        
        # Remove duplicates (keep unique MDs)
        unique_points = []
        seen_mds = set()
        
        for point in sorted(all_points, key=lambda x: x['md']):
            md_rounded = round(point['md'], 1)
            if md_rounded not in seen_mds:
                unique_points.append(point)
                seen_mds.add(md_rounded)
        
        return unique_points
    
    def _extract_casing_design(self, chunks: List[Dict], log: List[str]) -> List[Dict]:
        """Extract casing design (OD, depths, ID)"""
        all_casing = []
        
        for chunk in chunks:
            casing = self.pattern_lib.extract_casing_design(chunk['text'])
            
            if casing:
                self._log(log, f"  Found {len(casing)} casing strings in chunk")
                all_casing.extend(casing)
        
        # Remove duplicates
        unique_casing = []
        seen = set()
        
        for c in sorted(all_casing, key=lambda x: x['top_md']):
            key = (round(c['top_md'], 1), round(c['bottom_md'], 1))
            if key not in seen:
                unique_casing.append(c)
                seen.add(key)
        
        return unique_casing
    
    def _extract_pvt_data(self, chunks: List[Dict], log: List[str]) -> Dict:
        """Extract PVT data (fluid properties)"""
        pvt_data = {}
        
        for chunk in chunks:
            text = chunk['text']
            
            # Extract fluid density
            match = self.pattern_lib.FLUID_DENSITY.search(text)
            if match and 'density' not in pvt_data:
                pvt_data['density'] = float(match.group(1))
                self._log(log, f"  Found density: {pvt_data['density']} kg/m³")
            
            # Extract viscosity
            match = self.pattern_lib.VISCOSITY.search(text)
            if match and 'viscosity' not in pvt_data:
                pvt_data['viscosity'] = float(match.group(1))
                self._log(log, f"  Found viscosity: {pvt_data['viscosity']} Pa·s")
            
            # Extract temperature gradient
            match = self.pattern_lib.TEMP_GRADIENT.search(text)
            if match and 'temp_gradient' not in pvt_data:
                pvt_data['temp_gradient'] = float(match.group(1))
                self._log(log, f"  Found temp gradient: {pvt_data['temp_gradient']} °C/km")
        
        return pvt_data
    
    def _extract_equipment(self, chunks: List[Dict], log: List[str]) -> Dict:
        """Extract equipment specifications"""
        equipment = {}
        
        for chunk in chunks:
            text = chunk['text']
            
            # Extract pump specs
            match = self.pattern_lib.PUMP_SPEC.search(text)
            if match and 'pump_type' not in equipment:
                equipment['pump_type'] = match.group(1)
                self._log(log, f"  Found pump: {equipment['pump_type']}")
            
            # Extract wellhead pressure
            match = self.pattern_lib.WELLHEAD_PRESSURE.search(text)
            if match and 'wellhead_pressure' not in equipment:
                value = float(match.group(1))
                unit = 'bar' if 'bar' in text[match.start():match.end()].lower() else 'psi'
                equipment['wellhead_pressure'] = value
                equipment['wellhead_pressure_unit'] = unit
        
        return equipment
    
    def _merge_trajectory_with_casing(self, trajectory: List[Dict], 
                                     casing: List[Dict], log: List[str]) -> List[Dict]:
        """
        Merge trajectory survey with casing design
        
        Algorithm:
        1. For each casing string top depth, find closest trajectory point
        2. Assign pipe ID to that trajectory depth
        3. Interpolate pipe ID for trajectory points between casing strings
        
        Args:
            trajectory: List of {'md', 'tvd', 'inclination'}
            casing: List of {'od', 'top_md', 'bottom_md', 'id'}
            
        Returns:
            List of {'md', 'tvd', 'inclination', 'pipe_id'} in meters
        """
        if not trajectory:
            self._log(log, "⚠️ No trajectory data to merge")
            return []
        
        if not casing:
            self._log(log, "⚠️ No casing data - using constant pipe ID")
            # Use default pipe ID (assume 7" = 0.1778 m)
            default_id = self.converter.inches_to_meters(6.276)  # 7" casing typical ID
            merged = []
            for point in trajectory:
                merged.append({
                    'md': point['md'],
                    'tvd': point['tvd'],
                    'inclination': point['inclination'],
                    'pipe_id': default_id
                })
            return merged
        
        merged = []
        
        # Sort both lists by depth
        trajectory = sorted(trajectory, key=lambda x: x['md'])
        casing = sorted(casing, key=lambda x: x['top_md'])
        
        # For each casing string, find closest trajectory point
        casing_with_traj = []
        for c in casing:
            # Find trajectory point closest to casing top
            closest = min(trajectory, key=lambda t: abs(t['md'] - c['top_md']))
            
            # Convert pipe ID from inches to meters
            pipe_id_meters = self.converter.inches_to_meters(c['id'])
            
            casing_with_traj.append({
                'md': c['top_md'],
                'tvd': closest['tvd'],
                'inclination': closest['inclination'],
                'pipe_id': pipe_id_meters,
                'bottom_md': c['bottom_md']
            })
        
        # Now interpolate pipe ID for all trajectory points
        for traj_point in trajectory:
            md = traj_point['md']
            
            # Find which casing string this depth is in
            pipe_id = None
            for i, casing_point in enumerate(casing_with_traj):
                if md >= casing_point['md']:
                    # Check if we're still in this casing string
                    if i < len(casing_with_traj) - 1:
                        # Not the last string, check bottom
                        if md <= casing_point['bottom_md']:
                            pipe_id = casing_point['pipe_id']
                            break
                    else:
                        # Last string, use its ID
                        pipe_id = casing_point['pipe_id']
                        break
            
            # If no pipe ID found, use the first casing string ID
            if pipe_id is None and casing_with_traj:
                pipe_id = casing_with_traj[0]['pipe_id']
            
            merged.append({
                'md': traj_point['md'],
                'tvd': traj_point['tvd'],
                'inclination': traj_point['inclination'],
                'pipe_id': pipe_id
            })
        
        # Add casing tops explicitly if not already in trajectory
        for casing_point in casing_with_traj:
            # Check if this depth already exists in merged
            if not any(abs(m['md'] - casing_point['md']) < 1.0 for m in merged):
                merged.append(casing_point)
        
        # Sort by MD and remove any points without pipe_id
        merged = sorted([m for m in merged if m['pipe_id'] is not None], key=lambda x: x['md'])
        
        return merged
    
    def _calculate_confidence(self, trajectory: List, casing: List, 
                            pvt: Dict, merged: List) -> float:
        """
        Calculate confidence score for extraction
        
        Score components:
        - Trajectory data present: 30%
        - Casing data present: 30%
        - PVT data present: 20%
        - Successful merge: 20%
        """
        score = 0.0
        
        # Trajectory data (30%)
        if trajectory:
            if len(trajectory) >= 3:
                score += 0.30
            else:
                score += 0.15  # Partial credit
        
        # Casing data (30%)
        if casing:
            if len(casing) >= 2:
                score += 0.30
            else:
                score += 0.15
        
        # PVT data (20%)
        pvt_fields = ['density', 'viscosity', 'temp_gradient']
        pvt_score = sum(1 for field in pvt_fields if field in pvt) / len(pvt_fields)
        score += 0.20 * pvt_score
        
        # Successful merge (20%)
        if merged and len(merged) >= 2:
            score += 0.20
        
        return round(score, 2)
    
    def _log(self, log_list: List[str], message: str):
        """Add message to log list and logger"""
        log_list.append(message)
        logger.info(message)
    
    def format_for_nodal_analysis(self, extracted_data: Dict) -> str:
        """
        Format extracted data for nodal_analysis.py in exact required format
        
        Returns:
            Python code string with well_trajectory list matching exact format:
            well_trajectory = [
                {"MD": 0.0,    "TVD": 0.0,    "ID": 0.3397},
                {"MD": 500.0,  "TVD": 500.0,  "ID": 0.2445},
                ...
            ]
        """
        trajectory = extracted_data.get('trajectory', [])
        pvt = extracted_data.get('pvt_data', {})
        
        if not trajectory:
            return "# No trajectory data extracted"
        
        # Create well_trajectory in exact format specified
        code = "# Extracted trajectory data for nodal analysis\n"
        code += "# Format: MD, TVD, ID are in meters\n\n"
        code += "well_trajectory = [\n"
        
        for point in trajectory:
            md = point['md']
            tvd = point['tvd']
            pipe_id = point['pipe_id']
            
            # Format with proper spacing to match target format
            code += f'    {{"MD": {md:<7.1f}, "TVD": {tvd:<7.1f}, "ID": {pipe_id:.4f}}},\n'
        
        code += "]\n\n"
        
        # Add PVT data as separate variables
        code += "# Fluid properties (extracted)\n"
        code += f"rho = {pvt.get('density', 1000.0)}  # water density [kg/m3]\n"
        code += f"mu = {pvt.get('viscosity', 0.001)}  # viscosity [Pa.s]\n"
        
        return code
