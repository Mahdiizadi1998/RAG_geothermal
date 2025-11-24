"""
Enhanced Table Parser - Comprehensive extraction of all well data
Extracts: General Data, Timeline, Depths, Casing (with Pipe IDs), Cementing, Fluids
Preserves exact table structure from PDF
"""

import re
from typing import Dict, List, Optional, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedTableParser:
    """
    Comprehensive parser for all well report tables
    Focus: Tabular data only (NO geological/technical logs)
    
    Extracts:
    1. General Data: License, Well Type, Coordinates, Rig Name, Target Formation
    2. Timeline: Spud Date, End of Operations, Total Days
    3. Depths: TD (MD), TVD, Sidetrack Start Depth
    4. Casing: OD, Weight, Grade, Connection, Pipe ID (Nominal + Drift), Depths
    5. Cementing: Lead/Tail volumes, Densities, TOC
    6. Fluids: Hole Size, Fluid Type, Density Range
    """
    
    def __init__(self):
        # Table type identification keywords
        self.table_type_keywords = {
            'general_data': [
                'license', 'operator', 'rig', 'target', 'well type', 'location',
                'coordinate', 'latitude', 'longitude', 'x', 'y', 'spud', 'completion'
            ],
            'casing': [
                'casing', 'pipe', 'tubular', 'string', 'od', 'id', 'drift', 'nominal',
                'weight', 'grade', 'connection', 'shoe', 'liner', 'conductor'
            ],
            'cementing': [
                'cement', 'slurry', 'toc', 'top of cement', 'stage', 'plug',
                'lead', 'tail', 'density', 'spacer', 'volume'
            ],
            'fluids': [
                'mud', 'fluid', 'hole size', 'density', 'viscosity', 'ph',
                'mud weight', 'circulation', 'loss', 'drilling fluid'
            ],
            'formations': [
                'formation', 'lithology', 'top', 'base', 'depth', 'age',
                'stratigraphy', 'geology', 'layer', 'zone'
            ]
        }
        
        # Exclusion keywords for geological/technical logs (IGNORE THESE)
        self.exclude_keywords = [
            'geological', 'litholog', 'mud log', 'gas show', 'chromatograph',
            'sample', 'description', 'cutting', 'core', 'log interpretation',
            'wireline', 'petrophysical', 'reservoir evaluation'
        ]
        
        # Regex patterns for value extraction
        self.depth_pattern = re.compile(r'(\d+(?:[.,]\d+)?)\s*(?:m|ft|meter|feet)?', re.IGNORECASE)
        self.diameter_pattern = re.compile(r'(\d+(?:\s+\d+/\d+)?)\s*(?:in|inch|"|mm)?', re.IGNORECASE)
        self.weight_pattern = re.compile(r'(\d+(?:[.,]\d+)?)\s*(?:lb/ft|kg/m|ppf|#/ft)', re.IGNORECASE)
        self.fraction_pattern = re.compile(r'(\d+)\s+(\d+)/(\d+)')
        self.date_pattern = re.compile(r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})|(\d{4}[-/]\d{1,2}[-/]\d{1,2})')
        self.coordinate_pattern = re.compile(r'(\d+(?:[.,]\d+)?)\s*(?:E|N|W|S)?', re.IGNORECASE)
        
    def should_exclude_table(self, headers: List[str], context: str = "") -> bool:
        """Check if table should be excluded (geological/technical logs)"""
        text = ' '.join(headers).lower() + ' ' + context.lower()
        return any(keyword in text for keyword in self.exclude_keywords)
    
    def identify_table_type(self, headers: List[str], rows: List[List[str]], 
                           context: str = "") -> str:
        """
        Identify table type based on headers and content
        Returns: 'general_data', 'casing', 'cementing', 'fluids', 'formations', or 'unknown'
        """
        # Check exclusions first
        if self.should_exclude_table(headers, context):
            logger.debug("Table excluded: geological/technical log detected")
            return 'excluded'
        
        # Combine text for analysis
        analysis_text = ' '.join(headers).lower()
        if rows:
            analysis_text += ' ' + ' '.join([' '.join(row[:5]) for row in rows[:3]]).lower()
        analysis_text += ' ' + context.lower()
        
        # Score each table type
        scores = {}
        for table_type, keywords in self.table_type_keywords.items():
            score = sum(1 for keyword in keywords if keyword in analysis_text)
            scores[table_type] = score
        
        # Return highest scoring type
        if scores:
            best_type = max(scores, key=scores.get)
            if scores[best_type] >= 2:
                logger.debug(f"Identified: {best_type} (score: {scores[best_type]})")
                return best_type
        
        return 'unknown'
    
    def parse_general_data_table(self, headers: List[str], rows: List[List[str]], 
                                 page: int = None) -> Dict[str, Any]:
        """
        Parse general well data table (often key-value pairs)
        Extracts: License, Well Type, Coordinates, Rig Name, Target Formation, Dates, etc.
        """
        data = {
            'source_page': page
        }
        
        # Normalize headers
        headers_lower = [h.lower().strip() for h in headers]
        
        # Try to find key-value pairs
        for row in rows:
            if len(row) < 2 or not any(row):
                continue
            
            key = row[0].lower().strip()
            value = row[1].strip() if len(row) > 1 else None
            
            if not value:
                continue
            
            # License
            if 'license' in key or 'permit' in key:
                data['license_number'] = value
            
            # Well Type
            elif 'well type' in key or 'type' in key:
                data['well_type'] = value
            
            # Operator
            elif 'operator' in key or 'company' in key:
                data['operator'] = value
            
            # Rig Name
            elif 'rig' in key or 'drilling rig' in key:
                data['rig_name'] = value
            
            # Target Formation
            elif 'target' in key or 'objective' in key:
                data['target_formation'] = value
            
            # Location
            elif 'location' in key or 'field' in key:
                data['location'] = value
            
            # Coordinates
            elif 'x' == key or 'easting' in key or 'x coord' in key:
                data['coordinate_x'] = self._extract_number(value)
            elif 'y' == key or 'northing' in key or 'y coord' in key:
                data['coordinate_y'] = self._extract_number(value)
            elif 'coordinate system' in key or 'datum' in key:
                data['coordinate_system'] = value
            
            # Dates
            elif 'spud' in key:
                data['spud_date'] = self._extract_date(value)
            elif 'completion' in key or 'completed' in key:
                data['completion_date'] = self._extract_date(value)
            elif 'end of operations' in key or 'eoo' in key:
                data['end_of_operations'] = self._extract_date(value)
            elif 'total days' in key or 'duration' in key:
                data['total_days'] = self._extract_integer(value)
            
            # Depths
            elif 'td' in key or 'total depth' in key:
                if 'tvd' in key:
                    data['total_depth_tvd'] = self._extract_depth(value)
                else:
                    data['total_depth_md'] = self._extract_depth(value)
            elif 'sidetrack' in key and 'depth' in key:
                data['sidetrack_start_depth_md'] = self._extract_depth(value)
        
        return data if len(data) > 1 else {}  # More than just source_page
    
    def parse_casing_table(self, headers: List[str], rows: List[List[str]], 
                          page: int = None) -> List[Dict]:
        """
        Parse casing program table - COMPREHENSIVE
        Extracts: Type, OD, Weight, Grade, Connection, Pipe ID (Nominal + Drift), Depths
        """
        casing_strings = []
        
        headers_lower = [h.lower().strip() for h in headers]
        
        # Find column indices (flexible matching)
        type_col = self._find_column(headers_lower, ['type', 'casing type', 'string'])
        size_col = self._find_column(headers_lower, ['size', 'od', 'outer diameter', 'diameter', 'casing od'])
        
        # Pipe ID matching - EXACT match for "pipe id" to avoid confusion with other ID columns
        id_nominal_col = self._find_exact_column(headers_lower, 'pipe id')
        if id_nominal_col is None:
            # Fallback to other ID column names
            id_nominal_col = self._find_column(headers_lower, ['nominal id', 'inner diameter', 'internal diameter'])
        
        id_drift_col = self._find_column(headers_lower, ['drift', 'drift id', 'drift diameter'])
        weight_col = self._find_column(headers_lower, ['weight', 'lb/ft', 'kg/m', '#/ft', 'ppf'])
        grade_col = self._find_column(headers_lower, ['grade', 'material', 'steel grade'])
        connection_col = self._find_column(headers_lower, ['connection', 'thread', 'coupling'])
        top_col = self._find_column(headers_lower, ['top', 'from', 'top depth', 'top md'])
        bottom_col = self._find_column(headers_lower, ['bottom', 'to', 'set', 'bottom depth', 'shoe depth', 'md'])
        
        for i, row in enumerate(rows, 1):
            if not row or not any(row):
                continue
            
            # Skip header rows within data
            if any(h in row[0].lower() for h in ['type', 'casing', 'size', 'od']):
                continue
            
            # Extract values
            casing_type = self._get_cell(row, type_col)
            od = self._extract_diameter(self._get_cell(row, size_col))
            id_nominal = self._extract_diameter(self._get_cell(row, id_nominal_col))
            id_drift = self._extract_diameter(self._get_cell(row, id_drift_col))
            weight = self._extract_weight(self._get_cell(row, weight_col))
            grade = self._get_cell(row, grade_col)
            connection = self._get_cell(row, connection_col)
            top_depth = self._extract_depth(self._get_cell(row, top_col))
            bottom_depth = self._extract_depth(self._get_cell(row, bottom_col))
            
            # Require at least OD or bottom depth
            if od or bottom_depth:
                casing_strings.append({
                    'string_number': i,
                    'casing_type': casing_type or 'casing',
                    'outer_diameter': od,
                    'diameter_unit': 'inch',
                    'pipe_id_nominal': id_nominal,
                    'pipe_id_drift': id_drift,
                    'id_unit': 'inch',
                    'weight': weight,
                    'weight_unit': 'lb/ft',
                    'grade': grade,
                    'connection_type': connection,
                    'top_depth_md': top_depth or 0,
                    'bottom_depth_md': bottom_depth,
                    'depth_unit': 'm',
                    'shoe_depth_md': bottom_depth,
                    'shoe_depth_tvd': None,
                    'source_page': page,
                    'source_table': f'casing_table_p{page}'
                })
        
        return casing_strings
    
    def parse_cementing_table(self, headers: List[str], rows: List[List[str]], 
                             page: int = None) -> List[Dict]:
        """
        Parse cementing table - COMPREHENSIVE
        Extracts: Stage, Lead/Tail volumes, Lead/Tail densities, TOC (MD + TVD)
        """
        cement_jobs = []
        
        headers_lower = [h.lower().strip() for h in headers]
        
        # Find columns
        stage_col = self._find_column(headers_lower, ['stage', 'job', 'number'])
        type_col = self._find_column(headers_lower, ['type', 'cement type', 'slurry'])
        lead_vol_col = self._find_column(headers_lower, ['lead', 'lead volume', 'lead vol'])
        lead_dens_col = self._find_column(headers_lower, ['lead density', 'lead dens'])
        tail_vol_col = self._find_column(headers_lower, ['tail', 'tail volume', 'tail vol'])
        tail_dens_col = self._find_column(headers_lower, ['tail density', 'tail dens'])
        toc_md_col = self._find_column(headers_lower, ['toc', 'top of cement', 'toc md'])
        toc_tvd_col = self._find_column(headers_lower, ['toc tvd', 'toc (tvd)'])
        volume_col = self._find_column(headers_lower, ['volume', 'total volume', 'vol'])
        density_col = self._find_column(headers_lower, ['density', 'dens', 'sg'])
        
        for i, row in enumerate(rows, 1):
            if not row or not any(row):
                continue
            
            stage = self._extract_integer(self._get_cell(row, stage_col)) or i
            cement_type = self._get_cell(row, type_col)
            lead_volume = self._extract_number(self._get_cell(row, lead_vol_col))
            lead_density = self._extract_number(self._get_cell(row, lead_dens_col))
            tail_volume = self._extract_number(self._get_cell(row, tail_vol_col))
            tail_density = self._extract_number(self._get_cell(row, tail_dens_col))
            toc_md = self._extract_depth(self._get_cell(row, toc_md_col))
            toc_tvd = self._extract_depth(self._get_cell(row, toc_tvd_col))
            volume = self._extract_number(self._get_cell(row, volume_col))
            density = self._extract_number(self._get_cell(row, density_col))
            
            # Require at least stage or TOC
            if stage or toc_md:
                cement_jobs.append({
                    'stage_number': stage,
                    'cement_type': cement_type,
                    'lead_volume': lead_volume,
                    'lead_density': lead_density,
                    'tail_volume': tail_volume,
                    'tail_density': tail_density,
                    'top_of_cement_md': toc_md,
                    'toc_tvd': toc_tvd,
                    'volume': volume,
                    'volume_unit': 'm3',
                    'density': density,
                    'density_unit': 'kg/m3',
                    'source_page': page
                })
        
        return cement_jobs
    
    def parse_fluids_table(self, headers: List[str], rows: List[List[str]], 
                          page: int = None) -> List[Dict]:
        """
        Parse drilling fluids table
        Extracts: Hole Size, Fluid Type, Density Min/Max (Range)
        """
        fluids = []
        
        headers_lower = [h.lower().strip() for h in headers]
        
        # Find columns
        hole_size_col = self._find_column(headers_lower, ['hole', 'hole size', 'section', 'bit size'])
        fluid_type_col = self._find_column(headers_lower, ['type', 'fluid type', 'mud type', 'system'])
        density_col = self._find_column(headers_lower, ['density', 'mud weight', 'mw', 'sg'])
        density_min_col = self._find_column(headers_lower, ['min', 'minimum', 'from'])
        density_max_col = self._find_column(headers_lower, ['max', 'maximum', 'to'])
        depth_from_col = self._find_column(headers_lower, ['depth from', 'from depth', 'top'])
        depth_to_col = self._find_column(headers_lower, ['depth to', 'to depth', 'bottom'])
        
        for row in rows:
            if not row or not any(row):
                continue
            
            hole_size = self._extract_diameter(self._get_cell(row, hole_size_col))
            fluid_type = self._get_cell(row, fluid_type_col)
            
            # Handle density range (min/max)
            density_str = self._get_cell(row, density_col)
            density_min = self._extract_number(self._get_cell(row, density_min_col))
            density_max = self._extract_number(self._get_cell(row, density_max_col))
            
            # Try to extract range from single density column (e.g., "1000-1200")
            if density_str and '-' in density_str and not density_min:
                parts = density_str.split('-')
                if len(parts) == 2:
                    density_min = self._extract_number(parts[0])
                    density_max = self._extract_number(parts[1])
            elif density_str and not density_min:
                # Single value - use as both min and max
                val = self._extract_number(density_str)
                density_min = val
                density_max = val
            
            depth_from = self._extract_depth(self._get_cell(row, depth_from_col))
            depth_to = self._extract_depth(self._get_cell(row, depth_to_col))
            
            if hole_size or fluid_type:
                fluids.append({
                    'hole_size': hole_size,
                    'hole_size_unit': 'inch',
                    'fluid_type': fluid_type,
                    'density_min': density_min,
                    'density_max': density_max,
                    'density_unit': 'kg/m3',
                    'depth_interval_from': depth_from,
                    'depth_interval_to': depth_to,
                    'depth_unit': 'm',
                    'source_page': page
                })
        
        return fluids
    
    def parse_formation_table(self, headers: List[str], rows: List[List[str]], 
                             page: int = None) -> List[Dict]:
        """Parse formation tops table"""
        formations = []
        
        headers_lower = [h.lower().strip() for h in headers]
        
        name_col = self._find_column(headers_lower, ['formation', 'name', 'unit', 'layer'])
        top_md_col = self._find_column(headers_lower, ['top', 'depth', 'md', 'top md'])
        top_tvd_col = self._find_column(headers_lower, ['tvd', 'true vertical', 'top tvd'])
        lithology_col = self._find_column(headers_lower, ['lithology', 'rock', 'type'])
        age_col = self._find_column(headers_lower, ['age', 'period', 'epoch'])
        
        for row in rows:
            if not row or not any(row):
                continue
            
            name = self._get_cell(row, name_col)
            top_md = self._extract_depth(self._get_cell(row, top_md_col))
            top_tvd = self._extract_depth(self._get_cell(row, top_tvd_col))
            lithology = self._get_cell(row, lithology_col)
            age = self._get_cell(row, age_col)
            
            if name and top_md:
                formations.append({
                    'formation_name': name,
                    'top_md': top_md,
                    'top_tvd': top_tvd,
                    'lithology': lithology,
                    'age': age,
                    'source_page': page
                })
        
        return formations
    
    # ========== Helper Methods ==========
    
    def _find_column(self, headers: List[str], keywords: List[str]) -> Optional[int]:
        """Find column index by matching keywords"""
        for i, header in enumerate(headers):
            if any(keyword in header for keyword in keywords):
                return i
        return None
    
    def _find_exact_column(self, headers: List[str], exact_match: str) -> Optional[int]:
        """Find column index by exact match (case-insensitive, strips whitespace)"""
        exact_match = exact_match.lower().strip()
        for i, header in enumerate(headers):
            # Clean header: remove units, extra spaces
            clean_header = header.lower().strip()
            # Remove common unit suffixes in parentheses or brackets
            clean_header = re.sub(r'\s*[\(\[].*?[\)\]]', '', clean_header)
            clean_header = clean_header.strip()
            
            if clean_header == exact_match:
                return i
        return None
    
    def _get_cell(self, row: List[str], col_idx: Optional[int]) -> str:
        """Safely get cell value"""
        if col_idx is not None and col_idx < len(row):
            return row[col_idx].strip()
        return ''
    
    def _extract_depth(self, text: str) -> Optional[float]:
        """Extract depth value from text"""
        if not text:
            return None
        match = self.depth_pattern.search(text)
        if match:
            return float(match.group(1).replace(',', '.'))
        return None
    
    def _extract_diameter(self, text: str) -> Optional[float]:
        """Extract diameter, handling fractions (e.g., 13 3/8)"""
        if not text:
            return None
        
        # Check for fractions first
        fraction_match = self.fraction_pattern.search(text)
        if fraction_match:
            whole = int(fraction_match.group(1))
            numerator = int(fraction_match.group(2))
            denominator = int(fraction_match.group(3))
            return whole + (numerator / denominator)
        
        # Standard decimal
        match = self.diameter_pattern.search(text)
        if match:
            value = match.group(1).strip()
            return float(value.replace(',', '.'))
        return None
    
    def _extract_weight(self, text: str) -> Optional[float]:
        """Extract weight value"""
        if not text:
            return None
        match = self.weight_pattern.search(text)
        if match:
            return float(match.group(1).replace(',', '.'))
        return None
    
    def _extract_number(self, text: str) -> Optional[float]:
        """Extract any numerical value"""
        if not text:
            return None
        # Remove non-numeric except decimal point and comma
        cleaned = re.sub(r'[^\d.,]', '', text)
        cleaned = cleaned.replace(',', '.')
        try:
            return float(cleaned) if cleaned else None
        except ValueError:
            return None
    
    def _extract_integer(self, text: str) -> Optional[int]:
        """Extract integer value"""
        if not text:
            return None
        match = re.search(r'\d+', text)
        if match:
            return int(match.group())
        return None
    
    def _extract_date(self, text: str) -> Optional[str]:
        """Extract date value"""
        if not text:
            return None
        match = self.date_pattern.search(text)
        if match:
            return match.group(0)
        return None
