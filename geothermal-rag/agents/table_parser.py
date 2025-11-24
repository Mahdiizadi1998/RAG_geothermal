"""
Table Parser - Intelligent parsing of well report tables
Identifies table types (casing, cementing, formations, etc.) and extracts structured data
"""

import re
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TableParser:
    """
    Parse well report tables and classify them by type
    
    Supported table types:
    - casing: Casing program specifications
    - cementing: Cement job details
    - formations: Formation tops and lithology
    - trajectory: Directional survey data
    - fluids: Drilling fluids/mud properties
    - operations: Time-based operations log
    """
    
    def __init__(self):
        # Keywords for table type identification
        self.table_type_keywords = {
            'casing': [
                'casing', 'pipe', 'tubular', 'string', 'od', 'id', 'weight', 'grade',
                'connection', 'shoe', 'liner', 'conductor'
            ],
            'cementing': [
                'cement', 'slurry', 'toc', 'top of cement', 'stage', 'plug',
                'cement volume', 'density', 'spacer'
            ],
            'formations': [
                'formation', 'lithology', 'top', 'base', 'depth', 'age',
                'stratigraphy', 'geology', 'layer', 'zone'
            ],
            'trajectory': [
                'survey', 'inclination', 'azimuth', 'md', 'tvd', 'north', 'east',
                'vertical section', 'dogleg', 'build rate'
            ],
            'fluids': [
                'mud', 'fluid', 'density', 'viscosity', 'ph', 'solids',
                'mud weight', 'circulation', 'loss'
            ],
            'operations': [
                'activity', 'operation', 'time', 'date', 'duration', 'hours',
                'start', 'end', 'status', 'npt'
            ]
        }
        
        # Patterns for extracting specific values
        self.depth_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*(?:m|ft)', re.IGNORECASE)
        self.diameter_pattern = re.compile(r'(\d+(?:\s+\d+/\d+)?)\s*(?:in|inch|")', re.IGNORECASE)
        self.weight_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*(?:lb/ft|kg/m|ppf)', re.IGNORECASE)
    
    def identify_table_type(self, headers: List[str], rows: List[List[str]], 
                           context: str = "") -> str:
        """
        Identify table type based on headers, content, and surrounding context
        
        Args:
            headers: Table column headers
            rows: Table data rows
            context: Surrounding text context
            
        Returns:
            Table type string ('casing', 'cementing', 'formations', etc.) or 'unknown'
        """
        # Combine headers and first few rows for analysis
        analysis_text = ' '.join(headers).lower()
        if rows:
            analysis_text += ' ' + ' '.join([' '.join(row[:5]) for row in rows[:3]]).lower()
        analysis_text += ' ' + context.lower()
        
        # Score each table type
        scores = {}
        for table_type, keywords in self.table_type_keywords.items():
            score = sum(1 for keyword in keywords if keyword in analysis_text)
            scores[table_type] = score
        
        # Return type with highest score
        if scores:
            best_type = max(scores, key=scores.get)
            if scores[best_type] >= 2:  # Minimum 2 keyword matches
                logger.debug(f"Identified table type: {best_type} (score: {scores[best_type]})")
                return best_type
        
        logger.debug(f"Unknown table type. Scores: {scores}")
        return 'unknown'
    
    def parse_casing_table(self, headers: List[str], rows: List[List[str]], 
                          page: int = None) -> List[Dict]:
        """
        Parse casing program table
        
        Expected columns (flexible order):
        - Size/OD/Diameter (inch)
        - Weight (lb/ft or kg/m)
        - Grade (L80, K55, etc.)
        - Connection type
        - Top/Bottom depth (m or ft)
        - Shoe depth
        """
        casing_strings = []
        
        # Normalize headers
        headers_lower = [h.lower().strip() for h in headers]
        
        # Find column indices
        size_col = self._find_column(headers_lower, ['size', 'od', 'diameter', 'casing'])
        weight_col = self._find_column(headers_lower, ['weight', 'lb/ft', 'kg/m'])
        grade_col = self._find_column(headers_lower, ['grade', 'material'])
        connection_col = self._find_column(headers_lower, ['connection', 'thread'])
        depth_col = self._find_column(headers_lower, ['depth', 'md', 'set', 'bottom'])
        shoe_col = self._find_column(headers_lower, ['shoe'])
        top_col = self._find_column(headers_lower, ['top', 'from'])
        
        for i, row in enumerate(rows, 1):
            if not row or not any(row):
                continue
            
            # Extract values
            size = self._extract_diameter(row[size_col] if size_col is not None else '')
            weight = self._extract_weight(row[weight_col] if weight_col is not None else '')
            grade = row[grade_col].strip() if grade_col is not None and grade_col < len(row) else None
            connection = row[connection_col].strip() if connection_col is not None and connection_col < len(row) else None
            bottom_depth = self._extract_depth(row[depth_col] if depth_col is not None and depth_col < len(row) else '')
            shoe_depth = self._extract_depth(row[shoe_col] if shoe_col is not None and shoe_col < len(row) else '')
            top_depth = self._extract_depth(row[top_col] if top_col is not None and top_col < len(row) else '')
            
            if size or bottom_depth:  # At least size or depth is required
                casing_strings.append({
                    'string_number': i,
                    'casing_type': 'casing',
                    'outer_diameter': size,
                    'diameter_unit': 'inch',
                    'weight': weight,
                    'weight_unit': 'lb/ft',
                    'grade': grade,
                    'connection_type': connection,
                    'top_depth_md': top_depth or 0,
                    'bottom_depth_md': bottom_depth,
                    'depth_unit': 'm',
                    'shoe_depth_md': shoe_depth or bottom_depth,
                    'shoe_depth_tvd': None,
                    'source_page': page,
                    'source_table': f'casing_table_p{page}'
                })
        
        return casing_strings
    
    def parse_formation_table(self, headers: List[str], rows: List[List[str]], 
                             page: int = None) -> List[Dict]:
        """Parse formation tops table"""
        formations = []
        
        headers_lower = [h.lower().strip() for h in headers]
        
        # Find columns
        name_col = self._find_column(headers_lower, ['formation', 'name', 'unit', 'layer'])
        top_md_col = self._find_column(headers_lower, ['top', 'depth', 'md'])
        top_tvd_col = self._find_column(headers_lower, ['tvd', 'true vertical'])
        lithology_col = self._find_column(headers_lower, ['lithology', 'rock', 'type'])
        age_col = self._find_column(headers_lower, ['age', 'period', 'epoch'])
        
        for row in rows:
            if not row or not any(row):
                continue
            
            name = row[name_col].strip() if name_col is not None and name_col < len(row) else None
            top_md = self._extract_depth(row[top_md_col] if top_md_col is not None and top_md_col < len(row) else '')
            top_tvd = self._extract_depth(row[top_tvd_col] if top_tvd_col is not None and top_tvd_col < len(row) else '')
            lithology = row[lithology_col].strip() if lithology_col is not None and lithology_col < len(row) else None
            age = row[age_col].strip() if age_col is not None and age_col < len(row) else None
            
            if name and top_md:
                formations.append({
                    'formation_name': name,
                    'top_md': top_md,
                    'top_tvd': top_tvd,
                    'bottom_md': None,
                    'bottom_tvd': None,
                    'lithology': lithology,
                    'age': age,
                    'source_page': page
                })
        
        return formations
    
    def _find_column(self, headers: List[str], keywords: List[str]) -> Optional[int]:
        """Find column index by matching keywords"""
        for i, header in enumerate(headers):
            if any(keyword in header for keyword in keywords):
                return i
        return None
    
    def _extract_depth(self, text: str) -> Optional[float]:
        """Extract depth value from text"""
        if not text:
            return None
        
        match = self.depth_pattern.search(text)
        if match:
            return float(match.group(1))
        
        # Try plain number
        try:
            return float(text.strip().replace(',', ''))
        except ValueError:
            return None
    
    def _extract_diameter(self, text: str) -> Optional[float]:
        """Extract diameter value (handle fractions like 13 3/8)"""
        if not text:
            return None
        
        # Handle fractions like "13 3/8"
        fraction_pattern = re.compile(r'(\d+)\s+(\d+)/(\d+)')
        match = fraction_pattern.search(text)
        if match:
            whole = float(match.group(1))
            num = float(match.group(2))
            denom = float(match.group(3))
            return whole + (num / denom)
        
        # Regular number with unit
        match = self.diameter_pattern.search(text)
        if match:
            return float(match.group(1))
        
        # Plain number
        try:
            return float(text.strip().replace(',', ''))
        except ValueError:
            return None
    
    def _extract_weight(self, text: str) -> Optional[float]:
        """Extract weight value"""
        if not text:
            return None
        
        match = self.weight_pattern.search(text)
        if match:
            return float(match.group(1))
        
        try:
            return float(text.strip().replace(',', ''))
        except ValueError:
            return None
